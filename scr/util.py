"""Losses and metrics. Graph negatives must be re-encoded corrupted inputs."""
import random
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate_metrics(labels, probabilities, threshold=0.5):
    y, p = np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("Nonempty labels and probabilities must have the same 1-D shape")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Use 0/1 labels and finite positive-class probabilities in [0,1]")
    return {"AUC": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
            "AUPR": float(average_precision_score(y, p)) if y.any() else 0.0,
            "Binary-F1": float(f1_score(y, p >= threshold, zero_division=0))}


def node_contrastive_loss(z1, z2, temperature=0.2, max_nodes=512):
    if z1.shape != z2.shape or len(z1) < 2:
        raise ValueError("Node contrastive learning requires at least two aligned nodes")
    if temperature <= 0 or max_nodes < 2:
        raise ValueError("temperature > 0 and max_nodes >= 2 are required")
    if len(z1) > max_nodes:
        indices = torch.randperm(len(z1), device=z1.device)[:max_nodes]
        z1, z2 = z1[indices], z2[indices]
    logits = F.normalize(z1, dim=-1) @ F.normalize(z2, dim=-1).T / temperature
    target = torch.arange(len(z1), device=z1.device)
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2


def graph_contrastive_loss(z1, z2, negative_views, temperature=0.2):
    """Symmetric InfoNCE against genuinely corrupted graph encodings.

    `negative_views` is a nonempty sequence of node-embedding matrices obtained
    by encoding corrupted raw inputs/edges, never a row permutation of z2.
    No detach is used: the shared encoder receives positive and negative gradients.
    """
    if temperature <= 0 or not negative_views:
        raise ValueError("Provide temperature > 0 and at least one encoded negative graph")
    if not len(z1) or not len(z2) or any(not len(z) for z in negative_views):
        raise ValueError("Graph readout requires nonempty node sets")
    readout = lambda z: F.normalize(z.mean(0, keepdim=True), dim=-1)
    g1, g2 = readout(z1), readout(z2)
    negatives = torch.cat([readout(z) for z in negative_views], dim=0)
    positive = (g1 * g2).sum(-1, keepdim=True)
    target = torch.zeros(1, device=z1.device, dtype=torch.long)
    first = torch.cat((positive, g1 @ negatives.T), -1) / temperature
    second = torch.cat((positive, g2 @ negatives.T), -1) / temperature
    return (F.cross_entropy(first, target) + F.cross_entropy(second, target)) / 2


def intent_decorrelation_loss(intent_reps):
    if not intent_reps:
        raise ValueError("Empty intents: skip decorrelation when the hypergraph is removed")
    loss = intent_reps[0].sum() * 0
    for k in range(len(intent_reps)):
        for other in range(k + 1, len(intent_reps)):
            # Column unit norms give an explicit, size-stable matrix normalization.
            a, b = F.normalize(intent_reps[k], dim=0), F.normalize(intent_reps[other], dim=0)
            loss = loss + 2 * (a.T @ b).square().sum()
    return loss


def total_loss(out, labels, view1, view2, args, negative_views=None, model=None, return_parts=False):
    zero = out["logits"].sum() * 0
    pred = F.cross_entropy(out["logits"], labels.long())
    node = (node_contrastive_loss(view1["all"], view2["all"], args.temperature, args.batch_size)
            if args.lambda_node else zero)
    graph = (graph_contrastive_loss(view1["all"], view2["all"], [v["all"] for v in (negative_views or [])],
                                    args.graph_temperature) if args.lambda_graph else zero)
    decor = ((intent_decorrelation_loss(out["positive_intents"]) +
              intent_decorrelation_loss(out["negative_intents"])) / 2 if args.lambda_decor else zero)
    if args.weight_decay and model is None:
        raise ValueError("Pass model to include explicit beta * ||Theta||^2")
    regularization = sum((p.square().sum() for p in model.parameters()), zero) if args.weight_decay else zero
    total = pred + args.lambda_node * node + args.lambda_graph * graph + args.lambda_decor * decor
    total = total + args.weight_decay * regularization
    parts = {"prediction": pred, "node": node, "graph": graph, "decorrelation": decor,
             "parameter_l2": regularization, "total": total}
    return (total, parts) if return_parts else total
