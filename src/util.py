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


def evaluate_metrics(labels, probabilities, threshold=0.5):
    y = np.asarray(labels).astype(int)
    p = np.asarray(probabilities)
    pred = (p >= threshold).astype(int)
    return {
        "AUC": float(roc_auc_score(y, p)),
        "AUPR": float(average_precision_score(y, p)),
        "Binary-F1": float(f1_score(y, pred)),
    }


def node_contrastive_loss(z1, z2, temperature=0.2):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.T / temperature
    target = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (
        F.cross_entropy(logits, target)
        + F.cross_entropy(logits.T, target)
    )


def graph_contrastive_loss(z1, z2, temperature=0.2):
    g1 = F.normalize(z1.mean(dim=0, keepdim=True), dim=-1)
    g2 = F.normalize(z2.mean(dim=0, keepdim=True), dim=-1)

    corrupted = torch.flip(z2, dims=[0])
    gc = F.normalize(corrupted.mean(dim=0, keepdim=True), dim=-1)

    pos = (g1 * g2).sum(dim=-1, keepdim=True) / temperature
    neg = g1 @ gc.T / temperature

    logits = torch.cat([pos, neg], dim=1)
    target = torch.zeros(1, dtype=torch.long, device=z1.device)
    return F.cross_entropy(logits, target)


def intent_decorrelation_loss(intent_reps):
    if len(intent_reps) < 2:
        return intent_reps[0].new_tensor(0.0)

    loss = intent_reps[0].new_tensor(0.0)
    count = 0
    for i in range(len(intent_reps)):
        zi = F.normalize(intent_reps[i], dim=-1)
        for j in range(i + 1, len(intent_reps)):
            zj = F.normalize(intent_reps[j], dim=-1)
            corr = zi.T @ zj / max(1, zi.size(0))
            loss = loss + corr.pow(2).mean()
            count += 1
    return loss / max(1, count)


def total_loss(out, labels, view1, view2, args):
    pred_loss = F.binary_cross_entropy_with_logits(
        out["logits"], labels.float()
    )
    node_loss = node_contrastive_loss(
        view1["all"], view2["all"], args.temperature
    )
    graph_loss = graph_contrastive_loss(
        view1["all"], view2["all"], args.graph_temperature
    )
    decor_loss = 0.5 * (
        intent_decorrelation_loss(out["positive_intents"])
        + intent_decorrelation_loss(out["negative_intents"])
    )

    total = (
        args.lambda_sup * pred_loss
        + args.lambda_node * node_loss
        + args.lambda_graph * graph_loss
        + args.lambda_decor * decor_loss
    )
    return total
