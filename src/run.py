import copy
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from graph import CSIH
from generate import load_dataset, build_training_graph, sign_preserving_edge_dropout
from parameter import get_args
from util import set_seed, evaluate_metrics, total_loss


@torch.no_grad()
def evaluate(model, drug_feat, gene_feat, pos_edges, neg_edges, pairs, labels):
    model.eval()
    out = model(drug_feat, gene_feat, pos_edges, neg_edges, pairs)
    prob = out["prob_positive"].cpu().numpy()
    return evaluate_metrics(labels.cpu().numpy(), prob)


def train_fold(data, train_idx, test_idx, args, seed, device):
    tr_idx, val_idx = train_test_split(
        train_idx,
        test_size=args.val_ratio,
        stratify=data.labels[train_idx].numpy(),
        random_state=seed,
    )

    tr_pairs = data.pairs[tr_idx].to(device)
    tr_labels = data.labels[tr_idx].to(device)
    val_pairs = data.pairs[val_idx].to(device)
    val_labels = data.labels[val_idx].to(device)
    test_pairs = data.pairs[test_idx].to(device)
    test_labels = data.labels[test_idx].to(device)

    pos_edges, neg_edges = build_training_graph(
        tr_pairs.cpu(), tr_labels.cpu()
    )
    pos_edges = pos_edges.to(device)
    neg_edges = neg_edges.to(device)

    drug_feat = data.drug_features.to(device)
    gene_feat = data.gene_features.to(device)

    model = CSIH(
        args,
        num_drugs=len(data.drug_names),
        num_genes=len(data.gene_names),
        drug_feature_dim=drug_feat.size(1),
        gene_feature_dim=gene_feat.size(1),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_auc = -1
    best_state = None
    wait = 0

    for epoch in range(args.epochs):
        model.train()

        out = model(
            drug_feat, gene_feat, pos_edges, neg_edges, tr_pairs
        )

        p1 = sign_preserving_edge_dropout(pos_edges, args.edge_drop)
        n1 = sign_preserving_edge_dropout(neg_edges, args.edge_drop)
        p2 = sign_preserving_edge_dropout(pos_edges, args.edge_drop)
        n2 = sign_preserving_edge_dropout(neg_edges, args.edge_drop)

        view1 = model.encode(drug_feat, gene_feat, p1, n1)
        view2 = model.encode(drug_feat, gene_feat, p2, n2)

        loss = total_loss(out, tr_labels, view1, view2, args)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        val = evaluate(
            model, drug_feat, gene_feat,
            pos_edges, neg_edges, val_pairs, val_labels
        )

        if val["AUC"] > best_auc:
            best_auc = val["AUC"]
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if wait >= args.patience:
            break

    model.load_state_dict(best_state)

    return evaluate(
        model, drug_feat, gene_feat,
        pos_edges, neg_edges, test_pairs, test_labels
    )


def main():
    args = get_args()
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )

    data = load_dataset(
        args.data,
        args.drug_features,
        args.gene_features,
    )

    seed_results = []

    for seed in args.seeds:
        set_seed(seed)
        y = data.labels.numpy().astype(int)

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=seed,
        )

        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(
            cv.split(np.zeros(len(y)), y), start=1
        ):
            score = train_fold(
                data, train_idx, test_idx, args, seed, device
            )
            fold_results.append(score)
            print(
                f"seed={seed}, fold={fold}, "
                f"AUC={score['AUC']:.4f}, "
                f"AUPR={score['AUPR']:.4f}, "
                f"F1={score['Binary-F1']:.4f}"
            )

        seed_mean = {
            key: np.mean([x[key] for x in fold_results])
            for key in fold_results[0]
        }
        seed_results.append(seed_mean)

    print("\nFinal results")
    for metric in ["AUC", "AUPR", "Binary-F1"]:
        values = np.array([x[metric] for x in seed_results])
        print(
            f"{metric}: {values.mean():.4f} ± "
            f"{values.std(ddof=1):.4f}"
        )


if __name__ == "__main__":
    main()
