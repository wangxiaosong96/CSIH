"""Draw ROC/PR from saved predictions, computing every legend value anew."""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--predictions", nargs="+", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", required=True, help="PNG or PDF; one dataset/protocol per call")
    a = p.parse_args()
    frames = []
    for file in a.predictions:
        frame = pd.read_csv(file)
        if "method" not in frame:
            frame["method"] = frame["variant"]
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    table = table[table.seed == a.seed]
    if table.empty or len(table[["dataset", "protocol"]].drop_duplicates()) != 1:
        raise ValueError("Select nonempty predictions for exactly one dataset/protocol")
    if table.duplicated(["method", "row_index"]).any():
        raise ValueError("Duplicate predictions per method/row; provide one selected seed only")
    row_sets = [set(g.row_index) for _, g in table.groupby("method")]
    if any(values != row_sets[0] for values in row_sets):
        raise ValueError("All methods must be evaluated on identical row sets")
    if table.groupby("row_index").true_sign.nunique().gt(1).any():
        raise ValueError("Methods disagree on ground-truth labels")
    output = Path(a.output)
    if output.exists():
        raise FileExistsError(output)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for method, group in table.groupby("method", sort=False):
        y, probability = (group.true_sign == 1).astype(int), group.prob_positive
        if y.nunique() != 2 or probability.isna().any() or not probability.between(0, 1).all():
            raise ValueError("Curves require two sign classes and probabilities in [0,1]")
        fpr, tpr, _ = roc_curve(y, probability)
        precision, recall, _ = precision_recall_curve(y, probability)
        axes[0].plot(fpr, tpr, label=f"{method}: AUC={roc_auc_score(y, probability):.4f}")
        axes[1].plot(recall, precision, label=f"{method}: AP={average_precision_score(y, probability):.4f}")
    axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=.7)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate")
    axes[1].set(xlabel="Recall", ylabel="Precision")
    for axis in axes:
        axis.set(xlim=(0, 1), ylim=(0, 1))
        axis.legend(fontsize=7)
    figure.suptitle(f"{table.dataset.iloc[0]} | {table.protocol.iloc[0]} | seed {a.seed} | pooled test predictions")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300)
    plt.close(figure)


if __name__ == "__main__":
    main()
