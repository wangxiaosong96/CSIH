"""Four training-only sign-frequency baselines on recorded CSIH splits."""
import argparse
import json
from pathlib import Path
import numpy as np
try:
    from .generate import load_dataset
    from .run import validate_split
    from .save_results import write_csv, write_json
    from .util import evaluate_metrics
except ImportError:
    from generate import load_dataset
    from run import validate_split
    from save_results import write_csv, write_json
    from util import evaluate_metrics

METHODS = ("Global-Prior", "Drug-Prior", "Gene-Prior", "Drug-Gene-Prior")


class SignFrequencyBaseline:
    def fit(self, pairs, labels):
        pairs, labels = np.asarray(pairs), np.asarray(labels)
        if not len(pairs) or not np.isin(labels, [0, 1]).all():
            raise ValueError("Fit on nonempty training-only 0/1 labels")
        self.global_prior = float(labels.mean())
        self.drug_prior = {int(i): float(labels[pairs[:, 0] == i].mean()) for i in np.unique(pairs[:, 0])}
        self.gene_prior = {int(i): float(labels[pairs[:, 1] == i].mean()) for i in np.unique(pairs[:, 1])}
        return self

    def predict(self, pairs, method):
        if method not in METHODS:
            raise ValueError(method)
        scores = []
        for d, g in np.asarray(pairs):
            dp, gp = self.drug_prior.get(int(d)), self.gene_prior.get(int(g))
            if method == "Global-Prior":
                value = self.global_prior
            elif method == "Drug-Prior":
                value = self.global_prior if dp is None else dp
            elif method == "Gene-Prior":
                value = self.global_prior if gp is None else gp
            else:
                seen = [x for x in (dp, gp) if x is not None]
                value = float(np.mean(seen)) if seen else self.global_prior
            scores.append(value)
        return np.asarray(scores)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--splits", required=True, help="CSIH output/splits directory")
    p.add_argument("--protocol", choices=("random", "drug", "gene", "double"), default="random")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    output = Path(a.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty output directory")
    data = load_dataset(a.data)
    files = sorted((Path(a.splits) / a.protocol).glob("seed_*/fold_*.json"))
    if not files:
        raise FileNotFoundError("No saved splits found")
    rows, seed_rows, predictions = [], [], []
    for file in files:
        split = json.loads(file.read_text(encoding="utf-8"))
        if split["dataset_sha256"] != data.fingerprint or split["protocol"] != a.protocol:
            raise ValueError("Baseline split dataset hash/protocol mismatch")
        validate_split(data, split, a.protocol)
        tr, te = np.asarray(split["train_idx"]), np.asarray(split["test_idx"])
        baseline = SignFrequencyBaseline().fit(data.pairs[tr], data.labels[tr])
        for method in METHODS:
            probabilities = baseline.predict(data.pairs[te], method)
            metrics = evaluate_metrics(data.labels[te], probabilities)
            row = {"dataset": Path(a.data).stem, "protocol": a.protocol, "method": method,
                   "seed": split["seed"], "fold": split["fold"], **metrics}
            rows.append(row)
            for idx, probability in zip(te, probabilities):
                di, gi = data.pairs[idx].tolist()
                predictions.append({"dataset": Path(a.data).stem, "protocol": a.protocol,
                                    "method": method, "seed": split["seed"], "fold": split["fold"],
                                    "row_index": int(idx), "drug": data.drug_names[di], "gene": data.gene_names[gi],
                                    "true_sign": 1 if data.labels[idx] else -1,
                                    "prob_positive": float(probability), "prob_negative": float(1 - probability),
                                    "predicted_sign": 1 if probability >= 0.5 else -1})
    for method in METHODS:
        for seed in sorted({r["seed"] for r in rows}):
            current = [r for r in rows if r["method"] == method and r["seed"] == seed]
            seed_rows.append({"dataset": Path(a.data).stem, "protocol": a.protocol, "method": method, "seed": seed,
                              **{key: float(np.mean([r[key] for r in current])) for key in ("AUC", "AUPR", "Binary-F1")}})
    write_csv(output / "fold_metrics.csv", rows)
    write_csv(output / "seed_scores.csv", seed_rows)
    write_csv(output / "test_predictions.csv", predictions)
    write_json(output / "manifest.json", {"dataset_sha256": data.fingerprint, "source_split_files": [str(f) for f in files],
                                           "threshold": 0.5, "positive_class": "+1"})
    print(f"Saved {len(rows)} fold-level baseline scores")


if __name__ == "__main__":
    main()
