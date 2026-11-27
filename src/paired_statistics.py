"""Paired seed-level t intervals, Cohen dz and joint Holm correction."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(pvalues[order] * np.arange(len(pvalues), 0, -1)))
    return adjusted


def compare_seed_scores(scores, reference, comparators):
    keys = ["dataset", "protocol", "method", "seed"]
    if not set(keys + ["AUC", "AUPR", "Binary-F1"]).issubset(scores.columns):
        raise ValueError("Expected dataset,protocol,method,seed,AUC,AUPR,Binary-F1 columns")
    if scores.duplicated(keys).any():
        raise ValueError("Seed-level rows must be unique; average folds within each seed first")
    if reference in comparators or len(set(comparators)) != len(comparators):
        raise ValueError("Comparators must be unique and differ from the reference")
    rows = []
    for (dataset, protocol), group in scores.groupby(["dataset", "protocol"]):
        first = group[group.method == reference].set_index("seed")
        if first.empty:
            raise ValueError(f"Missing reference scores for {dataset}/{protocol}")
        for comparator in comparators:
            other = group[group.method == comparator].set_index("seed")
            if set(first.index) != set(other.index) or len(first) < 2:
                raise ValueError(f"Need >=2 identical matched seed sets for {dataset}/{comparator}")
            index = sorted(first.index)
            for metric in ("AUC", "AUPR", "Binary-F1"):
                x, y = first.loc[index, metric].to_numpy(float), other.loc[index, metric].to_numpy(float)
                if not np.isfinite(x).all() or not np.isfinite(y).all():
                    raise ValueError("Statistics require finite seed-level scores")
                differences = x - y
                mean, sd, n = differences.mean(), differences.std(ddof=1), len(differences)
                margin = float(t.ppf(.975, n - 1) * sd / np.sqrt(n))
                raw_p = float(2 * t.sf(abs(mean / sd * np.sqrt(n)), n - 1)) if sd > 0 else (1.0 if mean == 0 else 0.0)
                rows.append({"dataset": dataset, "protocol": protocol, "reference": reference, "comparator": comparator,
                             "metric": metric, "n_seeds": n, "reference_mean": x.mean(), "comparator_mean": y.mean(),
                             "mean_difference": mean, "CI95_low": mean - margin, "CI95_high": mean + margin,
                             "Cohen_dz": mean / sd if sd > 0 else np.nan, "raw_p": raw_p,
                             "degenerate_difference_variance": bool(sd == 0)})
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No requested paired comparisons")
    result["Holm_p"] = holm_adjust(result.raw_p)
    result["Holm_significant_0.05"] = result.Holm_p < .05
    result["Holm_family_size"] = len(result)
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--reference", default="full")
    p.add_argument("--comparators", nargs="+", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    output = Path(a.output)
    if output.exists():
        raise FileExistsError("Choose a new output file")
    result = compare_seed_scores(pd.concat([pd.read_csv(f) for f in a.inputs], ignore_index=True), a.reference, a.comparators)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Saved {len(result)} comparisons; pointwise unadjusted CIs, Holm correction across all rows")


if __name__ == "__main__":
    main()
