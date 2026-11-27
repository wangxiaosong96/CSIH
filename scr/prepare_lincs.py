"""Import released labels OR separately audit condition-level expression scores."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
try:
    from .preprocess import standardize_interaction_table
    from .save_results import write_json
    from .generate import sha256_file
except ImportError:
    from preprocess import standardize_interaction_table
    from save_results import write_json
    from generate import sha256_file

CONDITION_COLUMNS = ["drug", "cell_line", "dose", "dose_unit", "duration", "duration_unit"]


def audit_conditions(expression_csv, benchmark_csv, output_dir, informative_threshold=2.0,
                     label_threshold=2.0, aggregation_scope="all", min_signatures=1):
    """Input: mapped Level-5 long table; no inferred chemistry/metadata linkage.

    Repeated signature rows within exactly the same condition/gene are reduced
    by their median. Median/mean are then computed over either all usable
    conditions or just informative conditions. A label is retained only when
    abs(aggregate) >= label_threshold and aggregate != 0. Published main labels
    are read for comparison only and are never overwritten.
    """
    if informative_threshold <= 0 or label_threshold < 0 or min_signatures < 1:
        raise ValueError("Invalid response thresholds/minimum signature count")
    if aggregation_scope not in ("all", "informative"):
        raise ValueError("aggregation_scope must be all or informative")
    outdir = Path(output_dir)
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError("Use an empty audit output directory")
    raw = pd.read_csv(expression_csv, dtype={c: str for c in ("drug", "gene", "cell_line", "dose_unit", "duration_unit")})
    required = set(CONDITION_COLUMNS + ["gene", "z_score", "signature_id"])
    if not required.issubset(raw.columns):
        raise ValueError(f"Condition table is missing {required - set(raw.columns)}")
    benchmark = pd.read_csv(benchmark_csv, dtype={"drug": str, "gene": str})
    if not {"drug", "gene", "sign"}.issubset(benchmark) or benchmark.duplicated(["drug", "gene"]).any():
        raise ValueError("Benchmark requires unique drug,gene,sign rows")
    if benchmark[["drug", "gene", "sign"]].isna().any().any() or not benchmark.sign.isin([-1, 1]).all():
        raise ValueError("Invalid released benchmark labels")
    # The same signature/gene must not be counted twice as independent evidence.
    key = ["drug", "gene", "signature_id"]
    if raw.signature_id.isna().any() or raw.signature_id.astype(str).str.strip().eq("").any() or raw.duplicated(key).any():
        raise ValueError("Signature IDs must be nonmissing and unique within drug/gene")
    raw_rows = len(raw)
    for column in ("dose", "duration", "z_score"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    missing_key = raw[CONDITION_COLUMNS + ["gene"]].isna().any(axis=1)
    for column in ("drug", "gene", "cell_line", "dose_unit", "duration_unit"):
        raw[column] = raw[column].str.strip()
        missing_key |= raw[column].eq("")
    linked = raw.loc[~missing_key].merge(benchmark[["drug", "gene"]], on=["drug", "gene"], how="inner")
    candidate_pairs = linked[["drug", "gene"]].drop_duplicates()
    usable = linked[np.isfinite(linked.z_score) & np.isfinite(linked.dose) & np.isfinite(linked.duration)].copy()
    group_keys = CONDITION_COLUMNS + ["gene"]
    conditions = usable.groupby(group_keys, as_index=False).agg(z_score=("z_score", "median"),
                                                                signature_count=("z_score", "size"))
    conditions = conditions[conditions.signature_count >= min_signatures].copy()
    conditions["informative_up"] = conditions.z_score >= informative_threshold
    conditions["informative_down"] = conditions.z_score <= -informative_threshold
    pair_stats = conditions.groupby(["drug", "gene"], as_index=False).agg(
        usable_conditions=("z_score", "size"), up_conditions=("informative_up", "sum"),
        down_conditions=("informative_down", "sum"))
    pair_stats = candidate_pairs.merge(pair_stats, on=["drug", "gene"], how="left")
    for column in ("usable_conditions", "up_conditions", "down_conditions"):
        pair_stats[column] = pair_stats[column].fillna(0).astype(int)
    pair_stats["informative_conditions"] = pair_stats.up_conditions + pair_stats.down_conditions
    pair_stats["discordant"] = (pair_stats.up_conditions > 0) & (pair_stats.down_conditions > 0)
    subset = conditions if aggregation_scope == "all" else conditions[conditions.informative_up | conditions.informative_down]
    aggregates = subset.groupby(["drug", "gene"], as_index=False).agg(median_z=("z_score", "median"), mean_z=("z_score", "mean"))
    reconstructed = {}
    for name in ("median", "mean"):
        column = name + "_z"
        table = aggregates.loc[(aggregates[column].abs() >= label_threshold) & (aggregates[column] != 0),
                               ["drug", "gene", column]].copy()
        table["sign"] = np.where(table[column] > 0, 1, -1)
        reconstructed[name] = table
    consistency = reconstructed["median"].merge(pair_stats[["drug", "gene", "discordant"]], on=["drug", "gene"])
    reconstructed["median_consistent"] = consistency.loc[~consistency.discordant].drop(columns="discordant")
    agreements = []
    for first, second, left, right in (("median", "released", reconstructed["median"], benchmark),
                                       ("mean", "released", reconstructed["mean"], benchmark),
                                       ("median", "mean", reconstructed["median"], reconstructed["mean"])):
        shared = left.merge(right, on=["drug", "gene"], suffixes=("_left", "_right"))
        equal = int((shared.sign_left == shared.sign_right).sum())
        agreements.append({"comparison": first + " vs " + second, "shared_pairs": len(shared), "agreeing_labels": equal,
                           "agreement_fraction": equal / len(shared) if len(shared) else None})
    outdir.mkdir(parents=True, exist_ok=True)
    conditions.to_csv(outdir / "condition_scores.csv", index=False)
    pair_stats.to_csv(outdir / "pair_condition_audit.csv", index=False)
    for name, table in reconstructed.items():
        table.to_csv(outdir / (name + "_reconstructed.csv"), index=False)
    n_multi = int((pair_stats.informative_conditions >= 2).sum())
    discordant = int(pair_stats.discordant.sum())
    audit = {"analysis": "supplementary expression audit; released main labels unchanged",
             "expression_input_sha256": sha256_file(expression_csv), "benchmark_sha256": sha256_file(benchmark_csv),
             "raw_expression_rows": raw_rows, "rows_missing_condition_keys": int(missing_key.sum()),
             "candidate_pairs_with_condition_metadata": len(candidate_pairs),
             "pairs_without_usable_condition_scores": int((pair_stats.usable_conditions == 0).sum()),
             "no_informative_conditions": int((pair_stats.informative_conditions == 0).sum()),
             "one_informative_condition": int((pair_stats.informative_conditions == 1).sum()),
             "at_least_two_informative_conditions": n_multi, "discordant_pairs": discordant,
             "discordance_fraction_among_multi": discordant / n_multi if n_multi else None,
             "condition_key": CONDITION_COLUMNS, "within_condition_aggregation": "median",
             "informative_threshold_absolute_z": informative_threshold, "label_threshold_absolute_aggregate": label_threshold,
             "aggregation_scope": aggregation_scope, "minimum_signature_rows_per_condition_gene": min_signatures,
             "retention_rule": "abs(aggregate)>=label_threshold and aggregate!=0; inclusive bounds",
             "cell_lines": sorted(conditions.cell_line.unique().tolist()),
             "dose_values_by_unit": {str(unit): sorted(group.dose.unique().tolist()) for unit, group in conditions.groupby("dose_unit")},
             "duration_values_by_unit": {str(unit): sorted(group.duration.unique().tolist()) for unit, group in conditions.groupby("duration_unit")},
             "retention": {name: {"pairs": len(table), "positive": int((table.sign == 1).sum()),
                                   "negative": int((table.sign == -1).sum())} for name, table in reconstructed.items()},
             "agreements": agreements}
    write_json(outdir / "audit_summary.json", audit)
    return audit


def main():
    p = argparse.ArgumentParser(__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    imp = commands.add_parser("import-benchmark", help="Import existing released labels without rebuilding them")
    for flag in ("input", "output", "positive-label", "negative-label"):
        imp.add_argument("--" + flag, required=True)
    imp.add_argument("--drug-col", default="drug")
    imp.add_argument("--gene-col", default="gene")
    imp.add_argument("--label-col", default="label")
    imp.add_argument("--sep", default=",")
    aud = commands.add_parser("audit-conditions")
    aud.add_argument("--expression", required=True)
    aud.add_argument("--benchmark", required=True)
    aud.add_argument("--output", required=True)
    aud.add_argument("--informative-threshold", type=float, default=2)
    aud.add_argument("--label-threshold", type=float, default=2)
    aud.add_argument("--aggregation-scope", choices=("all", "informative"), default="all")
    aud.add_argument("--min-signatures", type=int, default=1)
    a = p.parse_args()
    if a.command == "import-benchmark":
        positive, negative = a.positive_label.strip().lower(), a.negative_label.strip().lower()
        if positive == negative:
            p.error("The positive and negative released codes must differ")
        out = standardize_interaction_table(a.input, a.output, a.drug_col, a.gene_col, a.label_col,
                                           mapping={positive: 1, negative: -1}, sep=a.sep, reject_unknown=True)
        print(f"Imported {len(out)} released LINCS pairs")
    else:
        result = audit_conditions(a.expression, a.benchmark, a.output, a.informative_threshold,
                                  a.label_threshold, a.aggregation_scope, a.min_signatures)
        print(result["retention"])


if __name__ == "__main__":
    main()
