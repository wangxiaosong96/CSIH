"""
LINCS L1000 sign construction from perturbational z-scores.

The manuscript describes aggregation at the drug-gene level followed by
thresholding of the aggregated score. The exact filtering thresholds and
metadata rules used for the reported paper results must be filled with the
authors' original preprocessing settings.
"""
import argparse
import pandas as pd


def aggregate_lincs(input_csv, output_csv, drug_col, gene_col, z_col, threshold=0.0):
    df = pd.read_csv(input_csv)
    df = df[[drug_col, gene_col, z_col]].dropna()
    agg = (
        df.groupby([drug_col, gene_col], as_index=False)[z_col]
        .median()
        .rename(columns={drug_col: "drug", gene_col: "gene", z_col: "median_z"})
    )
    agg = agg[agg["median_z"] != threshold].copy()
    agg["sign"] = (agg["median_z"] > threshold).map({True: 1, False: -1})
    agg[["drug", "gene", "sign"]].to_csv(output_csv, index=False)
    return agg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--drug-col", default="drug")
    p.add_argument("--gene-col", default="gene")
    p.add_argument("--z-col", default="z_score")
    p.add_argument("--threshold", type=float, default=0.0)
    args = p.parse_args()

    df = aggregate_lincs(
        args.input, args.output, args.drug_col, args.gene_col,
        args.z_col, args.threshold
    )
    print(f"Saved {len(df)} aggregated LINCS drug-gene pairs to {args.output}")


if __name__ == "__main__":
    main()
