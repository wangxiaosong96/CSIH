"""
DGIdb preprocessing entry point.

The manuscript states that only explicit directional interaction types are
mapped to activation/inhibition and ambiguous/non-directional records are removed.
Because raw DGIdb exports can use different column names, these are arguments.
"""
import argparse
from preprocess import standardize_interaction_table


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--drug-col", default="drug")
    p.add_argument("--gene-col", default="gene")
    p.add_argument("--interaction-col", default="interaction_type")
    args = p.parse_args()

    df = standardize_interaction_table(
        args.input, args.output,
        args.drug_col, args.gene_col, args.interaction_col
    )
    print(f"Saved {len(df)} directional DGIdb interactions to {args.output}")


if __name__ == "__main__":
    main()
