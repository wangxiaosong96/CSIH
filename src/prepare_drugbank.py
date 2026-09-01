"""
DrugBank preprocessing entry point.

This script intentionally does not download or redistribute DrugBank data.
Provide a legally obtained local export and map its explicit action/mechanism
field to the standardized signed interaction table.
"""
import argparse
from preprocess import standardize_interaction_table


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--drug-col", default="drug")
    p.add_argument("--gene-col", default="gene")
    p.add_argument("--action-col", default="action")
    args = p.parse_args()

    df = standardize_interaction_table(
        args.input, args.output,
        args.drug_col, args.gene_col, args.action_col
    )
    print(f"Saved {len(df)} directional DrugBank interactions to {args.output}")


if __name__ == "__main__":
    main()
