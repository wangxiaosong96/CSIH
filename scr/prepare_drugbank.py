"""Import released expression-direction labels, not pharmacological actions."""
import argparse
try:
    from .preprocess import standardize_interaction_table
except ImportError:
    from preprocess import standardize_interaction_table


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--drug-col", default="drug")
    p.add_argument("--gene-col", default="gene")
    p.add_argument("--label-col", default="label")
    p.add_argument("--positive-label", required=True, help="Exact released code for increased gene expression")
    p.add_argument("--negative-label", required=True, help="Exact released code for decreased gene expression")
    p.add_argument("--sep", default=",")
    a = p.parse_args()
    positive, negative = a.positive_label.strip().lower(), a.negative_label.strip().lower()
    if positive == negative:
        p.error("The two released label codes must differ")
    out = standardize_interaction_table(a.input, a.output, a.drug_col, a.gene_col, a.label_col,
                                       mapping={positive: 1, negative: -1}, sep=a.sep, reject_unknown=True)
    print(f"Saved {len(out)} expression-labeled benchmark pairs; source encoding recorded in audit JSON")


if __name__ == "__main__":
    main()
