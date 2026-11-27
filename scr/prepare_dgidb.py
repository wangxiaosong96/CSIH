"""Convert a specified DGIdb export using a versioned exact dictionary."""
import argparse
import json
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
    p.add_argument("--interaction-col", default="interaction_type")
    p.add_argument("--sep", default=",")
    p.add_argument("--mapping-json", help="Optional exact token -> +/-1 dictionary used in your data construction")
    a = p.parse_args()
    mapping = json.load(open(a.mapping_json, encoding="utf-8")) if a.mapping_json else None
    result = standardize_interaction_table(a.input, a.output, a.drug_col, a.gene_col,
                                          a.interaction_col, mapping=mapping, sep=a.sep)
    print(f"Saved {len(result)} DGIdb pairs and an audit JSON")


if __name__ == "__main__":
    main()
