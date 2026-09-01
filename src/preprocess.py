"""Common preprocessing utilities for signed drug-gene interaction tables."""
from pathlib import Path
import pandas as pd


POSITIVE_TERMS = {
    "activator", "activation", "agonist", "inducer", "upregulator",
    "up-regulator", "stimulator", "positive modulator"
}
NEGATIVE_TERMS = {
    "inhibitor", "inhibition", "antagonist", "blocker", "suppressor",
    "downregulator", "down-regulator", "negative modulator"
}


def normalize_sign(value):
    text = str(value).strip().lower()
    if text in {"+1", "1", "positive", "activation", "activate", "up"}:
        return 1
    if text in {"-1", "negative", "inhibition", "inhibit", "down"}:
        return -1
    if any(term in text for term in POSITIVE_TERMS):
        return 1
    if any(term in text for term in NEGATIVE_TERMS):
        return -1
    return None


def standardize_interaction_table(
    input_csv, output_csv,
    drug_col="drug", gene_col="gene", sign_col="sign"
):
    df = pd.read_csv(input_csv)
    out = pd.DataFrame({
        "drug": df[drug_col].astype(str).str.strip(),
        "gene": df[gene_col].astype(str).str.strip(),
        "sign": df[sign_col].map(normalize_sign),
    })
    out = out.dropna().drop_duplicates()
    out["sign"] = out["sign"].astype(int)

    # Remove contradictory duplicate pairs rather than leaking an arbitrary label.
    counts = out.groupby(["drug", "gene"])["sign"].nunique()
    contradictory = counts[counts > 1].index
    if len(contradictory):
        bad = set(contradictory)
        mask = [(d, g) not in bad for d, g in zip(out.drug, out.gene)]
        out = out.loc[mask]

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out
