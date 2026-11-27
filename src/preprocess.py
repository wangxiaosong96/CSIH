"""Exact annotation mapping; mixed-sign records and pairs are excluded."""
import json
import math
import re
from pathlib import Path
import pandas as pd

POSITIVE_TERMS = {"activator", "activation", "activate", "agonist", "partial agonist", "inducer",
                  "upregulator", "up-regulator", "stimulator", "positive modulator"}
NEGATIVE_TERMS = {"inhibitor", "inhibition", "inhibit", "antagonist", "partial antagonist", "blocker",
                  "suppressor", "downregulator", "down-regulator", "negative modulator", "inverse agonist"}
ACTION_MAPPING = {**dict.fromkeys(POSITIVE_TERMS, 1), **dict.fromkeys(NEGATIVE_TERMS, -1)}
SIGN_MAPPING = {"1": 1, "+1": 1, "1.0": 1, "-1": -1, "-1.0": -1,
                "positive": 1, "negative": -1, "up": 1, "down": -1}


def annotation_tokens(value):
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip().lower() for v in value]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip().lower()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(v).strip().lower() for v in parsed]
        except json.JSONDecodeError:
            pass
    return [v.strip() for v in re.split(r"[;,|]", text) if v.strip()]


def normalize_sign(value, mapping=None):
    dictionary = {**ACTION_MAPPING, **SIGN_MAPPING} if mapping is None else mapping
    signs = {dictionary[t] for t in annotation_tokens(value) if t in dictionary}
    return next(iter(signs)) if len(signs) == 1 else None


def standardize_interaction_table(input_csv, output_csv, drug_col="drug", gene_col="gene", sign_col="sign",
                                  mapping=None, sep=",", audit_path=None, reject_unknown=False):
    df = pd.read_csv(input_csv, sep=sep, dtype=str)
    needed = {drug_col, gene_col, sign_col}
    if not needed.issubset(df.columns):
        raise ValueError(f"Missing columns: {needed - set(df.columns)}")
    cleaned = df[[drug_col, gene_col, sign_col]].dropna().copy()
    cleaned[drug_col] = cleaned[drug_col].str.strip()
    cleaned[gene_col] = cleaned[gene_col].str.strip()
    cleaned = cleaned[(cleaned[drug_col] != "") & (cleaned[gene_col] != "")]
    dictionary = {**ACTION_MAPPING, **SIGN_MAPPING} if mapping is None else mapping
    if any(v not in (-1, 1) for v in dictionary.values()):
        raise ValueError("Mapping values must be -1 or +1")
    signs = cleaned[sign_col].map(lambda value: normalize_sign(value, dictionary))
    unmapped = sorted({str(v) for v in cleaned.loc[signs.isna(), sign_col]})
    if reject_unknown and unmapped:
        raise ValueError(f"Unmapped or conflicting relation labels: {unmapped[:20]}; supply an explicit mapping")
    out = pd.DataFrame({"drug": cleaned[drug_col], "gene": cleaned[gene_col], "sign": signs}).dropna()
    before = len(out)
    out = out.drop_duplicates()
    duplicate_rows = before - len(out)
    conflicts = out.groupby(["drug", "gene"])["sign"].nunique()
    bad = set(conflicts[conflicts > 1].index)
    mask = pd.Series([pair not in bad for pair in zip(out.drug, out.gene)], index=out.index, dtype=bool)
    out = out.loc[mask].copy()
    out["sign"] = out["sign"].astype(int)
    out = out.sort_values(["drug", "gene"]).reset_index(drop=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    audit = {"input_rows": len(df), "rows_with_usable_entities": len(cleaned), "unmapped_or_conflicting_values": unmapped,
             "exact_duplicates_removed": duplicate_rows, "opposite_sign_pairs_excluded": len(bad),
             "retained_pairs": len(out), "positive_pairs": int((out.sign == 1).sum()),
             "negative_pairs": int((out.sign == -1).sum()), "mapping": dictionary,
             "multi_type_rule": "exact tokens; keep only a single mapped sign; exclude mixed-sign records"}
    Path(audit_path or str(output_csv) + ".audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return out
