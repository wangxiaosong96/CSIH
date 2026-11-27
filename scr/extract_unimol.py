"""UniMolRepr adapter, with explicit extraction configuration and drug IDs.

Use the original precomputed vectors for a strict reproduction. This adapter
uses the installed official unimol_tools API and records its configuration;
its defaults are not evidence of the original manuscript checkpoint pipeline.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
try:
    from .feature_io import save_features, package_version
    from .generate import sha256_file
except ImportError:
    from feature_io import save_features, package_version
    from generate import sha256_file


def extract(input_csv, output, id_col="drug", smiles_col="smiles", encoder_config=None):
    from unimol_tools import UniMolRepr
    table = pd.read_csv(input_csv, dtype=str)
    if not {id_col, smiles_col}.issubset(table) or table[[id_col, smiles_col]].isna().any().any():
        raise ValueError("Input requires drug IDs and SMILES")
    if table[id_col].duplicated().any() or table[smiles_col].str.strip().eq("").any():
        raise ValueError("Supply unique drug IDs and nonempty standardized structures")
    config = {"data_type": "molecule", "remove_hs": False, **(encoder_config or {})}
    encoder = UniMolRepr(**config)
    result = encoder.get_repr(table[smiles_col].tolist(), return_atomic_reprs=False)
    vectors = np.asarray(result["cls_repr"], dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) != len(table):
        raise ValueError("Uni-Mol omitted/changed input rows; resolve failed structures before exporting IDs")
    hashes = {key: sha256_file(value) for key, value in config.items()
              if isinstance(value, str) and Path(value).is_file()}
    save_features(output, table[id_col].tolist(), vectors,
                  {"encoder": "official unimol_tools.UniMolRepr", "configuration": config,
                   "configured_file_sha256": hashes, "unimol_tools_version": package_version("unimol-tools"),
                   "input_sha256": sha256_file(input_csv), "pooling": "cls_repr"})


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--id-col", default="drug")
    p.add_argument("--smiles-col", default="smiles")
    p.add_argument("--encoder-config", help="JSON constructor settings, including your checkpoint/conformer settings")
    a = p.parse_args()
    config = json.loads(Path(a.encoder_config).read_text(encoding="utf-8")) if a.encoder_config else None
    extract(a.input, a.output, a.id_col, a.smiles_col, config)


if __name__ == "__main__":
    main()
