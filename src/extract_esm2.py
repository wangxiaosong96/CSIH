"""Frozen ESM-2 final-layer residue mean pooling; preserve gene IDs."""
import argparse
import numpy as np
import pandas as pd
import torch
try:
    from .feature_io import save_features, package_version
    from .generate import sha256_file
except ImportError:
    from feature_io import save_features, package_version
    from generate import sha256_file


@torch.no_grad()
def extract(input_csv, output, id_col="gene", sequence_col="sequence", device="cpu", checkpoint=None):
    import esm
    table = pd.read_csv(input_csv, dtype=str)
    if not {id_col, sequence_col}.issubset(table) or table[[id_col, sequence_col]].isna().any().any():
        raise ValueError("Input requires gene IDs and canonical protein sequences")
    if table[id_col].duplicated().any():
        raise ValueError("Supply one canonical protein product per gene ID")
    if checkpoint:
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(checkpoint)
    else:
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(device)
    converter = alphabet.get_batch_converter()
    features = []
    for identifier, sequence in zip(table[id_col], table[sequence_col]):
        sequence = "".join(sequence.split()).upper()
        if not sequence:
            raise ValueError(f"Empty sequence for {identifier}")
        _, _, tokens = converter([(identifier, sequence)])
        representation = model(tokens.to(device), repr_layers=[model.num_layers], return_contacts=False)["representations"][model.num_layers]
        features.append(representation[0, 1:len(sequence) + 1].mean(0).cpu().numpy())
    save_features(output, table[id_col].tolist(), np.stack(features),
                  {"encoder": "esm2_t33_650M_UR50D" if not checkpoint else "local ESM checkpoint",
                   "checkpoint": checkpoint, "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None,
                   "fair_esm_version": package_version("fair-esm"), "input_sha256": sha256_file(input_csv),
                   "layer": model.num_layers, "pooling": "mean residues; exclude BOS/EOS/padding", "truncation": "none"})


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--id-col", default="gene")
    p.add_argument("--sequence-col", default="sequence")
    p.add_argument("--device", default="cpu")
    p.add_argument("--checkpoint")
    a = p.parse_args()
    extract(a.input, a.output, a.id_col, a.sequence_col, a.device, a.checkpoint)


if __name__ == "__main__":
    main()
