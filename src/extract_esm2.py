"""
Optional ESM-2 feature extractor.

Paper setting: esm2_t33_650M_UR50D, final-layer residue representations,
mean pooling excluding special/padding tokens. Install `fair-esm` separately.
"""
import argparse
import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def extract(input_csv, output_npy, sequence_col="sequence", device="cuda"):
    try:
        import esm
    except ImportError as e:
        raise ImportError("Install fair-esm to use extract_esm2.py") from e

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model = model.eval().to(device)

    df = pd.read_csv(input_csv)
    features = []

    for i, seq in enumerate(df[sequence_col].astype(str)):
        _, _, tokens = batch_converter([(str(i), seq)])
        tokens = tokens.to(device)
        result = model(tokens, repr_layers=[33], return_contacts=False)
        rep = result["representations"][33][0, 1:len(seq)+1].mean(0)
        features.append(rep.cpu().numpy())

    np.save(output_npy, np.stack(features).astype(np.float32))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--sequence-col", default="sequence")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    extract(args.input, args.output, args.sequence_col, args.device)


if __name__ == "__main__":
    main()
