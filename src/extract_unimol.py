"""
Uni-Mol feature extraction placeholder/adapter.

The manuscript specifies frozen pretrained Uni-Mol molecular representations.
Uni-Mol installations/checkpoints and input conformer pipelines differ across
releases, so this file deliberately validates the expected output interface
instead of inventing an extraction API that may not match the authors' setup.

Replace `encode_with_your_unimol_pipeline` with the exact Uni-Mol preprocessing
and checkpoint code used for the paper, then save one vector per drug.
"""
import argparse
import numpy as np
import pandas as pd


def encode_with_your_unimol_pipeline(smiles_list):
    raise NotImplementedError(
        "Insert the exact Uni-Mol checkpoint/conformer extraction pipeline "
        "used in the experiments. The manuscript alone does not specify enough "
        "implementation detail to reconstruct that external pipeline safely."
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--smiles-col", default="smiles")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    vectors = encode_with_your_unimol_pipeline(df[args.smiles_col].astype(str).tolist())
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(df):
        raise ValueError("Uni-Mol output must be [num_drugs, feature_dim].")
    np.save(args.output, vectors)


if __name__ == "__main__":
    main()
