"""Cold-start split generation for drug, gene, and double cold-start evaluation."""
import numpy as np


def make_cold_start_split(pairs, mode="drug", holdout_ratio=0.2, seed=42):
    pairs = np.asarray(pairs)
    rng = np.random.default_rng(seed)

    drugs = np.unique(pairs[:, 0])
    genes = np.unique(pairs[:, 1])
    held_drugs = np.array([], dtype=np.int64)
    held_genes = np.array([], dtype=np.int64)

    if mode in ("drug", "double"):
        n = max(1, int(round(len(drugs) * holdout_ratio)))
        held_drugs = rng.choice(drugs, n, replace=False)
    if mode in ("gene", "double"):
        n = max(1, int(round(len(genes) * holdout_ratio)))
        held_genes = rng.choice(genes, n, replace=False)

    cold_d = np.isin(pairs[:, 0], held_drugs)
    cold_g = np.isin(pairs[:, 1], held_genes)

    if mode == "drug":
        test = cold_d
        train = ~cold_d
    elif mode == "gene":
        test = cold_g
        train = ~cold_g
    elif mode == "double":
        test = cold_d & cold_g
        # No held-out entity may enter the training graph.
        train = (~cold_d) & (~cold_g)
    else:
        raise ValueError("mode must be 'drug', 'gene', or 'double'")

    return {
        "train_idx": np.flatnonzero(train),
        "test_idx": np.flatnonzero(test),
        "held_drugs": held_drugs,
        "held_genes": held_genes,
    }
