"""Disjoint test/validation entities and training-only context masks."""
import numpy as np


def _partition_entities(values, test_ratio, val_ratio, rng):
    values = rng.permutation(np.unique(values))
    nt = max(1, round(len(values) * test_ratio))
    nv = max(1, round((len(values) - nt) * val_ratio))
    if nt + nv >= len(values):
        raise ValueError("Too few entities for disjoint train/validation/test partitions")
    return values[nt + nv:], values[nt:nt + nv], values[:nt]


def make_cold_start_split(pairs, mode="drug", holdout_ratio=0.2, seed=42, val_ratio=0.1):
    if mode not in ("drug", "gene", "double") or not 0 < holdout_ratio < 1 or not 0 < val_ratio < 1:
        raise ValueError("Invalid cold-start mode or holdout ratios")
    pairs = np.asarray(pairs)
    rng = np.random.default_rng(seed)
    empty = np.array([], dtype=np.int64)
    dt, dv, de = (np.unique(pairs[:, 0]), empty, empty)
    gt, gv, ge = (np.unique(pairs[:, 1]), empty, empty)
    if mode in ("drug", "double"):
        dt, dv, de = _partition_entities(pairs[:, 0], holdout_ratio, val_ratio, rng)
    if mode in ("gene", "double"):
        gt, gv, ge = _partition_entities(pairs[:, 1], holdout_ratio, val_ratio, rng)
    train = np.isin(pairs[:, 0], dt) & np.isin(pairs[:, 1], gt)
    # A warm endpoint must actually have a training interaction.
    warm_d, warm_g = np.unique(pairs[train, 0]), np.unique(pairs[train, 1])
    if mode == "drug":
        val = np.isin(pairs[:, 0], dv) & np.isin(pairs[:, 1], warm_g)
        test = np.isin(pairs[:, 0], de) & np.isin(pairs[:, 1], warm_g)
    elif mode == "gene":
        val = np.isin(pairs[:, 0], warm_d) & np.isin(pairs[:, 1], gv)
        test = np.isin(pairs[:, 0], warm_d) & np.isin(pairs[:, 1], ge)
    else:
        val = np.isin(pairs[:, 0], dv) & np.isin(pairs[:, 1], gv)
        test = np.isin(pairs[:, 0], de) & np.isin(pairs[:, 1], ge)
    return {"train_idx": np.flatnonzero(train), "val_idx": np.flatnonzero(val),
            "test_idx": np.flatnonzero(test), "held_drugs": de, "held_genes": ge,
            "validation_drugs": dv, "validation_genes": gv,
            "discarded_idx": np.flatnonzero(~(train | val | test))}
