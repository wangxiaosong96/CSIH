"""Strict dataset loading, feature alignment, and training-only augmentation."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class CSIHData:
    drug_names: list
    gene_names: list
    drug_features: torch.Tensor
    gene_features: torch.Tensor
    pairs: torch.Tensor
    labels: torch.Tensor
    fingerprint: str


def read_features(path, names, ids_path=None):
    if path is None:
        return torch.zeros((len(names), 0), dtype=torch.float32)
    if str(path).endswith(".npz"):
        with np.load(path, allow_pickle=False) as saved:
            values, ids = saved["features"], saved["ids"].astype(str).tolist()
    else:
        values = np.load(path, allow_pickle=False)
        if ids_path is None:
            raise ValueError("A .npy feature matrix requires --drug_ids/--gene_ids; use .npz with ids/features instead")
        text = Path(ids_path).read_text(encoding="utf-8")
        ids = json.loads(text) if str(ids_path).endswith(".json") else text.splitlines()
        ids = [str(x).strip() for x in ids]
    if values.ndim != 2 or not values.shape[1] or len(values) != len(ids) or len(set(ids)) != len(ids):
        raise ValueError("Features require a nonempty 2-D matrix and unique matching row IDs")
    if not np.isfinite(values).all():
        raise ValueError("Features contain NaN or infinity")
    positions = {name: i for i, name in enumerate(ids)}
    missing = [name for name in names if name not in positions]
    if missing:
        raise ValueError(f"Missing feature IDs: {missing[:10]}")
    return torch.tensor(values[[positions[name] for name in names]], dtype=torch.float32)


def load_dataset(csv_path, drug_feature_path=None, gene_feature_path=None, drug_ids=None, gene_ids=None):
    df = pd.read_csv(csv_path, dtype={"drug": str, "gene": str})
    if not {"drug", "gene", "sign"}.issubset(df.columns) or df[["drug", "gene", "sign"]].isna().any().any():
        raise ValueError("CSV requires nonmissing drug,gene,sign columns")
    df["drug"], df["gene"] = df.drug.str.strip(), df.gene.str.strip()
    if (df.drug == "").any() or (df.gene == "").any() or not df.sign.isin([-1, 1]).all():
        raise ValueError("Entity names must be nonempty and signs must be exactly -1/+1")
    if df.duplicated(["drug", "gene"]).any():
        raise ValueError("Duplicate drug-gene pairs can leak across folds; deduplicate/filter conflicts before training")
    if len(df) < 2 or df.sign.nunique() != 2:
        raise ValueError("A binary training benchmark requires both observed signs")
    drugs, genes = sorted(df.drug.unique()), sorted(df.gene.unique())
    dm, gm = {name: i for i, name in enumerate(drugs)}, {name: i for i, name in enumerate(genes)}
    pairs = torch.tensor([[dm[d], gm[g]] for d, g in zip(df.drug, df.gene)], dtype=torch.long)
    return CSIHData(drugs, genes, read_features(drug_feature_path, drugs, drug_ids),
                    read_features(gene_feature_path, genes, gene_ids), pairs,
                    torch.tensor((df.sign == 1).to_numpy(), dtype=torch.long), sha256_file(csv_path))


def build_training_graph(pairs, labels):
    return (pairs[labels == 1].T.contiguous(), pairs[labels == 0].T.contiguous())


def context_from_pairs(pairs, num_drugs, num_genes):
    mask = torch.zeros(num_drugs + num_genes, dtype=torch.bool, device=pairs.device)
    mask[pairs[:, 0]] = True
    mask[pairs[:, 1] + num_drugs] = True
    return mask


def sign_preserving_edge_dropout(edge_index, drop_rate):
    if not 0 <= drop_rate < 1:
        raise ValueError("edge dropout must lie in [0,1)")
    if edge_index.size(1) == 0 or drop_rate == 0:
        return edge_index
    keep = torch.rand(edge_index.size(1), device=edge_index.device) >= drop_rate
    if not keep.any():
        keep[torch.randint(edge_index.size(1), (1,), device=edge_index.device)] = True
    return edge_index[:, keep]


def _derangement(active_mask):
    permutation = torch.arange(len(active_mask), device=active_mask.device)
    active = active_mask.nonzero(as_tuple=True)[0]
    if len(active) > 1:
        order = active[torch.randperm(len(active), device=active.device)]
        permutation[order] = order.roll(1)
    return permutation


def corrupt_training_inputs(pos_edges, neg_edges, context_mask, num_drugs, mode="both", sign_flip_rate=0.2):
    """Corrupt raw node inputs and/or signed TRAINING edges before encoding."""
    if mode not in ("both", "feature_shuffle", "sign_flip") or not 0 < sign_flip_rate <= 1:
        raise ValueError("Use a supported corruption mode and sign_flip_rate in (0,1]")
    permutations = None
    if mode in ("both", "feature_shuffle"):
        permutations = (_derangement(context_mask[:num_drugs]), _derangement(context_mask[num_drugs:]))
    p, n = pos_edges, neg_edges
    if mode in ("both", "sign_flip"):
        edges = torch.cat((pos_edges, neg_edges), dim=1)
        positive = torch.arange(edges.size(1), device=edges.device) < pos_edges.size(1)
        count = min(edges.size(1), max(1, round(edges.size(1) * sign_flip_rate)))
        selected = torch.randperm(edges.size(1), device=edges.device)[:count]
        positive[selected] = ~positive[selected]
        p, n = edges[:, positive], edges[:, ~positive]
    return p, n, permutations
