from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch


@dataclass
class CSIHData:
    drug_names: list
    gene_names: list
    drug_features: torch.Tensor
    gene_features: torch.Tensor
    pairs: torch.Tensor
    labels: torch.Tensor


def load_dataset(csv_path, drug_feature_path=None, gene_feature_path=None):
    df = pd.read_csv(csv_path)
    required = {"drug", "gene", "sign"}
    if not required.issubset(df.columns):
        raise ValueError("CSV must contain drug, gene, sign.")

    drugs = sorted(df["drug"].astype(str).unique())
    genes = sorted(df["gene"].astype(str).unique())

    dmap = {name: i for i, name in enumerate(drugs)}
    gmap = {name: i for i, name in enumerate(genes)}

    pairs, labels = [], []
    for row in df.itertuples(index=False):
        sign = int(row.sign)
        if sign not in (-1, 1):
            raise ValueError("sign must be -1 or +1.")
        pairs.append([dmap[str(row.drug)], gmap[str(row.gene)]])
        labels.append(1 if sign == 1 else 0)

    if drug_feature_path is not None:
        drug_features = np.load(drug_feature_path)
    else:
        drug_features = np.eye(len(drugs), dtype=np.float32)

    if gene_feature_path is not None:
        gene_features = np.load(gene_feature_path)
    else:
        gene_features = np.eye(len(genes), dtype=np.float32)

    return CSIHData(
        drug_names=drugs,
        gene_names=genes,
        drug_features=torch.tensor(drug_features, dtype=torch.float32),
        gene_features=torch.tensor(gene_features, dtype=torch.float32),
        pairs=torch.tensor(pairs, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.float32),
    )


def build_training_graph(pairs, labels):
    """Only training edges are used for message passing."""
    pos = pairs[labels == 1]
    neg = pairs[labels == 0]

    pos_edges = pos.T.contiguous() if len(pos) else torch.empty((2, 0), dtype=torch.long)
    neg_edges = neg.T.contiguous() if len(neg) else torch.empty((2, 0), dtype=torch.long)
    return pos_edges, neg_edges


def sign_preserving_edge_dropout(edge_index, drop_rate):
    if edge_index.size(1) == 0 or drop_rate <= 0:
        return edge_index
    keep = torch.rand(edge_index.size(1), device=edge_index.device) >= drop_rate
    if keep.sum() == 0:
        keep[0] = True
    return edge_index[:, keep]
