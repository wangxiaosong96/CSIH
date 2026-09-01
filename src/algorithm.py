import torch
import torch.nn as nn
import torch.nn.functional as F


class BipartiteSignAggregator(nn.Module):
    """Sign-specific drug-gene message propagation."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.gene_to_drug = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.drug_to_gene = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.self_drug = nn.Linear(hidden_dim, hidden_dim)
        self.self_gene = nn.Linear(hidden_dim, hidden_dim)

    @staticmethod
    def _aggregate(src_x, src_ids, dst_ids, n_dst):
        out = src_x.new_zeros((n_dst, src_x.size(-1)))
        deg = src_x.new_zeros((n_dst,))
        if src_ids.numel() == 0:
            return out
        out.index_add_(0, dst_ids, src_x[src_ids])
        deg.index_add_(0, dst_ids, torch.ones_like(dst_ids, dtype=src_x.dtype))
        return out / deg.clamp_min(1.0).unsqueeze(-1)

    def forward(self, drug_x, gene_x, edge_index):
        d_idx, g_idx = edge_index
        d_msg = self._aggregate(gene_x, g_idx, d_idx, drug_x.size(0))
        g_msg = self._aggregate(drug_x, d_idx, g_idx, gene_x.size(0))
        d_out = F.relu(self.self_drug(drug_x) + self.gene_to_drug(d_msg)) + drug_x
        g_out = F.relu(self.self_gene(gene_x) + self.drug_to_gene(g_msg)) + gene_x
        return d_out, g_out


class SignAwareRelationEncoder(nn.Module):
    """Positive and negative branches remain separated."""

    def __init__(self, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.pos_layers = nn.ModuleList(
            [BipartiteSignAggregator(hidden_dim) for _ in range(num_layers)]
        )
        self.neg_layers = nn.ModuleList(
            [BipartiteSignAggregator(hidden_dim) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def _branch(self, d, g, edge_index, layers):
        for layer in layers:
            d, g = layer(d, g, edge_index)
            d, g = self.dropout(d), self.dropout(g)
        return d, g

    def forward(self, d, g, pos_edges, neg_edges):
        pos = self._branch(d, g, pos_edges, self.pos_layers)
        neg = self._branch(d, g, neg_edges, self.neg_layers)
        return pos, neg


class IntentHypergraphChannel(nn.Module):
    """One sign-specific latent intent hypergraph channel."""

    def __init__(self, hidden_dim, intent_dim, num_prototypes, topk, temperature, dropout):
        super().__init__()
        self.project = nn.Linear(hidden_dim, intent_dim)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, intent_dim) * 0.02)
        self.hyperedge_weight = nn.Parameter(torch.ones(num_prototypes))
        self.out = nn.Linear(intent_dim, hidden_dim)
        self.topk = topk
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def build_incidence(self, x):
        z = F.normalize(self.project(x), dim=-1)
        p = F.normalize(self.prototypes, dim=-1)
        scores = z @ p.T / self.temperature
        k = min(self.topk, scores.size(1))
        values, indices = torch.topk(scores, k=k, dim=1)
        weights = F.softmax(values, dim=1)
        H = torch.zeros_like(scores)
        H.scatter_(1, indices, weights)
        return H

    def propagate(self, x, H):
        z = self.project(x)
        w = F.softplus(self.hyperedge_weight) + 1e-8

        weighted_H = H * w.unsqueeze(0)
        dv = weighted_H.sum(dim=1).clamp_min(1e-8)
        de = H.sum(dim=0).clamp_min(1e-8)

        z = z / torch.sqrt(dv).unsqueeze(-1)
        e = H.T @ z
        e = e / de.unsqueeze(-1)
        e = e * w.unsqueeze(-1)

        out = H @ e
        out = out / torch.sqrt(dv).unsqueeze(-1)
        out = self.out(out)
        return x + self.dropout(F.relu(out))

    def forward(self, x):
        H = self.build_incidence(x)
        return self.propagate(x, H), H


class SignedIntentHypergraph(nn.Module):
    """Multi-intent hypergraph branch for one regulatory sign."""

    def __init__(self, hidden_dim, intent_dim, num_intents, num_prototypes,
                 topk, temperature, dropout):
        super().__init__()
        self.channels = nn.ModuleList([
            IntentHypergraphChannel(
                hidden_dim, intent_dim, num_prototypes, topk, temperature, dropout
            )
            for _ in range(num_intents)
        ])
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_intents, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        reps, incidences = [], []
        for channel in self.channels:
            h, H = channel(x)
            reps.append(h)
            incidences.append(H)
        fused = self.fusion(torch.cat(reps, dim=-1))
        return fused, reps, incidences


class LateAdaptiveSignFusion(nn.Module):
    """Fuse positive and negative high-order representations only at the end."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, pos_h, neg_h):
        alpha = F.softmax(self.gate(torch.cat([pos_h, neg_h], dim=-1)), dim=-1)
        z = alpha[:, :1] * pos_h + alpha[:, 1:] * neg_h
        return z, alpha
