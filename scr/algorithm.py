"""Sign-separated graph encoding and sparse prototype hypergraphs."""
import torch
from torch import nn
import torch.nn.functional as F


class BipartiteSignAggregator(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gene_to_drug = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.drug_to_gene = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.drug_update = nn.Linear(2 * hidden_dim, hidden_dim)
        self.gene_update = nn.Linear(2 * hidden_dim, hidden_dim)

    @staticmethod
    def _aggregate(src_x, src_ids, dst_ids, n_dst):
        out = src_x.new_zeros((n_dst, src_x.size(1)))
        if src_ids.numel() == 0:
            return out
        ds = torch.bincount(src_ids, minlength=src_x.size(0)).to(src_x.dtype)
        dt = torch.bincount(dst_ids, minlength=n_dst).to(src_x.dtype)
        weights = (ds[src_ids] * dt[dst_ids]).clamp_min(1).rsqrt()
        return out.index_add(0, dst_ids, src_x[src_ids] * weights[:, None])

    def forward(self, drug_x, gene_x, edges):
        di, gi = edges
        dm = self._aggregate(gene_x, gi, di, len(drug_x))
        gm = self._aggregate(drug_x, di, gi, len(gene_x))
        return (drug_x + F.relu(self.drug_update(torch.cat((drug_x, self.gene_to_drug(dm)), -1))),
                gene_x + F.relu(self.gene_update(torch.cat((gene_x, self.drug_to_gene(gm)), -1))))


class SignAwareRelationEncoder(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.pos_layers = nn.ModuleList(BipartiteSignAggregator(hidden_dim) for _ in range(num_layers))
        self.neg_layers = nn.ModuleList(BipartiteSignAggregator(hidden_dim) for _ in range(num_layers))
        self.dropout = nn.Dropout(dropout)

    def _branch(self, d, g, edges, layers):
        for layer in layers:
            d, g = layer(d, g, edges)
            d, g = self.dropout(d), self.dropout(g)
        return d, g

    def forward(self, d, g, pos_edges, neg_edges):
        return (self._branch(d, g, pos_edges, self.pos_layers),
                self._branch(d, g, neg_edges, self.neg_layers))


class IntentHypergraphChannel(nn.Module):
    """Queries use only hyperedge messages/degrees computed on context nodes."""
    def __init__(self, hidden_dim, intent_dim, num_prototypes, topk, temperature, dropout):
        super().__init__()
        self.project = nn.Linear(hidden_dim, intent_dim)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, intent_dim) * 0.02)
        self.hyperedge_weight = nn.Parameter(torch.ones(num_prototypes))
        self.transform = nn.Linear(intent_dim, intent_dim, bias=False)
        self.topk, self.temperature = topk, temperature
        self.dropout = nn.Dropout(dropout)

    def build_incidence(self, z):
        scores = F.normalize(z, dim=-1) @ F.normalize(self.prototypes, dim=-1).T
        scores = scores / self.temperature
        values, columns = scores.topk(min(self.topk, scores.size(1)), dim=1)
        weights = values.softmax(dim=1)
        rows = torch.arange(len(z), device=z.device)[:, None].expand_as(columns)
        return torch.sparse_coo_tensor(torch.stack((rows.flatten(), columns.flatten())),
                                       weights.flatten(), scores.shape, device=z.device).coalesce()

    def _node_degree(self, incidence, weights):
        return torch.sparse.mm(incidence, weights[:, None]).squeeze(1).clamp_min(1e-8)

    def forward(self, context_x, query_x=None):
        z = F.relu(self.project(context_x))
        incidence = self.build_incidence(z)
        w = F.softplus(self.hyperedge_weight) + 1e-8
        dv = self._node_degree(incidence, w)
        de = torch.sparse.sum(incidence, dim=0).to_dense().clamp_min(1e-8)
        edge_message = torch.sparse.mm(incidence.transpose(0, 1), z * dv.rsqrt()[:, None])
        edge_message = edge_message * (w / de)[:, None]
        message = torch.sparse.mm(incidence, edge_message) * dv.rsqrt()[:, None]
        result = z + self.dropout(F.relu(self.transform(message)))
        query_result = None
        if query_x is not None and len(query_x):
            qz = F.relu(self.project(query_x))
            qh = self.build_incidence(qz)
            qdv = self._node_degree(qh, w)
            qm = torch.sparse.mm(qh, edge_message) * qdv.rsqrt()[:, None]
            query_result = qz + self.dropout(F.relu(self.transform(qm)))
        return result, incidence, query_result


class SignedIntentHypergraph(nn.Module):
    def __init__(self, hidden_dim, intent_dim, num_intents, num_prototypes, topk, temperature, dropout):
        super().__init__()
        self.channels = nn.ModuleList(IntentHypergraphChannel(hidden_dim, intent_dim, num_prototypes,
                                                              topk, temperature, dropout)
                                      for _ in range(num_intents))
        self.fusion = nn.Sequential(nn.Linear(intent_dim * num_intents, hidden_dim), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, context_x, query_x=None):
        reps, incidences, query_reps = [], [], []
        for channel in self.channels:
            rep, incidence, qr = channel(context_x, query_x)
            reps.append(rep)
            incidences.append(incidence)
            if qr is not None:
                query_reps.append(qr)
        fused = self.fusion(torch.cat(reps, dim=-1))
        query_fused = self.fusion(torch.cat(query_reps, -1)) if query_reps else None
        return fused, reps, incidences, query_fused


class LateAdaptiveSignFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(3 * hidden_dim, 2)
        self.output = nn.Linear(3 * hidden_dim, hidden_dim)

    def forward(self, pos_h, neg_h, initial_x):
        alpha = self.gate(torch.cat((pos_h, neg_h, initial_x), -1)).softmax(dim=-1)
        z = F.relu(self.output(torch.cat((initial_x, alpha[:, :1] * pos_h, alpha[:, 1:] * neg_h), -1)))
        return z, alpha
