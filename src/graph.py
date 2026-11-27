"""CSIH, including controlled early fusion and feature-source variants."""
import torch
from torch import nn
import torch.nn.functional as F
try:
    from .algorithm import SignAwareRelationEncoder, SignedIntentHypergraph, LateAdaptiveSignFusion
except ImportError:
    from algorithm import SignAwareRelationEncoder, SignedIntentHypergraph, LateAdaptiveSignFusion


class FoundationFeatureEncoder(nn.Module):
    def __init__(self, num_drugs, num_genes, drug_dim, gene_dim, args):
        super().__init__()
        self.num_drugs, self.num_genes = num_drugs, num_genes
        self.use_drug_features = args.use_drug_features
        self.use_gene_features = args.use_gene_features
        self.drug_identity = nn.Embedding(num_drugs, args.identity_dim) if args.use_id_embeddings else None
        self.gene_identity = nn.Embedding(num_genes, args.identity_dim) if args.use_id_embeddings else None
        dd = (drug_dim if args.use_drug_features else 0) + (args.identity_dim if args.use_id_embeddings else 0)
        gd = (gene_dim if args.use_gene_features else 0) + (args.identity_dim if args.use_id_embeddings else 0)
        if dd == 0 or gd == 0:
            raise ValueError("Each node type requires a retained input source.")
        self.drug_proj = nn.Sequential(nn.Linear(dd, args.hidden_dim), nn.ReLU(), nn.Dropout(args.dropout))
        self.gene_proj = nn.Sequential(nn.Linear(gd, args.hidden_dim), nn.ReLU(), nn.Dropout(args.dropout))

    def _input(self, feat, identity, use_fm, cold_mask, permutation, mask_rate):
        parts = [feat] if use_fm else []
        if identity is not None:
            ids = identity(torch.arange(len(feat), device=feat.device))
            if cold_mask is not None:
                ids = ids.masked_fill(cold_mask[:, None], 0)
            parts.append(ids)
        raw = torch.cat(parts, -1)
        if permutation is not None:
            raw = raw[permutation]
        if mask_rate:
            raw = raw * (torch.rand_like(raw) >= mask_rate).to(raw.dtype)
        return raw

    def forward(self, drug_feat, gene_feat, cold_drug_mask=None, cold_gene_mask=None,
                permutations=None, mask_rate=0.0):
        dp, gp = (None, None) if permutations is None else permutations
        d = self._input(drug_feat, self.drug_identity, self.use_drug_features, cold_drug_mask, dp, mask_rate)
        g = self._input(gene_feat, self.gene_identity, self.use_gene_features, cold_gene_mask, gp, mask_rate)
        return self.drug_proj(d), self.gene_proj(g)


class PairDecoder(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(4 * hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden_dim, 2))

    def forward(self, drug_z, gene_z, pairs):
        d, g = drug_z[pairs[:, 0]], gene_z[pairs[:, 1]]
        return self.mlp(torch.cat((d, g, d * g, (d - g).abs()), -1))


class CSIH(nn.Module):
    def __init__(self, args, num_drugs, num_genes, drug_feature_dim, gene_feature_dim):
        super().__init__()
        self.num_drugs, self.num_genes = num_drugs, num_genes
        self.early_fusion = args.early_fusion
        self.use_hypergraph = args.use_hypergraph
        self.feature_encoder = FoundationFeatureEncoder(num_drugs, num_genes, drug_feature_dim, gene_feature_dim, args)
        self.relation_encoder = SignAwareRelationEncoder(args.hidden_dim, args.layers, args.dropout)
        if self.use_hypergraph:
            branch_args = (args.hidden_dim, args.intent_dim, args.n_intents, args.n_prototypes,
                           args.topk, args.hypergraph_temperature, args.dropout)
            self.pos_hypergraph = SignedIntentHypergraph(*branch_args)
            self.neg_hypergraph = SignedIntentHypergraph(*branch_args)
        self.sign_fusion = LateAdaptiveSignFusion(args.hidden_dim)
        self.decoder = PairDecoder(args.hidden_dim, args.dropout)

    def encode(self, drug_feat, gene_feat, pos_edges, neg_edges, context_mask=None,
               permutations=None, mask_rate=0.0, include_unseen=None):
        size = self.num_drugs + self.num_genes
        if context_mask is None:
            context_mask = torch.ones(size, dtype=torch.bool, device=drug_feat.device)
        if context_mask.dtype != torch.bool or context_mask.numel() != size or not context_mask.any():
            raise ValueError("context_mask must be a nonempty boolean mask over drugs followed by genes")
        for edges in (pos_edges, neg_edges):
            if edges.ndim != 2 or edges.size(0) != 2:
                raise ValueError("Graph edges must have shape [2,E]")
            if edges.numel() and not (context_mask[edges[0]] & context_mask[edges[1] + self.num_drugs]).all():
                raise ValueError("Graph contains an edge incident to a held-out entity")
        if include_unseen is None:
            include_unseen = not self.training
        dc, gc = ~context_mask[:self.num_drugs], ~context_mask[self.num_drugs:]
        d, g = self.feature_encoder(drug_feat, gene_feat, dc, gc, permutations, mask_rate)
        initial = torch.cat((d, g), 0)
        (pd, pg), (nd, ng) = self.relation_encoder(d, g, pos_edges, neg_edges)
        p, n = torch.cat((pd, pg), 0), torch.cat((nd, ng), 0)
        if self.early_fusion:
            # Same parameter count as full CSIH; only the hypergraph inputs change.
            p = n = (p + n) / 2
        ctx = context_mask.nonzero(as_tuple=True)[0]
        query = (~context_mask).nonzero(as_tuple=True)[0] if include_unseen else ctx[:0]
        p_intents, n_intents, p_inc, n_inc = [], [], [], []
        if self.use_hypergraph:
            ph, p_intents, p_inc, pq = self.pos_hypergraph(p[ctx], p[query] if len(query) else None)
            nh, n_intents, n_inc, nq = self.neg_hypergraph(n[ctx], n[query] if len(query) else None)
        else:
            ph, nh = p[ctx], n[ctx]
            pq, nq = p[query], n[query]
        zc, ac = self.sign_fusion(ph, nh, initial[ctx])
        z = initial.new_zeros((size, zc.size(1))).index_copy(0, ctx, zc)
        alpha = initial.new_zeros((size, 2)).index_copy(0, ctx, ac)
        if len(query):
            zq, aq = self.sign_fusion(pq, nq, initial[query])
            z = z.index_copy(0, query, zq)
            alpha = alpha.index_copy(0, query, aq)
        return {"drug": z[:self.num_drugs], "gene": z[self.num_drugs:], "node_embeddings": z,
                "all": z[ctx], "context_indices": ctx, "positive_intents": p_intents,
                "negative_intents": n_intents, "positive_incidence": p_inc,
                "negative_incidence": n_inc, "sign_weights": alpha}

    def forward(self, drug_feat, gene_feat, pos_edges, neg_edges, pairs, **kwargs):
        out = self.encode(drug_feat, gene_feat, pos_edges, neg_edges, **kwargs)
        out["logits"] = self.decoder(out["drug"], out["gene"], pairs)
        probabilities = out["logits"].softmax(-1)
        out["prob_positive"], out["prob_negative"] = probabilities[:, 1], probabilities[:, 0]
        return out
