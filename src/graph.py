import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithm import (
    SignAwareRelationEncoder,
    SignedIntentHypergraph,
    LateAdaptiveSignFusion,
)


class FoundationFeatureEncoder(nn.Module):
    """Uni-Mol/ESM-2 features + trainable identity embeddings."""

    def __init__(self, num_drugs, num_genes, drug_dim, gene_dim,
                 hidden_dim=128, identity_dim=32, dropout=0.2):
        super().__init__()
        self.num_drugs = num_drugs
        self.num_genes = num_genes

        self.drug_identity = nn.Embedding(num_drugs, identity_dim)
        self.gene_identity = nn.Embedding(num_genes, identity_dim)

        self.drug_proj = nn.Sequential(
            nn.Linear(drug_dim + identity_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gene_proj = nn.Sequential(
            nn.Linear(gene_dim + identity_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, drug_feat, gene_feat,
                cold_drug_mask=None, cold_gene_mask=None):
        device = drug_feat.device
        d_ids = torch.arange(self.num_drugs, device=device)
        g_ids = torch.arange(self.num_genes, device=device)

        d_id = self.drug_identity(d_ids)
        g_id = self.gene_identity(g_ids)

        if cold_drug_mask is not None:
            d_id = d_id.masked_fill(cold_drug_mask[:, None], 0.0)
        if cold_gene_mask is not None:
            g_id = g_id.masked_fill(cold_gene_mask[:, None], 0.0)

        d = self.drug_proj(torch.cat([drug_feat, d_id], dim=-1))
        g = self.gene_proj(torch.cat([gene_feat, g_id], dim=-1))
        return d, g


class PairDecoder(nn.Module):
    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, drug_z, gene_z, pairs):
        d = drug_z[pairs[:, 0]]
        g = gene_z[pairs[:, 1]]
        x = torch.cat([d, g, d * g], dim=-1)
        return self.mlp(x).squeeze(-1)


class CSIH(nn.Module):
    """Complete CSIH model."""

    def __init__(self, args, num_drugs, num_genes, drug_feature_dim, gene_feature_dim):
        super().__init__()
        self.num_drugs = num_drugs
        self.num_genes = num_genes

        self.feature_encoder = FoundationFeatureEncoder(
            num_drugs, num_genes,
            drug_feature_dim, gene_feature_dim,
            args.hidden_dim, args.identity_dim, args.dropout
        )

        self.relation_encoder = SignAwareRelationEncoder(
            args.hidden_dim, args.layers, args.dropout
        )

        branch_args = (
            args.hidden_dim,
            args.intent_dim,
            args.n_intents,
            args.n_prototypes,
            args.topk,
            args.temperature,
            args.dropout,
        )
        self.pos_hypergraph = SignedIntentHypergraph(*branch_args)
        self.neg_hypergraph = SignedIntentHypergraph(*branch_args)

        self.sign_fusion = LateAdaptiveSignFusion(args.hidden_dim)
        self.decoder = PairDecoder(args.hidden_dim, args.dropout)

    def encode(self, drug_feat, gene_feat, pos_edges, neg_edges,
               cold_drug_mask=None, cold_gene_mask=None):
        d, g = self.feature_encoder(
            drug_feat, gene_feat, cold_drug_mask, cold_gene_mask
        )

        (pd, pg), (nd, ng) = self.relation_encoder(
            d, g, pos_edges, neg_edges
        )

        p = torch.cat([pd, pg], dim=0)
        n = torch.cat([nd, ng], dim=0)

        p_high, p_intents, p_H = self.pos_hypergraph(p)
        n_high, n_intents, n_H = self.neg_hypergraph(n)

        z, sign_weights = self.sign_fusion(p_high, n_high)

        return {
            "drug": z[:self.num_drugs],
            "gene": z[self.num_drugs:],
            "all": z,
            "positive_intents": p_intents,
            "negative_intents": n_intents,
            "positive_incidence": p_H,
            "negative_incidence": n_H,
            "sign_weights": sign_weights,
        }

    def forward(self, drug_feat, gene_feat, pos_edges, neg_edges, pairs,
                cold_drug_mask=None, cold_gene_mask=None):
        out = self.encode(
            drug_feat, gene_feat, pos_edges, neg_edges,
            cold_drug_mask, cold_gene_mask
        )
        logits = self.decoder(out["drug"], out["gene"], pairs)
        out["logits"] = logits
        out["prob_positive"] = torch.sigmoid(logits)
        out["prob_negative"] = 1.0 - out["prob_positive"]
        return out
