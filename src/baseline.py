"""
Simple baselines for controlled comparisons.

These are lightweight reference implementations only.
For publication tables, use the official/reproduced implementations and
identical splits/features described in the manuscript.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBaseline(nn.Module):
    def __init__(self, drug_dim, gene_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.drug_proj = nn.Linear(drug_dim, hidden_dim)
        self.gene_proj = nn.Linear(gene_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, drug_features, gene_features, pairs):
        d = self.drug_proj(drug_features)[pairs[:, 0]]
        g = self.gene_proj(gene_features)[pairs[:, 1]]
        return self.decoder(
            torch.cat([d, g, d * g], dim=-1)
        ).squeeze(-1)
