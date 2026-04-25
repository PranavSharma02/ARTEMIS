#!/usr/bin/env python3
"""
ATHENA Cross-Attention Fusion Network

Modification 3 vs ARTEMIS: Replaces the static regime-weighted average
ensemble combination with a learned cross-attention fusion module.
The market state attends to each model's prediction embedding and
produces a context-aware fused signal.

Reference: Vaswani et al. (2017), "Attention Is All You Need"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """
    Fuses N model predictions using cross-attention conditioned on market
    context.

    Inputs
    ------
    model_features : (batch, n_models, feature_dim)
        Feature vectors from each ensemble model.
    market_state   : (batch, market_dim)
        Current market context features.

    Outputs
    -------
    fused_prediction : (batch, 1)
    attention_weights : (batch, n_models)   – for interpretability
    """

    def __init__(self, n_models=5, feature_dim=64, market_dim=20,
                 hidden_dim=128, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_models = n_models
        self.feature_dim = feature_dim

        self.model_proj = nn.Linear(feature_dim, hidden_dim)
        self.market_proj = nn.Linear(market_dim, hidden_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.weight_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_models),
        )

    def forward(self, model_features, market_state):
        # model_features: (B, N, F) -> project to hidden_dim
        model_emb = self.model_proj(model_features)          # (B, N, H)

        # market_state: (B, M) -> (B, 1, H)  query token
        market_query = self.market_proj(market_state).unsqueeze(1)  # (B, 1, H)

        # Cross-attention: market query attends to model embeddings
        attn_out, attn_weights_raw = self.cross_attention(
            query=market_query,
            key=model_emb,
            value=model_emb,
        )
        attn_out = self.attn_norm(attn_out + market_query)   # (B, 1, H)

        context = attn_out.squeeze(1)                        # (B, H)

        fused_prediction = self.fusion_head(context)         # (B, 1)

        interpretable_weights = F.softmax(
            self.weight_head(context), dim=-1
        )                                                    # (B, N)

        return fused_prediction, interpretable_weights


class SimpleFusion(nn.Module):
    """
    Lightweight fallback fusion that works on scalar predictions +
    market features (used during signal generation when full feature
    extraction is not available).
    """

    def __init__(self, n_models=5, market_dim=20, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_models + market_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.weight_net = nn.Sequential(
            nn.Linear(n_models + market_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_models),
        )

    def forward(self, predictions, market_features):
        """
        predictions     : (B, n_models) – scalar predictions
        market_features : (B, market_dim)
        """
        combined = torch.cat([predictions, market_features], dim=-1)
        fused = self.net(combined)
        weights = F.softmax(self.weight_net(combined), dim=-1)
        return fused, weights
