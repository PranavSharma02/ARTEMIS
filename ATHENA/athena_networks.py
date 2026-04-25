#!/usr/bin/env python3
"""
ATHENA Neural Network Architectures

Modification 1 vs ARTEMIS:
  1a. ATHENAUltra1Model  – adds residual connections between GRU layers
      (He et al., 2016) to improve gradient flow in deep stacks.
  1b. ATHENAReturnModel  – adds sinusoidal positional encoding to the
      Transformer (Shaw et al., 2018) for better temporal awareness.
  All 5 models receive minor hyperparameter adjustments (wider hidden
  dimensions, lower dropout) for increased representational capacity.

Each model exposes a 64-dim feature vector alongside its scalar
prediction so the CrossAttentionFusion module can operate on rich
representations.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Model 1 – Ultra-1: GRU + Attention + Residual Connections
# ---------------------------------------------------------------------------

class ATHENAUltra1Model(nn.Module):
    """
    ATHENA Ultra-1: Hierarchical GRU with multi-head attention and
    residual skip connections between GRU layers.
    """

    def __init__(self, input_size, hidden_size=208, dropout=0.10):
        super().__init__()
        self.hidden_size = hidden_size

        self.gru1 = nn.GRU(input_size, hidden_size, 3, dropout=dropout, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=16, dropout=dropout, batch_first=True)

        self.gru2 = nn.GRU(hidden_size, hidden_size // 2, 2, dropout=dropout, batch_first=True)
        self.gru3 = nn.GRU(hidden_size // 2, hidden_size // 4, 1, dropout=dropout, batch_first=True)

        # Residual projections to align dimensions across skip connections
        self.res_proj1 = nn.Linear(input_size, hidden_size)
        self.res_proj2 = nn.Linear(hidden_size, hidden_size // 2)
        self.res_norm1 = nn.LayerNorm(hidden_size)
        self.res_norm2 = nn.LayerNorm(hidden_size // 2)

        combined_feature_size = (hidden_size // 4) * 3

        self.feature_extractor = nn.Sequential(
            nn.Linear(combined_feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.18),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.14),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
        )

        self.return_head = nn.Linear(64, 1)
        self.momentum_head = nn.Linear(64, 1)
        self.volatility_head = nn.Linear(64, 1)
        self.regime_head = nn.Linear(64, 1)

    def forward(self, x):
        # GRU-1 + residual from input
        gru1_out, _ = self.gru1(x)
        residual1 = self.res_proj1(x)
        gru1_out = self.res_norm1(gru1_out + residual1)

        attn_out, _ = self.attention(gru1_out, gru1_out, gru1_out)

        # GRU-2 + residual from GRU-1
        gru2_out, hidden2 = self.gru2(attn_out)
        residual2 = self.res_proj2(attn_out)
        gru2_out = self.res_norm2(gru2_out + residual2)

        gru3_out, hidden3 = self.gru3(gru2_out)

        final_hidden = hidden3[-1]
        global_features = gru3_out.mean(dim=1)
        max_features = gru3_out.max(dim=1)[0]

        combined = torch.cat([final_hidden, global_features, max_features], dim=1)
        features = self.feature_extractor(combined)

        r = self.return_head(features)
        m = self.momentum_head(features)
        v = self.volatility_head(features)
        g = self.regime_head(features)

        prediction = 1.0 * r + 0.4 * m - 0.1 * v + 0.2 * g
        return prediction

    def extract_features(self, x):
        """Return the 64-dim feature vector (for fusion module)."""
        gru1_out, _ = self.gru1(x)
        residual1 = self.res_proj1(x)
        gru1_out = self.res_norm1(gru1_out + residual1)
        attn_out, _ = self.attention(gru1_out, gru1_out, gru1_out)
        gru2_out, _ = self.gru2(attn_out)
        residual2 = self.res_proj2(attn_out)
        gru2_out = self.res_norm2(gru2_out + residual2)
        gru3_out, hidden3 = self.gru3(gru2_out)
        combined = torch.cat([hidden3[-1], gru3_out.mean(dim=1), gru3_out.max(dim=1)[0]], dim=1)
        return self.feature_extractor(combined)


# ---------------------------------------------------------------------------
# Model 2 – Momentum: LSTM + TCN  (hyperparameter tweaks only)
# ---------------------------------------------------------------------------

class ATHENAMomentumModel(nn.Module):
    """ATHENA Momentum: LSTM + Temporal Convolution for momentum capture."""

    def __init__(self, input_size, hidden_size=176, dropout=0.12):
        super().__init__()
        self.hidden_size = hidden_size

        self.lstm1 = nn.LSTM(input_size, hidden_size, 2, dropout=dropout, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size // 2, 1, dropout=dropout, batch_first=True)

        self.tcn = nn.Conv1d(hidden_size // 2, hidden_size // 2, kernel_size=3, padding=1)
        self.momentum_attention = nn.MultiheadAttention(
            hidden_size // 2, num_heads=8, dropout=dropout, batch_first=True
        )

        combined_feature_size = hidden_size // 2 + hidden_size // 2

        self.momentum_extractor = nn.Sequential(
            nn.Linear(combined_feature_size, 256),
            nn.ReLU(),
            nn.Dropout(0.14),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
        )

        self.momentum_head = nn.Linear(64, 1)
        self.trend_head = nn.Linear(64, 1)
        self.acceleration_head = nn.Linear(64, 1)

    def forward(self, x):
        lstm1_out, _ = self.lstm1(x)
        lstm2_out, (hidden, _) = self.lstm2(lstm1_out)

        tcn_input = lstm2_out.transpose(1, 2)
        tcn_out = F.relu(self.tcn(tcn_input)).transpose(1, 2)

        attn_out, _ = self.momentum_attention(tcn_out, tcn_out, tcn_out)

        final_hidden = hidden[-1]
        momentum_features = attn_out.mean(dim=1)

        combined = torch.cat([final_hidden, momentum_features], dim=1)
        features = self.momentum_extractor(combined)

        m = self.momentum_head(features)
        t = self.trend_head(features)
        a = self.acceleration_head(features)

        return 0.6 * m + 0.3 * t + 0.1 * a

    def extract_features(self, x):
        lstm1_out, _ = self.lstm1(x)
        lstm2_out, (hidden, _) = self.lstm2(lstm1_out)
        tcn_out = F.relu(self.tcn(lstm2_out.transpose(1, 2))).transpose(1, 2)
        attn_out, _ = self.momentum_attention(tcn_out, tcn_out, tcn_out)
        combined = torch.cat([hidden[-1], attn_out.mean(dim=1)], dim=1)
        return self.momentum_extractor(combined)


# ---------------------------------------------------------------------------
# Model 3 – Return: Transformer + Sinusoidal Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for the Transformer model."""

    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ATHENAReturnModel(nn.Module):
    """
    ATHENA Return: Transformer with explicit sinusoidal positional
    encoding for improved temporal awareness.
    """

    def __init__(self, input_size, hidden_size=168, dropout=0.13):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, max_len=200, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.return_processor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
        )

        self.short_return_head = nn.Linear(64, 1)
        self.medium_return_head = nn.Linear(64, 1)
        self.long_return_head = nn.Linear(64, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.pos_encoder(embedded)

        transformer_out = self.transformer(embedded)
        return_features = transformer_out.mean(dim=1)
        features = self.return_processor(return_features)

        s = self.short_return_head(features)
        m = self.medium_return_head(features)
        l = self.long_return_head(features)

        return 0.5 * s + 0.3 * m + 0.2 * l

    def extract_features(self, x):
        embedded = self.pos_encoder(self.embedding(x))
        transformer_out = self.transformer(embedded)
        return self.return_processor(transformer_out.mean(dim=1))


# ---------------------------------------------------------------------------
# Model 4 – Trend: CNN + BiLSTM  (hyperparameter tweaks only)
# ---------------------------------------------------------------------------

class ATHENATrendModel(nn.Module):
    """ATHENA Trend: CNN + Bidirectional LSTM for trend analysis."""

    def __init__(self, input_size, hidden_size=152, dropout=0.14):
        super().__init__()
        self.hidden_size = hidden_size

        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, hidden_size, kernel_size=7, padding=3)

        self.bilstm = nn.LSTM(
            hidden_size, hidden_size // 2, 2,
            dropout=dropout, batch_first=True, bidirectional=True,
        )

        self.trend_analyzer = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.12),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
        )

        self.trend_direction_head = nn.Linear(64, 1)
        self.trend_strength_head = nn.Linear(64, 1)
        self.trend_persistence_head = nn.Linear(64, 1)

    def forward(self, x):
        xt = x.transpose(1, 2)
        c1 = F.relu(self.conv1(xt))
        c2 = F.relu(self.conv2(c1))
        c3 = F.relu(self.conv3(c2))
        cnn_out = c3.transpose(1, 2)

        bilstm_out, (hidden, _) = self.bilstm(cnn_out)
        trend_features = bilstm_out.mean(dim=1)
        features = self.trend_analyzer(trend_features)

        d = self.trend_direction_head(features)
        s = self.trend_strength_head(features)
        p = self.trend_persistence_head(features)

        return 0.5 * d + 0.3 * s + 0.2 * p

    def extract_features(self, x):
        xt = x.transpose(1, 2)
        c = F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(xt))))))
        bilstm_out, _ = self.bilstm(c.transpose(1, 2))
        return self.trend_analyzer(bilstm_out.mean(dim=1))


# ---------------------------------------------------------------------------
# Model 5 – HF: Multi-Scale CNN + GRU  (hyperparameter tweaks only)
# ---------------------------------------------------------------------------

class ATHENAHFModel(nn.Module):
    """ATHENA HF: Multi-scale 1-D CNN + GRU for short-term pattern detection."""

    def __init__(self, input_size, hidden_size=136, dropout=0.15):
        super().__init__()
        self.hidden_size = hidden_size

        self.conv_1min = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.conv_5min = nn.Conv1d(input_size, 32, kernel_size=5, padding=2)
        self.conv_15min = nn.Conv1d(input_size, 32, kernel_size=7, padding=3)
        self.conv_60min = nn.Conv1d(input_size, 32, kernel_size=15, padding=7)

        combined_features = 128

        self.gru = nn.GRU(combined_features, hidden_size, 3, dropout=dropout, batch_first=True)

        self.hf_detector = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.14),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
        )

        self.microtrend_head = nn.Linear(64, 1)
        self.volatility_head = nn.Linear(64, 1)
        self.noise_head = nn.Linear(64, 1)

    def forward(self, x):
        xt = x.transpose(1, 2)
        c1 = F.relu(self.conv_1min(xt))
        c5 = F.relu(self.conv_5min(xt))
        c15 = F.relu(self.conv_15min(xt))
        c60 = F.relu(self.conv_60min(xt))

        multi_scale = torch.cat([c1, c5, c15, c60], dim=1).transpose(1, 2)
        gru_out, hidden = self.gru(multi_scale)

        hf_features = hidden[-1]
        features = self.hf_detector(hf_features)

        mt = self.microtrend_head(features)
        vl = self.volatility_head(features)
        ns = self.noise_head(features)

        return 0.6 * mt - 0.2 * vl - 0.2 * ns

    def extract_features(self, x):
        xt = x.transpose(1, 2)
        ms = torch.cat([
            F.relu(self.conv_1min(xt)),
            F.relu(self.conv_5min(xt)),
            F.relu(self.conv_15min(xt)),
            F.relu(self.conv_60min(xt)),
        ], dim=1).transpose(1, 2)
        _, hidden = self.gru(ms)
        return self.hf_detector(hidden[-1])


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def get_athena_model_configs():
    return [
        {'model_class': ATHENAUltra1Model, 'name': 'ATHENA-Ultra-1',
         'hidden_size': 208, 'dropout': 0.10, 'ultra_aggressive': True},
        {'model_class': ATHENAMomentumModel, 'name': 'ATHENA-Momentum',
         'hidden_size': 176, 'dropout': 0.12, 'momentum_focus': True},
        {'model_class': ATHENAReturnModel, 'name': 'ATHENA-Return',
         'hidden_size': 168, 'dropout': 0.13, 'return_focus': True},
        {'model_class': ATHENATrendModel, 'name': 'ATHENA-Trend',
         'hidden_size': 152, 'dropout': 0.14, 'trend_focus': True},
        {'model_class': ATHENAHFModel, 'name': 'ATHENA-HF',
         'hidden_size': 136, 'dropout': 0.15, 'high_freq': True},
    ]


def create_athena_model(config, input_size):
    return config['model_class'](
        input_size=input_size,
        hidden_size=config['hidden_size'],
        dropout=config['dropout'],
    )
