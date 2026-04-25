#!/usr/bin/env python3
"""
ATHENA Dual-Horizon Signal Generator

Modification 4 vs ARTEMIS: Instead of a single-horizon trading signal,
ATHENA generates short-term (1-5 day) and medium-term (5-20 day) signals
and blends them through a learned gating mechanism.

Reference: Dauphin et al. (2017), "Language Modeling with Gated
Convolutional Networks"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualHorizonSignalGenerator(nn.Module):
    """
    Generates a blended trading signal from short-term and medium-term
    prediction heads with a learned gating network.

    final_signal = gate * short_signal + (1 - gate) * medium_signal
    """

    def __init__(self, input_dim=64, hidden_dim=64, dropout=0.1):
        super().__init__()

        self.short_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.medium_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.gate_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, features):
        """
        features : (B, input_dim)
        Returns  : (B, 1) blended signal, (B, 1) gate value
        """
        short_signal = self.short_head(features)
        medium_signal = self.medium_head(features)
        gate = self.gate_network(features)

        blended = gate * short_signal + (1.0 - gate) * medium_signal
        return blended, gate


class DualHorizonLoss(nn.Module):
    """
    Training loss for the dual-horizon generator.

    Evaluates predictions against short-term returns (next-day) and
    medium-term returns (5-day forward average), then combines with
    the standard return-maximisation objective.
    """

    def __init__(self, alpha=0.03, short_weight=0.6, medium_weight=0.4):
        super().__init__()
        self.alpha = alpha
        self.short_weight = short_weight
        self.medium_weight = medium_weight

    def forward(self, short_pred, medium_pred, gate, returns):
        """
        short_pred  : (B, 1)
        medium_pred : (B, 1)
        gate        : (B, 1)
        returns     : (B,) next-day returns used as primary target
        """
        short_signal = torch.tanh(short_pred.squeeze())
        medium_signal = torch.tanh(medium_pred.squeeze())
        blended_signal = torch.tanh(
            (gate.squeeze() * short_pred.squeeze() +
             (1 - gate.squeeze()) * medium_pred.squeeze())
        )

        short_pnl = short_signal * returns
        medium_pnl = medium_signal * returns
        blended_pnl = blended_signal * returns

        def _sharpe(pnl):
            return torch.mean(pnl) / (torch.std(pnl) + 1e-8)

        short_obj = (
            0.30 * _sharpe(short_pnl) +
            0.30 * torch.mean(torch.abs(short_pnl)) +
            0.20 * torch.mean((short_pnl > 0).float()) +
            0.20 * torch.mean(torch.relu(short_pnl - 0.005))
        )
        medium_obj = (
            0.30 * _sharpe(medium_pnl) +
            0.30 * torch.mean(torch.abs(medium_pnl)) +
            0.20 * torch.mean((medium_pnl > 0).float()) +
            0.20 * torch.mean(torch.relu(medium_pnl - 0.005))
        )
        blended_obj = (
            0.35 * _sharpe(blended_pnl) +
            0.35 * torch.mean(torch.abs(blended_pnl)) +
            0.15 * torch.mean((blended_pnl > 0).float()) +
            0.15 * torch.mean(torch.relu(blended_pnl - 0.005))
        )

        sorted_blended, _ = torch.sort(blended_pnl)
        n = max(1, int(len(sorted_blended) * self.alpha))
        cvar = -torch.mean(sorted_blended[:n])

        total_obj = (
            self.short_weight * short_obj +
            self.medium_weight * medium_obj +
            0.5 * blended_obj
        )

        return -(total_obj - 0.02 * cvar)
