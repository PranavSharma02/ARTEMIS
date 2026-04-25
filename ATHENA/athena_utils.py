#!/usr/bin/env python3
"""
ATHENA Utility Module

Contains loss functions, model configurations, training parameters,
regime weight tables, and performance estimation helpers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tabulate import tabulate
from matplotlib import pyplot as plt


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

class ATHENAReturnBoostLoss(nn.Module):
    """Return-maximising loss with CVaR risk penalty (same family as ARTEMIS)."""

    def __init__(self, alpha=0.03, return_weight=0.95):
        super().__init__()
        self.alpha = alpha
        self.return_weight = return_weight

    def forward(self, predictions, returns):
        signals = torch.tanh(predictions)
        pnl = returns * signals.squeeze()

        sharpe = torch.mean(pnl) / (torch.std(pnl) + 1e-8)
        mag = torch.mean(torch.abs(pnl))
        hit = torch.mean((pnl > 0).float())
        big = torch.mean(torch.relu(pnl - 0.01))

        if len(pnl) > 1:
            consistency = torch.mean(
                torch.sign(pnl[1:]) * torch.sign(pnl[:-1])
            )
            accel = torch.mean(torch.relu(pnl[1:] - pnl[:-1]))
        else:
            consistency = torch.tensor(0.0, device=signals.device)
            accel = torch.tensor(0.0, device=signals.device)

        sorted_pnl, _ = torch.sort(pnl)
        n = max(1, int(len(sorted_pnl) * self.alpha))
        cvar = -torch.mean(sorted_pnl[:n])

        churn = torch.tensor(0.0, device=signals.device)
        if len(signals) > 1:
            churn = torch.mean(torch.abs(signals[1:] - signals[:-1]))

        objective = (
            0.25 * sharpe + 0.35 * mag + 0.20 * hit +
            0.15 * big + 0.10 * consistency + 0.08 * accel
        )
        penalty = 0.02 * cvar + 0.005 * churn

        return -(self.return_weight * objective - penalty)


# ---------------------------------------------------------------------------
# Regime Detection (same structure as ARTEMIS)
# ---------------------------------------------------------------------------

class ATHENARegimeDetector(nn.Module):
    def __init__(self, input_size, hidden_size=80):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, 2, dropout=0.1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
        )

    def forward(self, x):
        _, hidden = self.gru(x)
        logits = self.classifier(hidden[-1])
        return F.softmax(logits, dim=1)


# ---------------------------------------------------------------------------
# Position Sizer (same structure as ARTEMIS)
# ---------------------------------------------------------------------------

class ATHENAPositionSizer(nn.Module):
    def __init__(self, input_size=20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, market_features):
        return 0.5 + self.net(market_features)


# ---------------------------------------------------------------------------
# Model Configs
# ---------------------------------------------------------------------------

def create_athena_model_configs():
    """Flat config list used by the core training loop."""
    return [
        {'hidden_size': 208, 'dropout': 0.10, 'use_attention': True,
         'ultra_aggressive': True, 'name': 'ATHENA-Ultra-1'},
        {'hidden_size': 176, 'dropout': 0.12, 'use_attention': True,
         'momentum_focus': True, 'aggressive': True, 'name': 'ATHENA-Momentum'},
        {'hidden_size': 168, 'dropout': 0.13, 'use_attention': True,
         'return_focus': True, 'aggressive': True, 'name': 'ATHENA-Return'},
        {'hidden_size': 152, 'dropout': 0.14, 'use_attention': True,
         'trend_focus': True, 'aggressive': True, 'name': 'ATHENA-Trend'},
        {'hidden_size': 136, 'dropout': 0.15, 'use_attention': True,
         'high_freq': True, 'ultra_aggressive': True, 'name': 'ATHENA-HF'},
    ]


def get_athena_regime_weights():
    """Regime → per-model weight vector (fallback for SimpleFusion)."""
    return {
        0: [0.35, 0.25, 0.20, 0.12, 0.08],
        1: [0.30, 0.30, 0.25, 0.10, 0.05],
        2: [0.25, 0.25, 0.30, 0.15, 0.05],
        3: [0.20, 0.25, 0.25, 0.25, 0.05],
        4: [0.15, 0.20, 0.25, 0.20, 0.20],
    }


# ---------------------------------------------------------------------------
# Training Parameters
# ---------------------------------------------------------------------------

ATHENA_TRAINING_PARAMS = {
    'learning_rate': 2e-4,
    'learning_rates': [2e-4, 1.8e-4, 1.6e-4, 1.4e-4, 1.2e-4],
    'weight_decay': 1e-6,
    'batch_size': 96,
    'max_epochs': 250,
    'patience': 15,
    'gradient_clip': 2.0,
    'regime_detector_epochs': 100,
    'position_sizer_epochs': 80,
    'fusion_epochs': 60,
    'signal_generator_epochs': 60,
}


# ---------------------------------------------------------------------------
# PerformanceEstimator (reused from ARTEMIS)
# ---------------------------------------------------------------------------

class PerformanceEstimator:
    def __init__(self, tradingData):
        self.data = tradingData

    def computePnL(self):
        self.PnL = self.data["Money"][-1] - self.data["Money"][0]
        return self.PnL

    def computeAnnualizedReturn(self):
        cumRet = self.data['Returns'].cumsum().iloc[-1]
        start = self.data.index[0].to_pydatetime()
        end = self.data.index[-1].to_pydatetime()
        days = (end - start).days
        if cumRet > -1 and days > 0:
            self.annualizedReturn = 100 * (((1 + cumRet) ** (365 / days)) - 1)
        else:
            self.annualizedReturn = -100
        return self.annualizedReturn

    def computeAnnualizedVolatility(self):
        self.annualizedVolatily = 100 * np.sqrt(252) * self.data['Returns'].std()
        return self.annualizedVolatily

    def computeSharpeRatio(self, riskFreeRate=0):
        mu = self.data['Returns'].mean()
        sigma = self.data['Returns'].std()
        self.sharpeRatio = np.sqrt(252) * (mu - riskFreeRate) / sigma if sigma else 0
        return self.sharpeRatio

    def computeSortinoRatio(self, riskFreeRate=0):
        mu = np.mean(self.data['Returns'])
        neg = [r for r in self.data['Returns'] if r < 0]
        sigma = np.std(neg) if neg else 0
        self.sortinoRatio = np.sqrt(252) * (mu - riskFreeRate) / sigma if sigma else 0
        return self.sortinoRatio

    def computeMaxDrawdown(self, plotting=False):
        capital = self.data['Money'].values
        through = np.argmax(np.maximum.accumulate(capital) - capital)
        if through != 0:
            peak = np.argmax(capital[:through])
            self.maxDD = 100 * (capital[peak] - capital[through]) / capital[peak]
            self.maxDDD = through - peak
        else:
            self.maxDD = 0
            self.maxDDD = 0
        return self.maxDD, self.maxDDD

    def computeProfitability(self):
        profits = [r for r in self.data['Returns'] if r > 0]
        losses = [r for r in self.data['Returns'] if r < 0]
        self.profitability = (len(profits) / len(self.data['Returns'])) * 100
        if profits and losses:
            self.averageProfitLossRatio = np.mean(profits) / (-np.mean(losses))
        else:
            self.averageProfitLossRatio = 0
        return self.profitability, self.averageProfitLossRatio

    def computeSkewness(self):
        r = self.data['Returns']
        mu, sigma = np.mean(r), np.std(r)
        self.skewness = np.mean(((r - mu) / sigma) ** 3) if sigma else 0
        return self.skewness

    def computePerformance(self):
        self.computePnL()
        self.computeAnnualizedReturn()
        self.computeAnnualizedVolatility()
        self.computeProfitability()
        self.computeSharpeRatio()
        self.computeSortinoRatio()
        self.computeMaxDrawdown()
        self.computeSkewness()
        return {
            'PnL': self.PnL,
            'annualizedReturn': self.annualizedReturn,
            'annualizedVolatility': self.annualizedVolatily,
            'profitability': self.profitability,
            'averageProfitLossRatio': self.averageProfitLossRatio,
            'sharpeRatio': self.sharpeRatio,
            'sortinoRatio': self.sortinoRatio,
            'maxDD': self.maxDD,
            'maxDDD': self.maxDDD,
            'skewness': self.skewness,
        }

    def displayPerformance(self, name):
        perf = self.computePerformance()
        data = [
            ['P&L', f"{perf['PnL']:.2f}"],
            ['Annualized Return (%)', f"{perf['annualizedReturn']:.2f}"],
            ['Annualized Volatility (%)', f"{perf['annualizedVolatility']:.2f}"],
            ['Profitability (%)', f"{perf['profitability']:.2f}"],
            ['Avg Profit/Loss Ratio', f"{perf['averageProfitLossRatio']:.2f}"],
            ['Sharpe Ratio', f"{perf['sharpeRatio']:.3f}"],
            ['Sortino Ratio', f"{perf['sortinoRatio']:.3f}"],
            ['Maximum Drawdown (%)', f"{perf['maxDD']:.2f}"],
            ['Maximum Drawdown Duration', f"{perf['maxDDD']:.0f}"],
            ['Skewness', f"{perf['skewness']:.3f}"],
        ]
        print(f"\n{name} Performance:")
        print(tabulate(data, headers=['Metric', 'Value'], tablefmt='grid'))
