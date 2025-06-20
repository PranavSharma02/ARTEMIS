#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tabulate import tabulate
from matplotlib import pyplot as plt

class UltraReturnBoostLoss(nn.Module):
    def __init__(self, alpha=0.03, return_weight=0.95, momentum_weight=0.05):
        super().__init__()
        self.alpha = alpha
        self.return_weight = return_weight
        self.momentum_weight = momentum_weight
        
    def forward(self, predictions, returns):
        signals = torch.tanh(predictions)
        portfolio_returns = returns * signals.squeeze()
        
        mean_return = torch.mean(portfolio_returns)
        std_return = torch.std(portfolio_returns) + 1e-8
        
        sharpe_ratio = mean_return / std_return
        return_magnitude = torch.mean(torch.abs(portfolio_returns))
        positive_returns = (portfolio_returns > 0).float()
        positive_frequency = torch.mean(positive_returns)
        large_returns = torch.mean(torch.relu(portfolio_returns - 0.01))
        
        if len(portfolio_returns) > 1:
            momentum_consistency = torch.mean(
                torch.sign(portfolio_returns[1:]) * torch.sign(portfolio_returns[:-1])
            )
            trend_acceleration = torch.mean(torch.relu(
                portfolio_returns[1:] - portfolio_returns[:-1]
            ))
        else:
            momentum_consistency = torch.tensor(0.0, device=signals.device)
            trend_acceleration = torch.tensor(0.0, device=signals.device)
            
        sorted_returns, _ = torch.sort(portfolio_returns)
        n_samples = sorted_returns.size(0)
        cvar_samples = max(1, int(n_samples * self.alpha))
        cvar = -torch.mean(sorted_returns[:cvar_samples])
        
        if len(signals) > 1:
            signal_changes = torch.mean(torch.abs(signals[1:] - signals[:-1]))
        else:
            signal_changes = torch.tensor(0.0, device=signals.device)
        
        return_components = (
            0.25 * sharpe_ratio +
            0.35 * return_magnitude +
            0.20 * positive_frequency +
            0.15 * large_returns +
            0.10 * momentum_consistency +
            0.08 * trend_acceleration
        )
        
        risk_penalties = (
            0.02 * cvar +
            0.005 * signal_changes
        )
        
        total_loss = -(self.return_weight * return_components - risk_penalties)
        
        return total_loss

class ReturnBoostLoss(nn.Module):
    def __init__(self, alpha=0.05, return_weight=0.9, momentum_weight=0.1):
        super().__init__()
        self.alpha = alpha
        self.return_weight = return_weight
        self.momentum_weight = momentum_weight
        
    def forward(self, predictions, returns):
        signals = torch.tanh(predictions)
        portfolio_returns = returns * signals.squeeze()
        
        mean_return = torch.mean(portfolio_returns)
        std_return = torch.std(portfolio_returns) + 1e-8
        
        sharpe_ratio = mean_return / std_return
        return_magnitude = torch.mean(torch.abs(portfolio_returns))
        positive_returns = (portfolio_returns > 0).float()
        positive_frequency = torch.mean(positive_returns)
        
        if len(portfolio_returns) > 1:
            momentum_consistency = torch.mean(
                torch.sign(portfolio_returns[1:]) * torch.sign(portfolio_returns[:-1])
            )
        else:
            momentum_consistency = torch.tensor(0.0, device=signals.device)
            
        sorted_returns, _ = torch.sort(portfolio_returns)
        n_samples = sorted_returns.size(0)
        cvar_samples = max(1, int(n_samples * self.alpha))
        cvar = -torch.mean(sorted_returns[:cvar_samples])
        
        if len(signals) > 1:
            signal_changes = torch.mean(torch.abs(signals[1:] - signals[:-1]))
        else:
            signal_changes = torch.tensor(0.0, device=signals.device)
        
        return_components = (
            0.35 * sharpe_ratio +
            0.30 * return_magnitude +
            0.20 * positive_frequency +
            0.15 * momentum_consistency
        )
        
        risk_penalties = (
            0.05 * cvar +
            0.01 * signal_changes
        )
        
        total_loss = -(self.return_weight * return_components - risk_penalties)
        
        return total_loss

class ReturnOptimizedModel(nn.Module):
    def __init__(self, input_size, hidden_size=160, dropout=0.15):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        self.gru1 = nn.GRU(input_size, hidden_size, 2, dropout=dropout, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size//2, 1, dropout=dropout, batch_first=True)
        
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, dropout=dropout, batch_first=True)
        
        combined_feature_size = hidden_size//2 + hidden_size//2
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(combined_feature_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.return_head = nn.Linear(64, 1)
        self.momentum_head = nn.Linear(64, 1)
        self.volatility_head = nn.Linear(64, 1)
        
    def forward(self, x):
        gru1_out, _ = self.gru1(x)
        
        attn_out, _ = self.attention(gru1_out, gru1_out, gru1_out)
        
        gru2_out, hidden = self.gru2(attn_out)
        
        final_hidden = hidden[-1]
        global_features = gru2_out.mean(dim=1)
        
        combined_features = torch.cat([final_hidden, global_features], dim=1)
        features = self.feature_extractor(combined_features)
        
        return_pred = self.return_head(features)
        momentum_pred = self.momentum_head(features)
        volatility_pred = self.volatility_head(features)
        
        combined_prediction = return_pred + 0.3 * momentum_pred - 0.1 * volatility_pred
        
        return combined_prediction

class ReturnBoostRegimeDetector(nn.Module):
    def __init__(self, input_size, hidden_size=80):
        super().__init__()
        
        self.gru = nn.GRU(input_size, hidden_size, 2, dropout=0.1, batch_first=True)
        
        self.regime_classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )
        
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        gru_out, hidden = self.gru(x)
        
        final_hidden = hidden[-1]
        
        regime_logits = self.regime_classifier(final_hidden)
        regime_probs = F.softmax(regime_logits, dim=1)
        
        return regime_probs

def create_enhanced_model_configs():
    return [
        {'name': 'artemis_enhanced_1', 'hidden_size': 160, 'dropout': 0.15, 'return_focus': True},
        {'name': 'artemis_enhanced_2', 'hidden_size': 128, 'dropout': 0.18, 'return_focus': True},
        {'name': 'artemis_enhanced_3', 'hidden_size': 144, 'dropout': 0.16, 'return_focus': True},
        {'name': 'artemis_enhanced_4', 'hidden_size': 112, 'dropout': 0.20, 'return_focus': True},
        {'name': 'artemis_enhanced_5', 'hidden_size': 96, 'dropout': 0.22, 'return_focus': True}
    ]

def get_enhanced_regime_weights():
    return {
        0: [0.3, 0.25, 0.2, 0.15, 0.1],
        1: [0.25, 0.3, 0.25, 0.12, 0.08],
        2: [0.2, 0.2, 0.35, 0.15, 0.1],
        3: [0.15, 0.2, 0.25, 0.3, 0.1],
        4: [0.1, 0.15, 0.2, 0.25, 0.3]
    }

class PositionSizer(nn.Module):
    def __init__(self, input_size=20):
        super().__init__()
        
        self.position_net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, market_features):
        position_multiplier = self.position_net(market_features)
        return 0.5 + position_multiplier

def create_ultra_aggressive_model_configs():
    return [
        {'hidden_size': 192, 'dropout': 0.12, 'use_attention': True, 'ultra_aggressive': True, 'name': 'ARTEMIS-Ultra-1'},
        {'hidden_size': 176, 'dropout': 0.14, 'use_attention': True, 'momentum_focus': True, 'aggressive': True, 'name': 'ARTEMIS-Momentum'},
        {'hidden_size': 160, 'dropout': 0.15, 'use_attention': True, 'return_focus': True, 'aggressive': True, 'name': 'ARTEMIS-Return'},
        {'hidden_size': 144, 'dropout': 0.16, 'use_attention': True, 'trend_focus': True, 'aggressive': True, 'name': 'ARTEMIS-Trend'},
        {'hidden_size': 128, 'dropout': 0.18, 'use_attention': True, 'high_freq': True, 'ultra_aggressive': True, 'name': 'ARTEMIS-HF'}
    ]

def get_ultra_aggressive_regime_weights():
    return {
        0: [0.35, 0.25, 0.20, 0.12, 0.08],
        1: [0.30, 0.30, 0.25, 0.10, 0.05],
        2: [0.25, 0.25, 0.30, 0.15, 0.05],
        3: [0.20, 0.25, 0.25, 0.25, 0.05],
        4: [0.15, 0.20, 0.25, 0.20, 0.20]
    }

class UltraReturnOptimizedModel(nn.Module):
    def __init__(self, input_size, hidden_size=192, dropout=0.12):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        self.gru1 = nn.GRU(input_size, hidden_size, 3, dropout=dropout, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size//2, 2, dropout=dropout, batch_first=True)
        self.gru3 = nn.GRU(hidden_size//2, hidden_size//4, 1, dropout=dropout, batch_first=True)
        
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=16, dropout=dropout, batch_first=True)
        
        combined_feature_size = hidden_size//4 + hidden_size//4 + hidden_size//4
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(combined_feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.05)
        )
        
        self.return_head = nn.Linear(64, 1)
        self.momentum_head = nn.Linear(64, 1)
        self.volatility_head = nn.Linear(64, 1)
        self.regime_head = nn.Linear(64, 1)
        
    def forward(self, x):
        gru1_out, _ = self.gru1(x)
        
        attn_out, _ = self.attention(gru1_out, gru1_out, gru1_out)
        
        gru2_out, hidden2 = self.gru2(attn_out)
        gru3_out, hidden3 = self.gru3(gru2_out)
        
        final_hidden = hidden3[-1]
        global_features = gru3_out.mean(dim=1)
        max_features = gru3_out.max(dim=1)[0]
        
        combined_features = torch.cat([final_hidden, global_features, max_features], dim=1)
        features = self.feature_extractor(combined_features)
        
        return_pred = self.return_head(features)
        momentum_pred = self.momentum_head(features)
        volatility_pred = self.volatility_head(features)
        regime_pred = self.regime_head(features)
        
        ultra_aggressive_prediction = (
            1.0 * return_pred + 
            0.4 * momentum_pred - 
            0.1 * volatility_pred + 
            0.2 * regime_pred
        )
        
        return ultra_aggressive_prediction

class SharpeOptimizedLoss(nn.Module):
    def __init__(self, alpha=0.03, return_weight=0.6, sharpe_weight=0.4):
        super().__init__()
        self.alpha = alpha
        self.return_weight = return_weight
        self.sharpe_weight = sharpe_weight
        
    def forward(self, predictions, returns):
        signals = torch.tanh(predictions)
        portfolio_returns = returns * signals.squeeze()
        
        mean_return = torch.mean(portfolio_returns)
        std_return = torch.std(portfolio_returns) + 1e-8
        
        sharpe_ratio = mean_return / std_return
        
        return_magnitude = torch.mean(torch.abs(portfolio_returns))
        
        positive_returns = (portfolio_returns > 0).float()
        positive_frequency = torch.mean(positive_returns)
        
        large_returns = torch.mean(torch.relu(portfolio_returns - 0.005))
        
        if len(portfolio_returns) > 1:
            momentum_consistency = torch.mean(
                torch.sign(portfolio_returns[1:]) * torch.sign(portfolio_returns[:-1])
            )
            trend_acceleration = torch.mean(torch.relu(
                portfolio_returns[1:] - portfolio_returns[:-1]
            ))
            volatility_consistency = 1.0 / (1.0 + torch.std(
                torch.abs(portfolio_returns[1:]) - torch.abs(portfolio_returns[:-1])
            ))
        else:
            momentum_consistency = torch.tensor(0.0, device=signals.device)
            trend_acceleration = torch.tensor(0.0, device=signals.device)
            volatility_consistency = torch.tensor(1.0, device=signals.device)
            
        sorted_returns, _ = torch.sort(portfolio_returns)
        n_samples = sorted_returns.size(0)
        cvar_samples = max(1, int(n_samples * self.alpha))
        cvar = -torch.mean(sorted_returns[:cvar_samples])
        
        if len(signals) > 1:
            signal_changes = torch.mean(torch.abs(signals[1:] - signals[:-1]))
        else:
            signal_changes = torch.tensor(0.0, device=signals.device)
        
        risk_adjusted_components = (
            0.40 * sharpe_ratio +
            0.20 * return_magnitude +
            0.15 * positive_frequency +
            0.10 * large_returns +
            0.08 * momentum_consistency +
            0.05 * trend_acceleration +
            0.02 * volatility_consistency
        )
        
        risk_penalties = (
            0.03 * cvar +
            0.01 * signal_changes
        )
        
        balanced_return_components = (
            0.35 * return_magnitude +
            0.25 * positive_frequency +
            0.20 * large_returns +
            0.12 * momentum_consistency +
            0.08 * trend_acceleration
        )
        
        total_objective = (
            self.sharpe_weight * risk_adjusted_components +
            self.return_weight * balanced_return_components
        )
        
        total_loss = -(total_objective - risk_penalties)
        
        return total_loss

class SharpeOptimizedModel(nn.Module):
    def __init__(self, input_size, hidden_size=192, dropout=0.12):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        self.gru1 = nn.GRU(input_size, hidden_size, 2, dropout=dropout, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size//2, 1, dropout=dropout, batch_first=True)
        
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=12, dropout=dropout, batch_first=True)
        self.self_attention = nn.MultiheadAttention(hidden_size//2, num_heads=8, dropout=dropout, batch_first=True)
        
        combined_feature_size = hidden_size//2 + hidden_size//2 + hidden_size//2
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(combined_feature_size, 384),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.BatchNorm1d(384),
            
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Dropout(0.12),
            nn.BatchNorm1d(192),
            
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.BatchNorm1d(96),
            
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Dropout(0.05)
        )
        
        self.return_head = nn.Linear(48, 1)
        self.risk_head = nn.Linear(48, 1)
        self.momentum_head = nn.Linear(48, 1)
        
    def forward(self, x):
        gru1_out, _ = self.gru1(x)
        
        attn_out, _ = self.attention(gru1_out, gru1_out, gru1_out)
        
        gru2_out, hidden = self.gru2(attn_out)
        
        self_attn_out, _ = self.self_attention(gru2_out, gru2_out, gru2_out)
        
        final_hidden = hidden[-1]
        global_features = self_attn_out.mean(dim=1)
        max_features = self_attn_out.max(dim=1)[0]
        
        combined_features = torch.cat([final_hidden, global_features, max_features], dim=1)
        features = self.feature_extractor(combined_features)
        
        return_pred = self.return_head(features)
        risk_pred = self.risk_head(features)
        momentum_pred = self.momentum_head(features)
        
        sharpe_optimized_prediction = (
            0.6 * return_pred - 
            0.3 * risk_pred + 
            0.1 * momentum_pred
        )
        
        return sharpe_optimized_prediction

def create_sharpe_optimized_configs():
    return [
        {'hidden_size': 192, 'dropout': 0.12, 'use_attention': True, 'sharpe_focus': True, 'name': 'ARTEMIS-Sharpe-1'},
        {'hidden_size': 176, 'dropout': 0.14, 'use_attention': True, 'balanced_focus': True, 'name': 'ARTEMIS-Sharpe-2'},
        {'hidden_size': 160, 'dropout': 0.15, 'use_attention': True, 'risk_adjusted': True, 'name': 'ARTEMIS-Sharpe-3'},
        {'hidden_size': 144, 'dropout': 0.16, 'use_attention': True, 'stable_focus': True, 'name': 'ARTEMIS-Sharpe-4'},
        {'hidden_size': 128, 'dropout': 0.18, 'use_attention': True, 'conservative': True, 'name': 'ARTEMIS-Sharpe-5'}
    ]

def get_sharpe_optimized_regime_weights():
    return {
        0: [0.30, 0.25, 0.20, 0.15, 0.10],
        1: [0.25, 0.30, 0.25, 0.15, 0.05],
        2: [0.20, 0.25, 0.30, 0.20, 0.05],
        3: [0.15, 0.20, 0.25, 0.30, 0.10],
        4: [0.10, 0.15, 0.20, 0.25, 0.30]
    }

ULTRA_AGGRESSIVE_TRAINING_PARAMS = {
    'learning_rate': 2e-4,
    'learning_rates': [2e-4, 1.8e-4, 1.6e-4, 1.4e-4, 1.2e-4],
    'weight_decay': 1e-6,
    'batch_size': 96,
    'max_epochs': 250,
    'patience': 15,
    'grad_clip': 2.0,
    'gradient_clip': 2.0,
    'scheduler_factor': 0.7,
    'scheduler_patience': 8,
    'regime_detector_epochs': 100,
    'position_sizer_epochs': 80
}

ENHANCED_TRAINING_PARAMS = {
    'learning_rate': 1e-4,
    'learning_rates': [1e-4, 9e-5, 8e-5, 7e-5, 6e-5],
    'weight_decay': 1e-5,
    'batch_size': 64,
    'max_epochs': 200,
    'patience': 12,
    'grad_clip': 1.0,
    'gradient_clip': 1.0,
    'scheduler_factor': 0.8,
    'scheduler_patience': 6,
    'regime_detector_epochs': 80,
    'position_sizer_epochs': 60
}

SHARPE_OPTIMIZED_TRAINING_PARAMS = {
    'learning_rate': 1.5e-4,
    'learning_rates': [1.5e-4, 1.3e-4, 1.1e-4, 9e-5, 7e-5],
    'weight_decay': 5e-6,
    'batch_size': 80,
    'max_epochs': 300,
    'patience': 20,
    'grad_clip': 1.5,
    'gradient_clip': 1.5,
    'scheduler_factor': 0.75,
    'scheduler_patience': 10,
    'regime_detector_epochs': 120,
    'position_sizer_epochs': 100
}

class PerformanceEstimator:
    def __init__(self, tradingData):
        self.data = tradingData

    def computePnL(self):
        self.PnL = self.data["Money"][-1] - self.data["Money"][0]
        return self.PnL
    
    def computeAnnualizedReturn(self):
        cumulativeReturn = self.data['Returns'].cumsum()
        cumulativeReturn = cumulativeReturn[-1]
        
        start = self.data.index[0].to_pydatetime()
        end = self.data.index[-1].to_pydatetime()     
        timeElapsed = end - start
        timeElapsed = timeElapsed.days

        if(cumulativeReturn > -1):
            self.annualizedReturn = 100 * (((1 + cumulativeReturn) ** (365/timeElapsed)) - 1)
        else:
            self.annualizedReturn = -100
        return self.annualizedReturn
    
    def computeAnnualizedVolatility(self):
        self.annualizedVolatily = 100 * np.sqrt(252) * self.data['Returns'].std()
        return self.annualizedVolatily
    
    def computeSharpeRatio(self, riskFreeRate=0):
        expectedReturn = self.data['Returns'].mean()
        volatility = self.data['Returns'].std()
        
        if expectedReturn != 0 and volatility != 0:
            self.sharpeRatio = np.sqrt(252) * (expectedReturn - riskFreeRate)/volatility
        else:
            self.sharpeRatio = 0
        return self.sharpeRatio
    
    def computeSortinoRatio(self, riskFreeRate=0):
        expectedReturn = np.mean(self.data['Returns'])
        
        negativeReturns = [returns for returns in self.data['Returns'] if returns < 0]
        volatility = np.std(negativeReturns)
        
        if expectedReturn != 0 and volatility != 0:
            self.sortinoRatio = np.sqrt(252) * (expectedReturn - riskFreeRate)/volatility
        else:
            self.sortinoRatio = 0
        return self.sortinoRatio
    
    def computeMaxDrawdown(self, plotting=False):
        capital = self.data['Money'].values
        through = np.argmax(np.maximum.accumulate(capital) - capital)
        if through != 0:
            peak = np.argmax(capital[:through])
            self.maxDD = 100 * (capital[peak] - capital[through])/capital[peak]
            self.maxDDD = through - peak
        else:
            self.maxDD = 0
            self.maxDDD = 0
            return self.maxDD, self.maxDDD

        if plotting:
            plt.figure(figsize=(10, 4))
            plt.fill_between(range(peak, through+1), capital[peak:through+1], capital[peak], 
                           color='red', alpha=0.3, label='Maximum Drawdown')
            plt.plot(capital, label='Capital Evolution')
            plt.axvline(x=peak, color='green', linestyle='--', label='Peak')
            plt.axvline(x=through, color='red', linestyle='--', label='Through')
            plt.title('Maximum Drawdown Visualization')
            plt.xlabel('Time')
            plt.ylabel('Capital')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        return self.maxDD, self.maxDDD
    
    def computeProfitability(self):
        profits = [returns for returns in self.data['Returns'] if returns > 0]
        losses = [returns for returns in self.data['Returns'] if returns < 0]
        
        self.profitability = (len(profits) / len(self.data['Returns'])) * 100
        
        if len(profits) > 0 and len(losses) > 0:
            self.averageProfitLossRatio = (np.mean(profits) / (-np.mean(losses)))
        else:
            self.averageProfitLossRatio = 0
            
        return self.profitability, self.averageProfitLossRatio
    
    def computeSkewness(self):
        returns = self.data['Returns']
        mean = np.mean(returns)
        std = np.std(returns)
        
        if std != 0:
            self.skewness = np.mean(((returns - mean) / std) ** 3)
        else:
            self.skewness = 0
            
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
        
        performance = {
            'PnL': self.PnL,
            'annualizedReturn': self.annualizedReturn,
            'annualizedVolatility': self.annualizedVolatily,
            'profitability': self.profitability,
            'averageProfitLossRatio': self.averageProfitLossRatio,
            'sharpeRatio': self.sharpeRatio,
            'sortinoRatio': self.sortinoRatio,
            'maxDD': self.maxDD,
            'maxDDD': self.maxDDD,
            'skewness': self.skewness
        }
        
        return performance
    
    def displayPerformance(self, name):
        performance = self.computePerformance()
        
        data = [
            ['P&L', f"{performance['PnL']:.2f}"],
            ['Annualized Return (%)', f"{performance['annualizedReturn']:.2f}"],
            ['Annualized Volatility (%)', f"{performance['annualizedVolatility']:.2f}"],
            ['Profitability (%)', f"{performance['profitability']:.2f}"],
            ['Avg Profit/Loss Ratio', f"{performance['averageProfitLossRatio']:.2f}"],
            ['Sharpe Ratio', f"{performance['sharpeRatio']:.3f}"],
            ['Sortino Ratio', f"{performance['sortinoRatio']:.3f}"],
            ['Maximum Drawdown (%)', f"{performance['maxDD']:.2f}"],
            ['Maximum Drawdown Duration', f"{performance['maxDDD']:.0f}"],
            ['Skewness', f"{performance['skewness']:.3f}"]
        ]
        
        print(f"\n{name} Performance:")
        print(tabulate(data, headers=['Metric', 'Value'], tablefmt='grid')) 