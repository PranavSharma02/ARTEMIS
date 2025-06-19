"""
Return Optimization Module for PyTorch Trading System
Focused improvements to boost annual returns while maintaining good Sharpe ratios
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class UltraReturnBoostLoss(nn.Module):
    """Ultra-aggressive loss function for maximum returns"""
    def __init__(self, alpha=0.03, return_weight=0.95, momentum_weight=0.05):
        super().__init__()
        self.alpha = alpha  # Reduced CVaR threshold for higher risk tolerance
        self.return_weight = return_weight
        self.momentum_weight = momentum_weight
        
    def forward(self, predictions, returns):
        signals = torch.tanh(predictions)
        portfolio_returns = returns * signals.squeeze()
        
        # Core return metrics
        mean_return = torch.mean(portfolio_returns)
        std_return = torch.std(portfolio_returns) + 1e-8
        
        # 1. Traditional Sharpe ratio (further reduced weight)
        sharpe_ratio = mean_return / std_return
        
        # 2. Return magnitude component (INCREASED)
        return_magnitude = torch.mean(torch.abs(portfolio_returns))
        
        # 3. Positive return frequency (INCREASED)
        positive_returns = (portfolio_returns > 0).float()
        positive_frequency = torch.mean(positive_returns)
        
        # 4. Large return bonus (NEW - rewards big wins)
        large_returns = torch.mean(torch.relu(portfolio_returns - 0.01))  # Bonus for returns > 1%
        
        # 5. Momentum capture (enhanced)
        if len(portfolio_returns) > 1:
            momentum_consistency = torch.mean(
                torch.sign(portfolio_returns[1:]) * torch.sign(portfolio_returns[:-1])
            )
            # Trend acceleration bonus
            trend_acceleration = torch.mean(torch.relu(
                portfolio_returns[1:] - portfolio_returns[:-1]
            ))
        else:
            momentum_consistency = torch.tensor(0.0, device=signals.device)
            trend_acceleration = torch.tensor(0.0, device=signals.device)
            
        # 6. CVaR component (very relaxed for ultra-aggressive mode)
        sorted_returns, _ = torch.sort(portfolio_returns)
        n_samples = sorted_returns.size(0)
        cvar_samples = max(1, int(n_samples * self.alpha))
        cvar = -torch.mean(sorted_returns[:cvar_samples])
        
        # 7. Signal turnover penalty (minimal)
        if len(signals) > 1:
            signal_changes = torch.mean(torch.abs(signals[1:] - signals[:-1]))
        else:
            signal_changes = torch.tensor(0.0, device=signals.device)
        
        # Ultra-aggressive return components with NEW elements
        return_components = (
            0.25 * sharpe_ratio +              # Further reduced Sharpe weight
            0.35 * return_magnitude +          # INCREASED absolute return focus
            0.20 * positive_frequency +        # Win rate
            0.15 * large_returns +             # NEW: Big win bonus
            0.10 * momentum_consistency +      # Trend consistency
            0.08 * trend_acceleration          # NEW: Trend acceleration bonus
        )
        
        # Minimal risk penalties for ultra-aggressive mode
        risk_penalties = (
            0.02 * cvar +                      # Very minimal downside protection
            0.005 * signal_changes             # Almost no trading frequency penalty
        )
        
        total_loss = -(self.return_weight * return_components - risk_penalties)
        
        return total_loss

# Keep original for backward compatibility
class ReturnBoostLoss(nn.Module):
    """Enhanced loss function optimized for higher returns"""
    def __init__(self, alpha=0.05, return_weight=0.9, momentum_weight=0.1):
        super().__init__()
        self.alpha = alpha
        self.return_weight = return_weight
        self.momentum_weight = momentum_weight
        
    def forward(self, predictions, returns):
        signals = torch.tanh(predictions)
        portfolio_returns = returns * signals.squeeze()
        
        # Core return metrics
        mean_return = torch.mean(portfolio_returns)
        std_return = torch.std(portfolio_returns) + 1e-8
        
        # 1. Traditional Sharpe ratio (weighted down for more aggressive returns)
        sharpe_ratio = mean_return / std_return
        
        # 2. Return magnitude component (NEW - focuses on absolute return size)
        return_magnitude = torch.mean(torch.abs(portfolio_returns))
        
        # 3. Positive return frequency (NEW - rewards frequent gains)
        positive_returns = (portfolio_returns > 0).float()
        positive_frequency = torch.mean(positive_returns)
        
        # 4. Momentum capture (NEW - rewards trend following)
        if len(portfolio_returns) > 1:
            momentum_consistency = torch.mean(
                torch.sign(portfolio_returns[1:]) * torch.sign(portfolio_returns[:-1])
            )
        else:
            momentum_consistency = torch.tensor(0.0, device=signals.device)
            
        # 5. CVaR component (relaxed threshold for higher risk tolerance)
        sorted_returns, _ = torch.sort(portfolio_returns)
        n_samples = sorted_returns.size(0)
        cvar_samples = max(1, int(n_samples * self.alpha))
        cvar = -torch.mean(sorted_returns[:cvar_samples])
        
        # 6. Signal turnover penalty (minimal for aggressive trading)
        if len(signals) > 1:
            signal_changes = torch.mean(torch.abs(signals[1:] - signals[:-1]))
        else:
            signal_changes = torch.tensor(0.0, device=signals.device)
        
        # Combined loss with return focus
        return_components = (
            0.35 * sharpe_ratio +           # Reduced weight on risk-adjusted returns
            0.30 * return_magnitude +       # High weight on absolute returns
            0.20 * positive_frequency +     # Reward win rate
            0.15 * momentum_consistency     # Reward trend consistency
        )
        
        risk_penalties = (
            0.05 * cvar +                   # Minimal downside protection
            0.01 * signal_changes           # Allow more frequent trading
        )
        
        total_loss = -(self.return_weight * return_components - risk_penalties)
        
        return total_loss

class ReturnOptimizedModel(nn.Module):
    """Enhanced model architecture optimized for returns"""
    def __init__(self, input_size, hidden_size=160, dropout=0.15):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Enhanced GRU with larger capacity
        self.gru1 = nn.GRU(input_size, hidden_size, 2, dropout=dropout, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size//2, 1, dropout=dropout, batch_first=True)
        
        # Multi-head attention for trend capture
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, dropout=dropout, batch_first=True)
        
        # Calculate correct feature size: final_hidden + global_features
        combined_feature_size = hidden_size//2 + hidden_size//2  # Both are hidden_size//2
        
        # Feature extraction layers
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
        
        # Multiple prediction heads
        self.return_head = nn.Linear(64, 1)      # Primary return prediction
        self.momentum_head = nn.Linear(64, 1)    # Momentum signal
        self.volatility_head = nn.Linear(64, 1)  # Volatility prediction
        
    def forward(self, x):
        # Hierarchical GRU processing
        gru1_out, _ = self.gru1(x)
        
        # Attention mechanism
        attn_out, _ = self.attention(gru1_out, gru1_out, gru1_out)
        
        # Second GRU layer
        gru2_out, hidden = self.gru2(attn_out)
        
        # Global feature extraction - both are from gru2 output
        final_hidden = hidden[-1]  # Shape: (batch, hidden_size//2)
        global_features = gru2_out.mean(dim=1)  # Shape: (batch, hidden_size//2)
        
        # Combine features
        combined_features = torch.cat([final_hidden, global_features], dim=1)
        
        # Extract high-level features
        features = self.feature_extractor(combined_features)
        
        # Multiple predictions
        return_pred = torch.tanh(self.return_head(features))
        momentum_pred = torch.tanh(self.momentum_head(features))
        volatility_pred = torch.sigmoid(self.volatility_head(features))
        
        # Adaptive signal combination based on volatility
        volatility_weight = volatility_pred.squeeze()
        adaptive_signal = (
            (1 - volatility_weight).unsqueeze(1) * return_pred +
            volatility_weight.unsqueeze(1) * momentum_pred
        )
        
        return adaptive_signal

class ReturnBoostRegimeDetector(nn.Module):
    """Enhanced regime detector with 5 market states for better return capture"""
    def __init__(self, input_size, hidden_size=80):
        super().__init__()
        
        # Enhanced CNN feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_size, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # Bidirectional LSTM for temporal modeling
        self.lstm = nn.LSTM(32, hidden_size, batch_first=True, bidirectional=True)
        
        # Classification head for 5 regimes
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 5)  # Bull, Bear, Sideways, High-Vol, Momentum
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        
        # CNN feature extraction
        conv_features = self.conv_layers(x)
        conv_features = conv_features.transpose(1, 2)  # Back to (batch, seq_len, features)
        
        # LSTM temporal modeling
        _, (hidden, _) = self.lstm(conv_features)
        
        # Concatenate bidirectional hidden states
        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        # Regime classification
        regime_probs = F.softmax(self.classifier(final_hidden), dim=1)
        
        return regime_probs

def create_enhanced_model_configs():
    """Create optimized model configurations for higher returns"""
    return [
        # Aggressive high-capacity model
        {'hidden_size': 160, 'dropout': 0.15, 'use_attention': True, 'aggressive': True},
        
        # Momentum-focused model  
        {'hidden_size': 144, 'dropout': 0.18, 'use_attention': True, 'momentum_focus': True},
        
        # Balanced return-optimized model
        {'hidden_size': 128, 'dropout': 0.20, 'use_attention': True, 'return_focus': True},
        
        # Trend-following model
        {'hidden_size': 112, 'dropout': 0.16, 'use_attention': True, 'trend_focus': True},
        
        # High-frequency trading model
        {'hidden_size': 96, 'dropout': 0.22, 'use_attention': False, 'high_freq': True}
    ]

def get_enhanced_regime_weights():
    """Enhanced regime-based weighting for higher returns"""
    return {
        0: np.array([0.35, 0.25, 0.20, 0.15, 0.05]),  # Bull market - favor aggressive models
        1: np.array([0.10, 0.25, 0.30, 0.25, 0.10]),  # Bear market - balanced approach
        2: np.array([0.15, 0.20, 0.25, 0.25, 0.15]),  # Sideways - momentum focus
        3: np.array([0.05, 0.15, 0.25, 0.35, 0.20]),  # High volatility - trend following
        4: np.array([0.40, 0.20, 0.15, 0.15, 0.10])   # Strong momentum - aggressive
    }

class PositionSizer(nn.Module):
    """Dynamic position sizing for enhanced returns"""
    def __init__(self, input_size=20):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
    def forward(self, market_features):
        # Returns position size multiplier (0.5 to 1.5 range)
        base_size = self.network(market_features)
        return 0.5 + base_size  # Range: 0.5 to 1.5

# Enhanced training parameters for return optimization
ENHANCED_TRAINING_PARAMS = {
    'learning_rates': [0.0015, 0.0012, 0.0010, 0.0008, 0.0006],  # Higher learning rates
    'batch_size': 64,                                             # Larger batches
    'epochs': 200,                                                # More training
    'patience': 20,                                               # More patience
    'weight_decay': 5e-6,                                         # Less regularization
    'gradient_clip': 1.5,                                         # Allow larger gradients
}

def create_ultra_aggressive_model_configs():
    """Create ultra-aggressive model configurations for maximum returns"""
    return [
        # Ultra-aggressive high-capacity model with maximum parameters
        {'hidden_size': 192, 'dropout': 0.12, 'use_attention': True, 'ultra_aggressive': True, 'name': 'MAREA-Ultra-1'},
        
        # High-momentum focus model
        {'hidden_size': 176, 'dropout': 0.14, 'use_attention': True, 'momentum_focus': True, 'aggressive': True, 'name': 'MAREA-Momentum'},
        
        # Large capacity return-optimized model
        {'hidden_size': 160, 'dropout': 0.15, 'use_attention': True, 'return_focus': True, 'aggressive': True, 'name': 'MAREA-Return'},
        
        # Enhanced trend-following model
        {'hidden_size': 144, 'dropout': 0.16, 'use_attention': True, 'trend_focus': True, 'aggressive': True, 'name': 'MAREA-Trend'},
        
        # High-frequency ultra-aggressive model
        {'hidden_size': 128, 'dropout': 0.18, 'use_attention': True, 'high_freq': True, 'ultra_aggressive': True, 'name': 'MAREA-HF'}
    ]

def get_ultra_aggressive_regime_weights():
    """Ultra-aggressive regime-based weighting for maximum returns"""
    return {
        0: np.array([0.45, 0.25, 0.15, 0.10, 0.05]),  # Bull market - MAXIMUM aggressive weighting
        1: np.array([0.15, 0.30, 0.25, 0.20, 0.10]),  # Bear market - still aggressive  
        2: np.array([0.20, 0.25, 0.25, 0.20, 0.10]),  # Sideways - momentum focus
        3: np.array([0.10, 0.20, 0.25, 0.30, 0.15]),  # High volatility - trend following
        4: np.array([0.50, 0.20, 0.15, 0.10, 0.05])   # Strong momentum - ULTRA aggressive
    }

class UltraReturnOptimizedModel(nn.Module):
    """Ultra-aggressive model architecture for maximum returns"""
    def __init__(self, input_size, hidden_size=192, dropout=0.12):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Ultra-enhanced GRU with maximum capacity
        self.gru1 = nn.GRU(input_size, hidden_size, 3, dropout=dropout, batch_first=True)  # 3 layers
        self.gru2 = nn.GRU(hidden_size, hidden_size//2, 2, dropout=dropout, batch_first=True)  # 2 layers
        
        # Multi-head attention with more heads for better trend capture
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=16, dropout=dropout, batch_first=True)
        
        # Enhanced feature extraction with larger capacity
        combined_feature_size = hidden_size//2 + hidden_size//2
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(combined_feature_size, 512),  # Increased capacity
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.12),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.05)
        )
        
        # Multiple specialized prediction heads
        self.return_head = nn.Linear(64, 1)         # Primary return prediction
        self.momentum_head = nn.Linear(64, 1)       # Momentum signal
        self.volatility_head = nn.Linear(64, 1)     # Volatility prediction
        self.trend_head = nn.Linear(64, 1)          # NEW: Trend strength prediction
        
    def forward(self, x):
        # Ultra-hierarchical GRU processing
        gru1_out, _ = self.gru1(x)
        
        # Enhanced attention mechanism
        attn_out, attn_weights = self.attention(gru1_out, gru1_out, gru1_out)
        
        # Second GRU layer with attention output
        gru2_out, hidden = self.gru2(attn_out)
        
        # Enhanced global feature extraction
        final_hidden = hidden[-1]
        global_avg = gru2_out.mean(dim=1)
        global_max = gru2_out.max(dim=1)[0]  # Add max pooling
        
        # Combine features (adjust for new global_max)
        combined_features = torch.cat([final_hidden, global_avg], dim=1)
        
        # Extract ultra high-level features
        features = self.feature_extractor(combined_features)
        
        # Multiple specialized predictions
        return_pred = torch.tanh(self.return_head(features))
        momentum_pred = torch.tanh(self.momentum_head(features))
        volatility_pred = torch.sigmoid(self.volatility_head(features))
        trend_pred = torch.tanh(self.trend_head(features))
        
        # Ultra-aggressive signal combination with trend weighting
        volatility_weight = volatility_pred.squeeze()
        trend_weight = torch.abs(trend_pred.squeeze())
        
        # More aggressive weighting favoring momentum and trend
        ultra_aggressive_signal = (
            0.4 * return_pred +
            0.3 * momentum_pred +
            0.2 * trend_pred +
            0.1 * (volatility_weight.unsqueeze(1) * return_pred)
        )
        
        return ultra_aggressive_signal

# Ultra-aggressive training parameters for maximum returns
ULTRA_AGGRESSIVE_TRAINING_PARAMS = {
    'learning_rates': [0.002, 0.0018, 0.0015, 0.0012, 0.001],  # Progressive learning rates
    'batch_size': 96,                                           # Large batch size
    'epochs': 250,                                              # Extended training
    'patience': 25,                                             # Enhanced patience
    'weight_decay': 1e-6,                                       # Minimal regularization
    'gradient_clip': 2.0,                                       # Allow larger gradients
    'regime_detector_epochs': 80,                               # Regime detector training
    'position_sizer_epochs': 50,                                # Position sizer training
}

class SharpeOptimizedLoss(nn.Module):
    """
    Sharpe-Optimized Loss Function for Maximum Risk-Adjusted Returns
    
    Maintains high absolute returns while dramatically improving Sharpe ratio through:
    1. Enhanced volatility control and smoothing
    2. Drawdown minimization techniques
    3. Return consistency optimization
    4. Volatility clustering reduction
    """
    def __init__(self, alpha=0.03, return_weight=0.6, sharpe_weight=0.4):
        super().__init__()
        self.alpha = alpha
        self.return_weight = return_weight
        self.sharpe_weight = sharpe_weight
        
    def forward(self, predictions, returns):
        signals = torch.tanh(predictions)
        portfolio_returns = returns * signals.squeeze()
        
        # Core metrics
        mean_return = torch.mean(portfolio_returns)
        std_return = torch.std(portfolio_returns) + 1e-8
        
        # 1. ENHANCED Sharpe ratio with stability bonus
        base_sharpe = mean_return / std_return
        
        # Volatility stability bonus (NEW - rewards consistent volatility)
        rolling_vol = []
        window_size = min(20, len(portfolio_returns) // 4)
        if len(portfolio_returns) >= window_size:
            for i in range(window_size, len(portfolio_returns)):
                vol_window = torch.std(portfolio_returns[i-window_size:i])
                rolling_vol.append(vol_window)
            
            if len(rolling_vol) > 1:
                vol_stability = 1.0 / (torch.std(torch.stack(rolling_vol)) + 1e-6)
            else:
                vol_stability = torch.tensor(1.0, device=signals.device)
        else:
            vol_stability = torch.tensor(1.0, device=signals.device)
        
        enhanced_sharpe = base_sharpe * (1 + 0.1 * vol_stability)
        
        # 2. Return magnitude (maintained for high returns)
        return_magnitude = torch.mean(torch.abs(portfolio_returns))
        
        # 3. ENHANCED positive return frequency with consistency
        positive_returns = (portfolio_returns > 0).float()
        positive_frequency = torch.mean(positive_returns)
        
        # Return consistency bonus (NEW - rewards smooth positive returns)
        if len(portfolio_returns) > 1:
            return_smoothness = 1.0 / (torch.mean(torch.abs(portfolio_returns[1:] - portfolio_returns[:-1])) + 1e-6)
        else:
            return_smoothness = torch.tensor(1.0, device=signals.device)
        
        # 4. ENHANCED momentum with drawdown control
        if len(portfolio_returns) > 1:
            momentum_consistency = torch.mean(
                torch.sign(portfolio_returns[1:]) * torch.sign(portfolio_returns[:-1])
            )
            
            # Drawdown penalty (NEW - heavily penalize large drawdowns)
            cumulative_returns = torch.cumsum(portfolio_returns, dim=0)
            running_max = torch.cummax(cumulative_returns, dim=0)[0]
            drawdowns = running_max - cumulative_returns
            max_drawdown = torch.max(drawdowns)
            drawdown_penalty = torch.relu(max_drawdown - 0.05)  # Penalize >5% drawdowns
        else:
            momentum_consistency = torch.tensor(0.0, device=signals.device)
            drawdown_penalty = torch.tensor(0.0, device=signals.device)
            
        # 5. REDUCED CVaR for controlled downside
        sorted_returns, _ = torch.sort(portfolio_returns)
        n_samples = sorted_returns.size(0)
        cvar_samples = max(1, int(n_samples * self.alpha))
        cvar = -torch.mean(sorted_returns[:cvar_samples])
        
        # 6. Signal smoothness (NEW - reduce trading noise)
        if len(signals) > 1:
            signal_smoothness = 1.0 / (torch.mean(torch.abs(signals[1:] - signals[:-1])) + 1e-6)
            signal_changes = torch.mean(torch.abs(signals[1:] - signals[:-1]))
        else:
            signal_smoothness = torch.tensor(1.0, device=signals.device)
            signal_changes = torch.tensor(0.0, device=signals.device)
        
        # SHARPE-OPTIMIZED COMPONENTS with return preservation
        sharpe_components = (
            0.40 * enhanced_sharpe +            # PRIMARY: Enhanced Sharpe ratio
            0.25 * return_magnitude +           # Maintain high returns
            0.15 * positive_frequency +         # Win rate consistency
            0.10 * momentum_consistency +       # Trend following
            0.05 * return_smoothness +          # NEW: Smooth return profile
            0.05 * signal_smoothness            # NEW: Smooth trading signals
        )
        
        # RISK CONTROL PENALTIES (increased focus on risk management)
        risk_penalties = (
            0.05 * cvar +                       # Downside protection
            0.02 * drawdown_penalty +           # NEW: Drawdown control
            0.01 * signal_changes               # Reduced trading frequency penalty
        )
        
        # Balanced optimization maintaining returns while improving Sharpe
        total_loss = -(self.return_weight * return_magnitude + 
                      self.sharpe_weight * sharpe_components - 
                      risk_penalties)
        
        return total_loss 

class SharpeOptimizedModel(nn.Module):
    """
    Sharpe-Optimized Model Architecture for Maximum Risk-Adjusted Returns
    
    Enhanced model that maintains high returns while dramatically improving Sharpe ratio through:
    1. Volatility-aware signal generation
    2. Risk-controlled prediction heads
    3. Adaptive signal smoothing
    4. Enhanced attention mechanisms for trend stability
    """
    def __init__(self, input_size, hidden_size=192, dropout=0.12):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.model_name = "MAREA-Sharpe-Optimized"
        
        # Enhanced GRU with risk-aware processing
        self.gru1 = nn.GRU(input_size, hidden_size, 3, dropout=dropout, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size//2, 2, dropout=dropout, batch_first=True)
        
        # Multi-head attention with stability focus
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=12, dropout=dropout, batch_first=True)
        
        # Volatility prediction network (NEW - for risk control)
        self.volatility_predictor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Volatility prediction [0,1]
        )
        
        # Risk-aware feature extraction
        combined_feature_size = hidden_size//2 + hidden_size//2 + 1  # +1 for volatility
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(combined_feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.12),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.05)
        )
        
        # Specialized prediction heads for risk-adjusted returns
        self.return_head = nn.Linear(64, 1)           # Primary return prediction
        self.momentum_head = nn.Linear(64, 1)         # Momentum signal
        self.risk_head = nn.Linear(64, 1)             # Risk control signal
        self.stability_head = nn.Linear(64, 1)        # NEW: Signal stability
        
        # Signal smoothing layer (NEW - reduces noise)
        self.signal_smoother = nn.Sequential(
            nn.Linear(4, 8),  # 4 prediction heads
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Enhanced GRU processing with attention
        gru1_out, gru1_hidden = self.gru1(x)
        
        # Stability-focused attention mechanism
        attn_out, attn_weights = self.attention(gru1_out, gru1_out, gru1_out)
        
        # Secondary GRU processing
        gru2_out, hidden = self.gru2(attn_out)
        
        # Volatility prediction from GRU1 output
        vol_pred = self.volatility_predictor(gru1_hidden[-1])
        
        # Feature combination with volatility awareness
        final_hidden = hidden[-1]
        global_avg = gru2_out.mean(dim=1)
        volatility_feature = vol_pred.squeeze().unsqueeze(1)
        
        combined_features = torch.cat([final_hidden, global_avg, volatility_feature], dim=1)
        
        # Risk-aware feature extraction
        features = self.feature_extractor(combined_features)
        
        # Multiple specialized predictions
        return_pred = torch.tanh(self.return_head(features))
        momentum_pred = torch.tanh(self.momentum_head(features))
        risk_pred = torch.sigmoid(self.risk_head(features))
        stability_pred = torch.sigmoid(self.stability_head(features))
        
        # Risk-adjusted signal combination
        volatility_weight = vol_pred.squeeze()
        risk_weight = risk_pred.squeeze()
        stability_weight = stability_pred.squeeze()
        
        # Sharpe-optimized signal fusion
        # Reduce signal magnitude in high volatility periods
        vol_adjustment = 1.0 - 0.3 * volatility_weight  # Reduce exposure in high vol
        risk_adjustment = 1.0 - 0.2 * (1.0 - risk_weight)  # Reduce exposure in high risk
        
        # Combine all predictions with risk control
        raw_signals = torch.cat([
            return_pred,
            momentum_pred, 
            risk_pred.unsqueeze(1),
            stability_pred.unsqueeze(1)
        ], dim=1)
        
        # Apply signal smoothing
        smoothed_signal = self.signal_smoother(raw_signals)
        
        # Final risk-adjusted signal
        sharpe_optimized_signal = smoothed_signal * vol_adjustment.unsqueeze(1) * risk_adjustment.unsqueeze(1)
        
        return sharpe_optimized_signal 

def create_sharpe_optimized_configs():
    """Create Sharpe-optimized model configurations for maximum risk-adjusted returns"""
    return [
        # Sharpe-Optimized Primary Model
        {'hidden_size': 192, 'dropout': 0.12, 'use_attention': True, 'sharpe_optimized': True, 'name': 'MAREA-Sharpe-1'},
        
        # Risk-Controlled High-Return Model
        {'hidden_size': 176, 'dropout': 0.14, 'use_attention': True, 'risk_controlled': True, 'aggressive': True, 'name': 'MAREA-Risk-Control'},
        
        # Volatility-Aware Return Model
        {'hidden_size': 160, 'dropout': 0.15, 'use_attention': True, 'volatility_aware': True, 'name': 'MAREA-Vol-Aware'},
        
        # Stability-Focused Trend Model
        {'hidden_size': 144, 'dropout': 0.16, 'use_attention': True, 'stability_focus': True, 'name': 'MAREA-Stable-Trend'},
        
        # Smooth Signal Generation Model
        {'hidden_size': 128, 'dropout': 0.18, 'use_attention': True, 'signal_smoothing': True, 'name': 'MAREA-Smooth'}
    ]

def get_sharpe_optimized_regime_weights():
    """Sharpe-optimized regime weights favoring stability and risk control"""
    return {
        0: np.array([0.40, 0.25, 0.20, 0.10, 0.05]),  # Bull: Favor Sharpe-optimized models
        1: np.array([0.20, 0.35, 0.25, 0.15, 0.05]),  # Bear: Risk-controlled approach
        2: np.array([0.25, 0.25, 0.30, 0.15, 0.05]),  # Sideways: Volatility-aware
        3: np.array([0.15, 0.20, 0.35, 0.25, 0.05]),  # High-Vol: Stability focus
        4: np.array([0.35, 0.20, 0.20, 0.20, 0.05])   # Momentum: Balanced Sharpe approach
    }

# Sharpe-optimized training parameters
SHARPE_OPTIMIZED_TRAINING_PARAMS = {
    'learning_rates': [0.0015, 0.0012, 0.0010, 0.0008, 0.0006],  # More conservative learning
    'batch_size': 64,                                             # Stable batch size
    'epochs': 300,                                                # Extended training for stability
    'patience': 30,                                               # More patience for convergence
    'weight_decay': 2e-6,                                         # Balanced regularization
    'gradient_clip': 1.5,                                         # Controlled gradients
    'regime_detector_epochs': 100,                                # Enhanced regime detection
    'position_sizer_epochs': 60,                                  # Better position sizing
    'early_stopping_delta': 1e-6,                                # Fine-tuned early stopping
    'sharpe_validation_window': 50,                               # Validation window for Sharpe calculation
} 