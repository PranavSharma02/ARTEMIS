#!/usr/bin/env python3
"""
MAREA-Ensemble Diverse Architectures
Implements different neural network architectures for each of the 5 models

This module provides diverse architectures:
1. MAREA-Ultra-1: GRU + Multi-Head Attention (Ultra-aggressive)
2. MAREA-Momentum: LSTM + Temporal Convolution (Momentum focus)
3. MAREA-Return: Transformer Encoder (Return optimization)
4. MAREA-Trend: CNN + Bidirectional LSTM (Trend following)
5. MAREA-HF: 1D CNN + GRU (High-frequency trading)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MAREAUltra1Model(nn.Module):
    """MAREA-Ultra-1: GRU + Multi-Head Attention (Ultra-aggressive)"""
    def __init__(self, input_size, hidden_size=192, dropout=0.12):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 3-layer hierarchical GRU
        self.gru1 = nn.GRU(input_size, hidden_size, 3, dropout=dropout, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size//2, 2, dropout=dropout, batch_first=True)
        
        # 16-head Multi-Head Attention
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=16, dropout=dropout, batch_first=True)
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size//2, 512),
            nn.ReLU(), nn.Dropout(0.15), nn.BatchNorm1d(512),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.12), nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.08), nn.BatchNorm1d(128),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.05)
        )
        
        # Multi-head predictions
        self.return_head = nn.Linear(64, 1)
        self.momentum_head = nn.Linear(64, 1)
        self.volatility_head = nn.Linear(64, 1)
        self.trend_head = nn.Linear(64, 1)
        
    def forward(self, x):
        # Ultra-hierarchical GRU processing
        gru1_out, _ = self.gru1(x)
        attn_out, _ = self.attention(gru1_out, gru1_out, gru1_out)
        gru2_out, hidden = self.gru2(attn_out)
        
        # Feature extraction - use only the final hidden state to avoid dimension mismatch
        final_hidden = hidden[-1]  # Shape: (batch, hidden_size//2)
        features = self.feature_extractor(final_hidden)
        
        # Multi-head predictions
        return_pred = torch.tanh(self.return_head(features))
        momentum_pred = torch.tanh(self.momentum_head(features))
        volatility_pred = torch.sigmoid(self.volatility_head(features))
        trend_pred = torch.tanh(self.trend_head(features))
        
        # Ultra-aggressive signal combination
        ultra_aggressive_signal = (
            0.4 * return_pred + 0.3 * momentum_pred + 
            0.2 * trend_pred + 0.1 * (volatility_pred * return_pred)
        )
        return ultra_aggressive_signal.squeeze(-1)

class MAREAMomentumModel(nn.Module):
    """MAREA-Momentum: LSTM + Temporal Convolution (Momentum focus)"""
    def __init__(self, input_size, hidden_size=176, dropout=0.14):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Temporal Convolutional Network for momentum detection
        self.tcn_layers = nn.ModuleList([
            nn.Conv1d(input_size, hidden_size//2, kernel_size=3, padding=1),
            nn.Conv1d(hidden_size//2, hidden_size//2, kernel_size=5, padding=2),
            nn.Conv1d(hidden_size//2, hidden_size//2, kernel_size=7, padding=3)
        ])
        
        # LSTM for sequence modeling - input size matches TCN output
        self.lstm = nn.LSTM(hidden_size//2, hidden_size, 2, dropout=dropout, batch_first=True)
        
        # Momentum-specific attention
        self.momentum_attention = nn.MultiheadAttention(hidden_size, num_heads=8, dropout=dropout, batch_first=True)
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.ReLU(), nn.Dropout(0.2), nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.15), nn.BatchNorm1d(128),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1)
        )
        
        # Momentum-focused predictions
        self.momentum_head = nn.Linear(64, 1)
        self.trend_strength_head = nn.Linear(64, 1)
        self.acceleration_head = nn.Linear(64, 1)
        
    def forward(self, x):
        # Temporal convolution for momentum detection
        x_conv = x.transpose(1, 2)  # (batch, features, seq_len)
        for conv in self.tcn_layers:
            x_conv = F.relu(conv(x_conv))
        x_conv = x_conv.transpose(1, 2)  # Back to (batch, seq_len, features)
        
        # LSTM processing
        lstm_out, (hidden, _) = self.lstm(x_conv)
        
        # Momentum attention
        attn_out, _ = self.momentum_attention(lstm_out, lstm_out, lstm_out)
        
        # Feature extraction - use only the final hidden state to avoid dimension mismatch
        final_hidden = hidden[-1]  # Shape: (batch, hidden_size)
        features = self.feature_extractor(final_hidden)
        
        # Momentum-focused predictions
        momentum_pred = torch.tanh(self.momentum_head(features))
        trend_strength = torch.sigmoid(self.trend_strength_head(features))
        acceleration = torch.tanh(self.acceleration_head(features))
        
        # Momentum-weighted signal
        momentum_signal = (
            0.5 * momentum_pred + 0.3 * trend_strength + 0.2 * acceleration
        )
        
        return momentum_signal

class MAREAReturnModel(nn.Module):
    """MAREA-Return: Transformer Encoder (Return optimization)"""
    def __init__(self, input_size, hidden_size=160, dropout=0.15):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Input projection
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # Transformer Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 60, hidden_size))
        
        # Return-focused feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.ReLU(), nn.Dropout(0.2), nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.15), nn.BatchNorm1d(128),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1)
        )
        
        # Return optimization heads
        self.return_prediction_head = nn.Linear(64, 1)
        self.volatility_head = nn.Linear(64, 1)
        self.risk_adjusted_head = nn.Linear(64, 1)
        
    def forward(self, x):
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        # Transformer encoding
        transformer_out = self.transformer(x)
        
        # Global pooling - use only average to avoid dimension mismatch
        global_avg = transformer_out.mean(dim=1)  # Shape: (batch, hidden_size)
        
        # Feature extraction
        features = self.feature_extractor(global_avg)
        
        # Return-focused predictions
        return_pred = torch.tanh(self.return_prediction_head(features))
        volatility = torch.sigmoid(self.volatility_head(features))
        risk_adjusted = torch.tanh(self.risk_adjusted_head(features))
        
        # Return-optimized signal
        return_signal = (
            0.6 * return_pred + 0.25 * risk_adjusted + 0.15 * volatility
        )
        
        return return_signal

class MAREATrendModel(nn.Module):
    """MAREA-Trend: CNN + Bidirectional LSTM (Trend following)"""
    def __init__(self, input_size, hidden_size=144, dropout=0.16):
        super().__init__()
        self.hidden_size = hidden_size
        
        # CNN layers for pattern detection
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(input_size, hidden_size//3, kernel_size=3, padding=1),
            nn.Conv1d(hidden_size//3, hidden_size//3, kernel_size=5, padding=2),
            nn.Conv1d(hidden_size//3, hidden_size//3, kernel_size=7, padding=3)
        ])
        
        # Bidirectional LSTM for trend analysis
        self.bi_lstm = nn.LSTM(
            hidden_size//3, hidden_size//2, 2, 
            dropout=dropout, batch_first=True, bidirectional=True
        )
        
        # Trend-specific attention
        self.trend_attention = nn.MultiheadAttention(
            hidden_size, num_heads=6, dropout=dropout, batch_first=True
        )
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.ReLU(), nn.Dropout(0.2), nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.15), nn.BatchNorm1d(128),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1)
        )
        
        # Trend-focused predictions
        self.trend_direction_head = nn.Linear(64, 1)
        self.trend_strength_head = nn.Linear(64, 1)
        self.pattern_recognition_head = nn.Linear(64, 1)
        
    def forward(self, x):
        # CNN pattern detection
        x_conv = x.transpose(1, 2)  # (batch, features, seq_len)
        for conv in self.conv_layers:
            x_conv = F.relu(conv(x_conv))
        x_conv = x_conv.transpose(1, 2)  # Back to (batch, seq_len, features)
        
        # Bidirectional LSTM
        lstm_out, (hidden, _) = self.bi_lstm(x_conv)
        
        # Trend attention
        attn_out, _ = self.trend_attention(lstm_out, lstm_out, lstm_out)
        
        # Feature extraction - use only the final hidden state to avoid dimension mismatch
        # For bidirectional LSTM, concatenate last layer's forward and backward hidden states
        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (batch, hidden_size)
        features = self.feature_extractor(final_hidden)
        
        # Trend-focused predictions
        trend_direction = torch.tanh(self.trend_direction_head(features))
        trend_strength = torch.sigmoid(self.trend_strength_head(features))
        pattern_recognition = torch.tanh(self.pattern_recognition_head(features))
        
        # Trend-weighted signal
        trend_signal = (
            0.4 * trend_direction + 0.4 * trend_strength + 0.2 * pattern_recognition
        )
        
        return trend_signal

class MAREAHFModel(nn.Module):
    """MAREA-HF: 1D CNN + GRU (High-frequency trading)"""
    def __init__(self, input_size, hidden_size=128, dropout=0.18):
        super().__init__()
        self.hidden_size = hidden_size
        
        # High-frequency 1D CNN layers
        self.cnn_layers = nn.ModuleList([
            nn.Conv1d(input_size, hidden_size//2, kernel_size=2, padding=0),  # Short-term
            nn.Conv1d(hidden_size//2, hidden_size//2, kernel_size=3, padding=1),  # Medium-term
            nn.Conv1d(hidden_size//2, hidden_size//2, kernel_size=5, padding=2),  # Long-term
        ])
        
        # GRU for high-frequency sequence modeling
        self.gru = nn.GRU(hidden_size//2, hidden_size, 2, dropout=dropout, batch_first=True)
        
        # High-frequency attention
        self.hf_attention = nn.MultiheadAttention(
            hidden_size, num_heads=4, dropout=dropout, batch_first=True
        )
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.ReLU(), nn.Dropout(0.2), nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.15), nn.BatchNorm1d(128),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1)
        )
        
        # High-frequency predictions
        self.hf_signal_head = nn.Linear(64, 1)
        self.micro_trend_head = nn.Linear(64, 1)
        self.volatility_head = nn.Linear(64, 1)
        
    def forward(self, x):
        # High-frequency CNN processing
        x_conv = x.transpose(1, 2)  # (batch, features, seq_len)
        for conv in self.cnn_layers:
            x_conv = F.relu(conv(x_conv))
        x_conv = x_conv.transpose(1, 2)  # Back to (batch, seq_len, features)
        
        # GRU processing
        gru_out, hidden = self.gru(x_conv)
        
        # High-frequency attention
        attn_out, _ = self.hf_attention(gru_out, gru_out, gru_out)
        
        # Feature extraction - use only the final hidden state to avoid dimension mismatch
        final_hidden = hidden[-1]
        features = self.feature_extractor(final_hidden)
        
        # High-frequency predictions
        hf_signal = torch.tanh(self.hf_signal_head(features))
        micro_trend = torch.tanh(self.micro_trend_head(features))
        volatility = torch.sigmoid(self.volatility_head(features))
        
        # High-frequency signal
        hf_signal_combined = (
            0.5 * hf_signal + 0.3 * micro_trend + 0.2 * volatility
        )
        
        return hf_signal_combined

def create_diverse_model_configs():
    """Create diverse model configurations with different architectures"""
    return [
        # Ultra-aggressive GRU + Attention
        {
            'name': 'MAREA-Ultra-1',
            'architecture': 'GRU_Attention',
            'hidden_size': 192,
            'dropout': 0.12,
            'model_class': MAREAUltra1Model,
            'specialization': 'Ultra-aggressive maximum returns'
        },
        
        # Momentum-focused LSTM + TCN
        {
            'name': 'MAREA-Momentum',
            'architecture': 'LSTM_TCN',
            'hidden_size': 176,
            'dropout': 0.14,
            'model_class': MAREAMomentumModel,
            'specialization': 'Momentum and trend following'
        },
        
        # Return-optimized Transformer
        {
            'name': 'MAREA-Return',
            'architecture': 'Transformer',
            'hidden_size': 160,
            'dropout': 0.15,
            'model_class': MAREAReturnModel,
            'specialization': 'Pure return maximization'
        },
        
        # Trend-following CNN + BiLSTM
        {
            'name': 'MAREA-Trend',
            'architecture': 'CNN_BiLSTM',
            'hidden_size': 144,
            'dropout': 0.16,
            'model_class': MAREATrendModel,
            'specialization': 'Trend following and pattern recognition'
        },
        
        # High-frequency CNN + GRU
        {
            'name': 'MAREA-HF',
            'architecture': 'CNN_GRU',
            'hidden_size': 128,
            'dropout': 0.18,
            'model_class': MAREAHFModel,
            'specialization': 'High-frequency trading signals'
        }
    ]

def get_diverse_regime_weights():
    """Diverse regime-based weighting for different architectures"""
    return {
        0: np.array([0.45, 0.25, 0.15, 0.10, 0.05]),  # Bull market - GRU + Transformer focus
        1: np.array([0.15, 0.30, 0.25, 0.20, 0.10]),  # Bear market - LSTM + CNN focus
        2: np.array([0.20, 0.25, 0.25, 0.20, 0.10]),  # Sideways - Balanced
        3: np.array([0.10, 0.20, 0.25, 0.30, 0.15]),  # High volatility - CNN + BiLSTM focus
        4: np.array([0.50, 0.20, 0.15, 0.10, 0.05])   # Strong momentum - GRU + TCN focus
    }

# Training parameters for diverse architectures
DIVERSE_TRAINING_PARAMS = {
    'learning_rates': [0.002, 0.0018, 0.0015, 0.0012, 0.001],  # Different LRs per architecture
    'batch_size': 96,
    'epochs': 250,
    'patience': 25,
    'weight_decay': 1e-6,
    'gradient_clip': 2.0,
    'regime_detector_epochs': 80,
    'position_sizer_epochs': 50,
}

if __name__ == "__main__":
    print("🧠 MAREA-Ensemble Diverse Architectures")
    print("=" * 50)
    
    configs = create_diverse_model_configs()
    for i, config in enumerate(configs, 1):
        print(f"{i}. {config['name']}")
        print(f"   Architecture: {config['architecture']}")
        print(f"   Hidden Size: {config['hidden_size']}")
        print(f"   Dropout: {config['dropout']}")
        print(f"   Specialization: {config['specialization']}")
        print()
    
    print("🎯 Benefits of Diverse Architectures:")
    print("✅ GRU: Better gradient flow for financial sequences")
    print("✅ LSTM: Long-term memory for trend analysis")
    print("✅ Transformer: Self-attention for complex patterns")
    print("✅ CNN: Pattern detection in time series")
    print("✅ BiLSTM: Bidirectional context understanding")
    print("✅ TCN: Temporal convolution for momentum")
    print()
    print("🚀 Enhanced ensemble diversity for better performance!") 