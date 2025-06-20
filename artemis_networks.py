#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ARTEMISUltra1Model(nn.Module):
    """ARTEMIS Ultra-Aggressive Model 1: Advanced GRU + Multi-Head Attention"""
    def __init__(self, input_size, hidden_size=192, dropout=0.12):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Triple-layer GRU for deep feature extraction
        self.gru1 = nn.GRU(input_size, hidden_size, 3, dropout=dropout, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size//2, 2, dropout=dropout, batch_first=True)
        self.gru3 = nn.GRU(hidden_size//2, hidden_size//4, 1, dropout=dropout, batch_first=True)
        
        # Multi-head attention for complex pattern recognition
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=16, dropout=dropout, batch_first=True)
        
        # Combined feature processing
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
        
        # Multiple prediction heads for ensemble-like behavior
        self.return_head = nn.Linear(64, 1)
        self.momentum_head = nn.Linear(64, 1)
        self.volatility_head = nn.Linear(64, 1)
        self.regime_head = nn.Linear(64, 1)
        
    def forward(self, x):
        # Deep GRU processing
        gru1_out, _ = self.gru1(x)
        
        # Attention mechanism
        attn_out, _ = self.attention(gru1_out, gru1_out, gru1_out)
        
        # Progressive dimensionality reduction
        gru2_out, hidden2 = self.gru2(attn_out)
        gru3_out, hidden3 = self.gru3(gru2_out)
        
        # Multi-scale feature combination
        final_hidden = hidden3[-1]
        global_features = gru3_out.mean(dim=1)
        max_features = gru3_out.max(dim=1)[0]
        
        combined_features = torch.cat([final_hidden, global_features, max_features], dim=1)
        features = self.feature_extractor(combined_features)
        
        # Multi-head predictions
        return_pred = self.return_head(features)
        momentum_pred = self.momentum_head(features)
        volatility_pred = self.volatility_head(features)
        regime_pred = self.regime_head(features)
        
        # ARTEMIS ultra-aggressive combination
        ultra_aggressive_prediction = (
            1.0 * return_pred + 
            0.4 * momentum_pred - 
            0.1 * volatility_pred + 
            0.2 * regime_pred
        )
        
        return ultra_aggressive_prediction

class ARTEMISMomentumModel(nn.Module):
    """ARTEMIS Momentum Model: Specialized for momentum and trend capture"""
    def __init__(self, input_size, hidden_size=176, dropout=0.14):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # LSTM layers for momentum memory
        self.lstm1 = nn.LSTM(input_size, hidden_size, 2, dropout=dropout, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size//2, 1, dropout=dropout, batch_first=True)
        
        # Temporal Convolutional Network for trend detection
        self.tcn = nn.Conv1d(hidden_size//2, hidden_size//2, kernel_size=3, padding=1)
        
        # Attention for momentum pattern focus
        self.momentum_attention = nn.MultiheadAttention(hidden_size//2, num_heads=8, dropout=dropout, batch_first=True)
        
        combined_feature_size = hidden_size//2 + hidden_size//2
        
        self.momentum_extractor = nn.Sequential(
            nn.Linear(combined_feature_size, 256),
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
        
        self.momentum_head = nn.Linear(64, 1)
        self.trend_head = nn.Linear(64, 1)
        self.acceleration_head = nn.Linear(64, 1)
        
    def forward(self, x):
        # LSTM processing for momentum memory
        lstm1_out, _ = self.lstm1(x)
        lstm2_out, (hidden, cell) = self.lstm2(lstm1_out)
        
        # TCN for trend detection
        tcn_input = lstm2_out.transpose(1, 2)  # (batch, features, sequence)
        tcn_out = F.relu(self.tcn(tcn_input))
        tcn_out = tcn_out.transpose(1, 2)  # back to (batch, sequence, features)
        
        # Momentum-focused attention
        attn_out, _ = self.momentum_attention(tcn_out, tcn_out, tcn_out)
        
        # Feature combination
        final_hidden = hidden[-1]
        momentum_features = attn_out.mean(dim=1)
        
        combined_features = torch.cat([final_hidden, momentum_features], dim=1)
        features = self.momentum_extractor(combined_features)
        
        # Momentum-specific predictions
        momentum_pred = self.momentum_head(features)
        trend_pred = self.trend_head(features)
        acceleration_pred = self.acceleration_head(features)
        
        # ARTEMIS momentum combination
        momentum_signal = (
            0.6 * momentum_pred + 
            0.3 * trend_pred + 
            0.1 * acceleration_pred
        )
        
        return momentum_signal

class ARTEMISReturnModel(nn.Module):
    """ARTEMIS Return Model: Optimized for pure return generation"""
    def __init__(self, input_size, hidden_size=160, dropout=0.15):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Transformer-like architecture for return patterns
        self.embedding = nn.Linear(input_size, hidden_size)
        
        # Multiple transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=8, 
            dim_feedforward=hidden_size*2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Return-focused processing
        self.return_processor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.05)
        )
        
        # Return prediction heads
        self.short_return_head = nn.Linear(64, 1)
        self.medium_return_head = nn.Linear(64, 1)
        self.long_return_head = nn.Linear(64, 1)
        
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        
        # Transformer processing
        transformer_out = self.transformer(embedded)
        
        # Global feature extraction
        return_features = transformer_out.mean(dim=1)
        features = self.return_processor(return_features)
        
        # Multi-horizon return predictions
        short_return = self.short_return_head(features)
        medium_return = self.medium_return_head(features)
        long_return = self.long_return_head(features)
        
        # ARTEMIS return combination
        return_signal = (
            0.5 * short_return + 
            0.3 * medium_return + 
            0.2 * long_return
        )
        
        return return_signal

class ARTEMISTrendModel(nn.Module):
    """ARTEMIS Trend Model: CNN + BiLSTM for trend analysis"""
    def __init__(self, input_size, hidden_size=144, dropout=0.16):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # CNN layers for local pattern detection
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, hidden_size, kernel_size=7, padding=3)
        
        # Bidirectional LSTM for trend direction
        self.bilstm = nn.LSTM(hidden_size, hidden_size//2, 2, 
                             dropout=dropout, batch_first=True, bidirectional=True)
        
        # Trend analysis layers
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
            nn.Dropout(0.05)
        )
        
        # Trend prediction heads
        self.trend_direction_head = nn.Linear(64, 1)
        self.trend_strength_head = nn.Linear(64, 1)
        self.trend_persistence_head = nn.Linear(64, 1)
        
    def forward(self, x):
        # CNN processing
        x_transposed = x.transpose(1, 2)  # (batch, features, sequence)
        conv1_out = F.relu(self.conv1(x_transposed))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        
        # Back to sequence format
        cnn_out = conv3_out.transpose(1, 2)  # (batch, sequence, features)
        
        # Bidirectional LSTM
        bilstm_out, (hidden, cell) = self.bilstm(cnn_out)
        
        # Feature extraction
        trend_features = bilstm_out.mean(dim=1)
        features = self.trend_analyzer(trend_features)
        
        # Trend predictions
        direction = self.trend_direction_head(features)
        strength = self.trend_strength_head(features)
        persistence = self.trend_persistence_head(features)
        
        # ARTEMIS trend combination
        trend_signal = (
            0.5 * direction + 
            0.3 * strength + 
            0.2 * persistence
        )
        
        return trend_signal

class ARTEMISHFModel(nn.Module):
    """ARTEMIS High-Frequency Model: CNN + GRU for high-frequency patterns"""
    def __init__(self, input_size, hidden_size=128, dropout=0.18):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Multi-scale CNN for different time scales
        self.conv_1min = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.conv_5min = nn.Conv1d(input_size, 32, kernel_size=5, padding=2)
        self.conv_15min = nn.Conv1d(input_size, 32, kernel_size=7, padding=3)
        self.conv_60min = nn.Conv1d(input_size, 32, kernel_size=15, padding=7)
        
        # Combine multi-scale features
        combined_features = 128  # 32 * 4
        
        # GRU for temporal dynamics
        self.gru = nn.GRU(combined_features, hidden_size, 3, 
                         dropout=dropout, batch_first=True)
        
        # High-frequency pattern detector
        self.hf_detector = nn.Sequential(
            nn.Linear(hidden_size, 256),
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
        
        # High-frequency prediction heads
        self.microtrend_head = nn.Linear(64, 1)
        self.volatility_head = nn.Linear(64, 1)
        self.noise_head = nn.Linear(64, 1)
        
    def forward(self, x):
        # Multi-scale convolutions
        x_transposed = x.transpose(1, 2)  # (batch, features, sequence)
        
        conv_1min = F.relu(self.conv_1min(x_transposed))
        conv_5min = F.relu(self.conv_5min(x_transposed))
        conv_15min = F.relu(self.conv_15min(x_transposed))
        conv_60min = F.relu(self.conv_60min(x_transposed))
        
        # Combine scales
        multi_scale = torch.cat([conv_1min, conv_5min, conv_15min, conv_60min], dim=1)
        multi_scale = multi_scale.transpose(1, 2)  # back to (batch, sequence, features)
        
        # GRU processing
        gru_out, hidden = self.gru(multi_scale)
        
        # High-frequency analysis
        hf_features = hidden[-1]
        features = self.hf_detector(hf_features)
        
        # High-frequency predictions
        microtrend = self.microtrend_head(features)
        volatility = self.volatility_head(features)
        noise = self.noise_head(features)
        
        # ARTEMIS high-frequency combination
        hf_signal = (
            0.6 * microtrend - 
            0.2 * volatility - 
            0.2 * noise
        )
        
        return hf_signal

def get_artemis_model_configs():
    """Get ARTEMIS model configurations"""
    return [
        {
            'model_class': ARTEMISUltra1Model,
            'name': 'ARTEMIS-Ultra-1',
            'hidden_size': 192,
            'dropout': 0.12,
            'ultra_aggressive': True
        },
        {
            'model_class': ARTEMISMomentumModel,
            'name': 'ARTEMIS-Momentum',
            'hidden_size': 176,
            'dropout': 0.14,
            'momentum_focus': True
        },
        {
            'model_class': ARTEMISReturnModel,
            'name': 'ARTEMIS-Return',
            'hidden_size': 160,
            'dropout': 0.15,
            'return_focus': True
        },
        {
            'model_class': ARTEMISTrendModel,
            'name': 'ARTEMIS-Trend',
            'hidden_size': 144,
            'dropout': 0.16,
            'trend_focus': True
        },
        {
            'model_class': ARTEMISHFModel,
            'name': 'ARTEMIS-HF',
            'hidden_size': 128,
            'dropout': 0.18,
            'high_freq': True
        }
    ]

def create_artemis_model(config, input_size):
    """Create an ARTEMIS model from configuration"""
    model_class = config['model_class']
    return model_class(
        input_size=input_size,
        hidden_size=config['hidden_size'],
        dropout=config['dropout']
    )

# Legacy compatibility functions
def create_ultra_aggressive_model_configs():
    """Legacy compatibility for ultra aggressive configs"""
    configs = get_artemis_model_configs()
    return [
        {
            'hidden_size': config['hidden_size'],
            'dropout': config['dropout'],
            'use_attention': True,
            'ultra_aggressive': config.get('ultra_aggressive', False),
            'momentum_focus': config.get('momentum_focus', False),
            'return_focus': config.get('return_focus', False),
            'trend_focus': config.get('trend_focus', False),
            'high_freq': config.get('high_freq', False),
            'name': config['name']
        }
        for config in configs
    ] 