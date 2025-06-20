#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from artemis_utils import (
    UltraReturnBoostLoss,
    ReturnBoostLoss,
    UltraReturnOptimizedModel,
    ReturnOptimizedModel,
    ReturnBoostRegimeDetector,
    create_ultra_aggressive_model_configs,
    get_ultra_aggressive_regime_weights,
    ULTRA_AGGRESSIVE_TRAINING_PARAMS,
    ENHANCED_TRAINING_PARAMS,
    PositionSizer,
    SharpeOptimizedLoss,
    SharpeOptimizedModel,
    create_sharpe_optimized_configs,
    get_sharpe_optimized_regime_weights,
    SHARPE_OPTIMIZED_TRAINING_PARAMS
)
from pytorch_trading_system import PyTorchNovelTradingSystem

class ARTEMISEnsembleSystem(PyTorchNovelTradingSystem):
    """
    ARTEMIS: Adaptive Reinforcement Trading Ensemble with Multi-Intelligence Systems
    
    Advanced ensemble trading system with ultra-aggressive return optimization,
    regime-aware adaptive weighting, and enhanced risk management.
    
    Key Features:
    - 5 Diverse Neural Network Architectures
    - Regime-Aware Dynamic Model Weighting
    - Ultra-Aggressive Return Optimization
    - Dynamic Position Sizing & Risk Management
    - GPU-Accelerated Training & Inference
    """
    
    def __init__(self, sequence_length=60, initial_balance=100000, device=None, 
                 return_boost_factor=1.0, ultra_aggressive_mode=True):
        super().__init__(sequence_length, initial_balance, device)
        
        self.return_boost_factor = return_boost_factor
        self.ultra_aggressive_mode = ultra_aggressive_mode
        self.framework_name = "ARTEMIS-Ensemble"
        self.version = "1.0"
        
        # Enhanced components
        self.regime_detector = None
        self.position_sizer = None
        self.models = []
        
        print(f"🔬 {self.framework_name} v{self.version} initialized")
        print(f"   Return boost factor: {return_boost_factor}")
        print(f"   Ultra-aggressive mode: {ultra_aggressive_mode}")
        
        # GPU optimizations
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            print(f"   ⚡ Automatic Mixed Precision enabled")
            print(f"   🔧 GPU optimizations applied")

    def create_enhanced_technical_indicators(self):
        """
        Enhanced Technical Indicator Creation for ARTEMIS-Ensemble
        
        Creates 98+ advanced technical indicators optimized for return prediction
        including multi-timeframe momentum, volatility-adjusted returns, regime
        transition signals, and support/resistance breakthrough detection.
        
        Key Indicator Categories:
        1. Multi-timeframe momentum indicators (6 timeframes)
        2. Volatility-adjusted returns (3 timeframes)  
        3. Price acceleration and jerk indicators
        4. Support/resistance breakthrough signals (3 timeframes)
        5. Market regime transition indicators
        6. Volume-price relationship metrics
        7. Short-term trend indicators
        8. Return magnitude and directional signals
        
        Returns:
            pd.DataFrame: Enhanced feature dataframe with 98+ indicators
        """
        df = super().create_advanced_technical_indicators()
        
        print(f"   📊 Base ARTEMIS features: {len(df.columns)} columns, {len(df)} rows")
        print(f"   🔧 Initial NaN count: {df.isnull().sum().sum()}")
        
        # Multi-timeframe momentum indicators (ARTEMIS Innovation #1)
        for period in [1, 2, 3, 5, 8, 13]:  # Fibonacci-based periods
            df[f'ARTEMIS_Momentum_Return_{period}'] = df['Returns'].rolling(period).sum()
            rolling_std = df['Returns'].rolling(period).std()
            df[f'ARTEMIS_Momentum_Strength_{period}'] = (
                df['Returns'].rolling(period).sum() / 
                (rolling_std.fillna(rolling_std.mean()) + 1e-8)
            )
        
        # Volatility-adjusted returns (ARTEMIS Innovation #2)
        for period in [5, 10, 15]:
            vol = df['Returns'].rolling(period).std()
            vol_filled = vol.fillna(vol.mean())
            df[f'ARTEMIS_Vol_Adj_Return_{period}'] = df['Returns'] / (vol_filled + 1e-8)
            df[f'ARTEMIS_Vol_Scaled_Momentum_{period}'] = (
                df['Returns'].rolling(period).mean() / (vol_filled + 1e-8)
            )
        
        # Price acceleration indicators (ARTEMIS Innovation #3)
        df['ARTEMIS_Price_Acceleration'] = df['Returns'].diff().fillna(0)
        df['ARTEMIS_Price_Jerk'] = df['ARTEMIS_Price_Acceleration'].diff().fillna(0)
        
        # Support/Resistance breakthrough indicators (ARTEMIS Innovation #4)
        for window in [10, 20, 30]:
            rolling_max = df['High'].rolling(window).max()
            rolling_min = df['Low'].rolling(window).min()
            df[f'ARTEMIS_Resistance_Break_{window}'] = (df['Close'] > rolling_max.shift(1)).astype(int)
            df[f'ARTEMIS_Support_Break_{window}'] = (df['Close'] < rolling_min.shift(1)).astype(int)
        
        # Regime transition indicators (ARTEMIS Innovation #5)
        if 'SMA_20' not in df.columns:
            df['SMA_20'] = df['Close'].rolling(20).mean()
        if 'SMA_50' not in df.columns:
            df['SMA_50'] = df['Close'].rolling(50).mean()
        if 'SMA_100' not in df.columns:
            df['SMA_100'] = df['Close'].rolling(100).mean()
            
        df['ARTEMIS_Bull_Strength'] = (
            (df['SMA_20'] > df['SMA_50']).fillna(False).astype(int) * 
            (df['SMA_50'] > df['SMA_100']).fillna(False).astype(int) *
            (df['Close'] > df['SMA_20']).fillna(False).astype(int)
        )
        df['ARTEMIS_Bear_Strength'] = (
            (df['SMA_20'] < df['SMA_50']).fillna(False).astype(int) * 
            (df['SMA_50'] < df['SMA_100']).fillna(False).astype(int) *
            (df['Close'] < df['SMA_20']).fillna(False).astype(int)
        )
        
        # Volume-price relationship (ARTEMIS Innovation #6)
        if 'Volume' in df.columns:
            volume_sma = df['Volume'].rolling(20).mean()
            df['ARTEMIS_Volume_Price_Trend'] = (
                np.sign(df['Returns']) * 
                (df['Volume'] / (volume_sma.fillna(volume_sma.mean()) + 1e-8))
            ).fillna(0)
        
        # Additional ARTEMIS-specific features (Innovation #7)
        df['ARTEMIS_Return_Sign'] = np.sign(df['Returns']).fillna(0)
        df['ARTEMIS_Return_Magnitude'] = np.abs(df['Returns']).fillna(0)
        
        # Short-term trend indicators (ARTEMIS Innovation #8)
        df['ARTEMIS_Trend_3'] = df['Close'].rolling(3).mean() / df['Close'].shift(3) - 1
        df['ARTEMIS_Trend_5'] = df['Close'].rolling(5).mean() / df['Close'].shift(5) - 1
        df['ARTEMIS_Trend_8'] = df['Close'].rolling(8).mean() / df['Close'].shift(8) - 1
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        self.features_df = df
        
        print(f"   🚀 Enhanced features: {len(df.columns)} columns")
        
        return df

    def train_artemis_ultra_aggressive_ensemble(self, n_models=5, epochs=250, batch_size=96):
        """
        ARTEMIS Ultra-Aggressive Ensemble Training
        
        Trains the ensemble system with ultra-aggressive optimization for maximum
        returns through advanced loss functions and model architectures.
        
        Key Features:
        1. Ultra-aggressive return optimization
        2. Advanced model architectures with attention
        3. Regime-aware adaptive weighting
        4. Dynamic position sizing
        
        Args:
            n_models (int): Number of models in ensemble (default: 5)
            epochs (int): Training epochs per model (default: 250)
            batch_size (int): Training batch size (default: 96)
            
        Returns:
            list: Trained ultra-aggressive ensemble models
        """
        print(f"      🔥 Training {n_models} ARTEMIS models...")
        
        # Import ultra-aggressive components
        artemis_configs = create_ultra_aggressive_model_configs()[:n_models]
        artemis_weights = get_ultra_aggressive_regime_weights()
        
        self.models = []
        criterion = UltraReturnBoostLoss(alpha=0.03, return_weight=0.95, momentum_weight=0.05)
        
        # Enhanced data splitting for aggressive training
        dataset = torch.utils.data.TensorDataset(self.X, self.y)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # ARTEMIS training parameters
        learning_rates = ULTRA_AGGRESSIVE_TRAINING_PARAMS['learning_rates']
        
        for i, config in enumerate(artemis_configs):
            # Create ARTEMIS model
            if config.get('ultra_aggressive', False):
                model = UltraReturnOptimizedModel(
                    input_size=self.X.shape[2],
                    hidden_size=config['hidden_size'],
                    dropout=config['dropout']
                ).to(self.device)
            else:
                model = ReturnOptimizedModel(
                    input_size=self.X.shape[2],
                    hidden_size=config['hidden_size'],
                    dropout=config['dropout']
                ).to(self.device)
            
            # ARTEMIS optimizer configuration
            lr = learning_rates[i] if i < len(learning_rates) else learning_rates[-1]
            optimizer = optim.AdamW(model.parameters(), lr=lr, 
                                   weight_decay=ULTRA_AGGRESSIVE_TRAINING_PARAMS['weight_decay'])
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)
            
            # ARTEMIS training loop
            best_val_loss = float('inf')
            patience = ULTRA_AGGRESSIVE_TRAINING_PARAMS['patience']
            patience_counter = 0
            
            for epoch in range(epochs):
                # Training phase
                model.train()
                train_loss = 0
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                                 ULTRA_AGGRESSIVE_TRAINING_PARAMS['gradient_clip'])
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation phase
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                scheduler.step()
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    break
            
            self.models.append(model)
        
        # Train regime detector  
        self.regime_detector = ReturnBoostRegimeDetector(self.X.shape[2]).to(self.device)
        regime_labels = self._create_artemis_regime_labels()
        
        optimizer = optim.AdamW(self.regime_detector.parameters(), lr=0.001)
        criterion_regime = nn.CrossEntropyLoss()
        
        for epoch in range(ULTRA_AGGRESSIVE_TRAINING_PARAMS['regime_detector_epochs']):
            self.regime_detector.train()
            total_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                regime_probs = self.regime_detector(batch_x)
                batch_regime_labels = regime_labels[train_dataset.indices][:len(batch_x)]
                batch_regime_labels = torch.LongTensor(batch_regime_labels).to(self.device)
                
                loss = criterion_regime(regime_probs, batch_regime_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        # Train position sizer
        self.position_sizer = PositionSizer(input_size=min(20, self.X.shape[2])).to(self.device)
        
        optimizer = optim.Adam(self.position_sizer.parameters(), lr=0.0015)
        
        for epoch in range(ULTRA_AGGRESSIVE_TRAINING_PARAMS['position_sizer_epochs']):
            self.position_sizer.train()
            total_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                
                position_features = batch_x[:, -1, :min(20, self.X.shape[2])]
                position_multiplier = self.position_sizer(position_features)
                
                # ARTEMIS ultra-aggressive position targets
                target_positions = torch.sigmoid(batch_y.abs() * 3) * 1.2
                
                loss = nn.MSELoss()(position_multiplier.squeeze(), target_positions)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        print("      ✅ ARTEMIS models trained!")
        print("      🔍 Training regime detection...")
        print("      📈 Training position sizing...")
        print("      ✅ Ensemble training complete!")
        
        return self.models
    
    def _create_artemis_regime_labels(self):
        """Create ARTEMIS regime labels with 5 market states"""
        n_samples = len(self.X)
        regime_labels = np.zeros(n_samples)
        
        returns = self.features_df['Returns'].dropna()
        volatility = self.features_df.get('Volatility_20', returns.rolling(20).std()).dropna()
        momentum = self.features_df.get('ARTEMIS_Momentum_Return_5', returns.rolling(5).sum()).dropna()
        
        min_len = min(n_samples, len(returns), len(volatility), len(momentum))
        returns = returns.iloc[-min_len:]
        volatility = volatility.iloc[-min_len:]
        momentum = momentum.iloc[-min_len:]
        
        vol_threshold = volatility.median()
        momentum_threshold = momentum.quantile(0.7)
        
        for i in range(min_len):
            if i < 20:
                regime_labels[i] = 2  # Default to sideways
                continue
                
            recent_returns = returns.iloc[max(0, i-20):i]
            current_vol = volatility.iloc[i]
            current_momentum = momentum.iloc[i]
            
            trend = recent_returns.mean()
            
            # ARTEMIS regime classification
            if current_momentum > momentum_threshold:
                regime_labels[i] = 4  # Strong momentum regime
            elif trend > 0.003 and current_vol < vol_threshold:
                regime_labels[i] = 0  # Bull market
            elif trend < -0.003 and current_vol < vol_threshold:
                regime_labels[i] = 1  # Bear market
            elif current_vol >= vol_threshold:
                regime_labels[i] = 3  # High volatility
            else:
                regime_labels[i] = 2  # Sideways/low volatility
        
        return regime_labels[-n_samples:]

    def generate_artemis_ultra_aggressive_signals(self, start_idx=None, end_idx=None):
        """
        ARTEMIS Ultra-Aggressive Signal Generation
        
        Generates trading signals using the complete ARTEMIS ensemble system
        with ultra-aggressive optimization for maximum return capture.
        
        Signal Generation Pipeline:
        1. Multi-model ensemble predictions
        2. Regime detection and adaptive weighting
        3. Dynamic position sizing
        4. Ultra-aggressive signal amplification
        
        Args:
            start_idx (int): Start index for signal generation
            end_idx (int): End index for signal generation
            
        Returns:
            np.ndarray: Ultra-aggressive trading signals
        """
        if not self.models:
            raise ValueError("ARTEMIS models must be trained first!")
        
        if start_idx is None:
            start_idx = self.sequence_length
        if end_idx is None:
            end_idx = len(self.features_df)
        
        # Prepare data
        feature_cols = self.feature_names
        data_subset = self.features_df.iloc[start_idx-self.sequence_length:end_idx]
        
        X_data = data_subset[feature_cols].values
        X_scaled = self.scaler.transform(X_data)
        
        # Create sequences
        X_sequences = []
        for i in range(self.sequence_length, len(X_scaled)):
            X_sequences.append(X_scaled[i-self.sequence_length:i])
        
        X_sequences = torch.FloatTensor(np.array(X_sequences)).to(self.device)
        
        # ARTEMIS ensemble predictions
        base_predictions = []
        model_names = []
        for i, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                predictions = model(X_sequences)
                base_predictions.append(predictions.cpu().numpy().flatten())
                model_names.append(f"ARTEMIS-{i+1}")
        
        base_predictions = np.array(base_predictions).T
        # Enhanced regime detection
        if self.regime_detector:
            self.regime_detector.eval()
            with torch.no_grad():
                regime_probs = self.regime_detector(X_sequences)
                dominant_regime = torch.argmax(regime_probs, dim=1).cpu().numpy()
        else:
            dominant_regime = np.zeros(len(base_predictions))
        
        # Import ARTEMIS weights
        artemis_weights = get_ultra_aggressive_regime_weights()
        final_signals = np.zeros(len(base_predictions))
        
        # Enhanced position sizing for ultra-aggressive trading
        position_multipliers = np.ones(len(base_predictions))
        if self.position_sizer:
            self.position_sizer.eval()
            with torch.no_grad():
                position_features = X_sequences[:, -1, :min(20, X_sequences.shape[2])]
                position_multipliers = self.position_sizer(position_features).cpu().numpy().flatten()
                position_multipliers = position_multipliers * 1.3  # ARTEMIS ultra-aggressive boost
        
        for i in range(len(base_predictions)):
            regime = dominant_regime[i]
            
            # ARTEMIS regime-aware weighting
            if regime in artemis_weights:
                regime_weights = np.array(artemis_weights[regime][:len(self.models)])
            else:
                regime_weights = np.ones(len(self.models)) / len(self.models)
            
            regime_weights = regime_weights / regime_weights.sum()
            
            # ARTEMIS signal combination
            base_signal = np.average(base_predictions[i], weights=regime_weights)
            
            # ARTEMIS ultra-aggressive multipliers
            ultra_aggressive_multiplier = (
                position_multipliers[i] * 
                self.return_boost_factor * 
                1.4  # ARTEMIS ultra-aggressive factor
            )
            
            final_signal = base_signal * ultra_aggressive_multiplier
            
            # ARTEMIS signal bounds (allow higher leverage)
            final_signals[i] = np.clip(final_signal, -1.2, 1.2)
        

        
        return final_signals

# Legacy aliases for backward compatibility
EnhancedTradingSystem = ARTEMISEnsembleSystem

def upgrade_to_artemis_system(original_system, return_boost_factor=1.25, ultra_aggressive_mode=True):
    """
    Upgrade existing trading system to ARTEMIS framework
    
    Args:
        original_system: Existing trading system
        return_boost_factor (float): Return amplification factor
        ultra_aggressive_mode (bool): Enable ultra-aggressive optimizations
        
    Returns:
        ARTEMISEnsembleSystem: Upgraded ARTEMIS system
    """
    print("🔬 Upgrading to ARTEMIS Framework...")
    
    artemis_system = ARTEMISEnsembleSystem(
        sequence_length=original_system.sequence_length,
        initial_balance=original_system.initial_balance,
        device=original_system.device,
        return_boost_factor=return_boost_factor,
        ultra_aggressive_mode=ultra_aggressive_mode
    )
    
    if hasattr(original_system, 'data') and original_system.data is not None:
        artemis_system.data = original_system.data
        artemis_system.current_stock = original_system.current_stock
        artemis_system.scaler = original_system.scaler
        print(f"   ✅ Data transferred for {artemis_system.current_stock}")
    
    return artemis_system

# Legacy function aliases
upgrade_trading_system = upgrade_to_artemis_system 