#!/usr/bin/env python3
"""
MAREA-Ensemble Diverse Architecture Runner
Runs the MAREA-Ensemble system with diverse neural network architectures

This script demonstrates the enhanced MAREA-Ensemble with 5 different architectures:
1. GRU + Multi-Head Attention (Ultra-aggressive)
2. LSTM + Temporal Convolution (Momentum focus)
3. Transformer Encoder (Return optimization)
4. CNN + Bidirectional LSTM (Trend following)
5. 1D CNN + GRU (High-frequency trading)
"""

import sys
import torch
import numpy as np
import pandas as pd
from marea_diverse_architectures import (
    create_diverse_model_configs,
    get_diverse_regime_weights,
    DIVERSE_TRAINING_PARAMS
)
from marea_ensemble_system import MAREAEnsembleSystem
from return_optimizer import UltraReturnBoostLoss
from tradingPerformance import PerformanceEstimator

class MAREADiverseEnsembleSystem(MAREAEnsembleSystem):
    """Enhanced MAREA-Ensemble with diverse architectures"""
    
    def __init__(self, sequence_length=60, initial_balance=100000, device=None, 
                 return_boost_factor=1.25, ultra_aggressive_mode=True):
        super().__init__(sequence_length, initial_balance, device, return_boost_factor, ultra_aggressive_mode)
        self.framework_name = "MAREA-Diverse-Ensemble"
        self.version = "2.0"
        
        print(f"🧠 {self.framework_name} v{self.version} initialized")
        print(f"   Diverse architectures: 5 different neural network types")
        print(f"   Return boost factor: {return_boost_factor}")
        print(f"   Ultra-aggressive mode: {ultra_aggressive_mode}")
    
    def train_diverse_ensemble(self, n_models=5, epochs=250, batch_size=96):
        """Train the diverse architecture ensemble"""
        print(f"🔥 MAREA DIVERSE ARCHITECTURE ENSEMBLE TRAINING")
        print(f"   🎯 Framework: {self.framework_name} v{self.version}")
        print(f"   🚀 Training {n_models} diverse architecture models")
        print(f"   ⚠️  WARNING: Ultra-aggressive mode with diverse architectures")
        
        # Get diverse model configurations
        diverse_configs = create_diverse_model_configs()[:n_models]
        
        self.models = []
        criterion = UltraReturnBoostLoss(alpha=0.03, return_weight=0.95)
        
        # Data preparation
        dataset = torch.utils.data.TensorDataset(self.X, self.y)
        train_size = int(0.88 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training parameters
        learning_rates = DIVERSE_TRAINING_PARAMS['learning_rates']
        
        for i, config in enumerate(diverse_configs):
            print(f"\n  🔥 Training {config['name']} ({i+1}/{n_models})")
            print(f"     Architecture: {config['architecture']}")
            print(f"     Hidden Size: {config['hidden_size']}")
            print(f"     Specialization: {config['specialization']}")
            
            # Create diverse model
            model = config['model_class'](
                input_size=self.X.shape[2],
                hidden_size=config['hidden_size'],
                dropout=config['dropout']
            ).to(self.device)
            
            # Optimizer with architecture-specific learning rate
            lr = learning_rates[i] if i < len(learning_rates) else learning_rates[-1]
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=lr, 
                weight_decay=DIVERSE_TRAINING_PARAMS['weight_decay']
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)
            
            # Training loop
            best_val_loss = float('inf')
            patience = DIVERSE_TRAINING_PARAMS['patience']
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
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        DIVERSE_TRAINING_PARAMS['gradient_clip']
                    )
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
                    print(f"     ⏹️  Early stopping at epoch {epoch+1}")
                    break
                
                if (epoch + 1) % 30 == 0:
                    print(f"     📊 Epoch {epoch+1}: Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}")
            
            self.models.append(model)
            print(f"     ✅ {config['name']} final loss: {best_val_loss:.6f}")
        
        # Train regime detector and position sizer (same as original)
        print(f"\n🔍 Training MAREA Regime Detection System...")
        from return_optimizer import ReturnBoostRegimeDetector, PositionSizer
        self.regime_detector = ReturnBoostRegimeDetector(self.X.shape[2]).to(self.device)
        regime_labels = self._create_marea_regime_labels()
        
        optimizer = torch.optim.AdamW(self.regime_detector.parameters(), lr=0.001)
        criterion_regime = torch.nn.CrossEntropyLoss()
        
        for epoch in range(DIVERSE_TRAINING_PARAMS['regime_detector_epochs']):
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
            
            if (epoch + 1) % 20 == 0:
                print(f"     📊 Regime Detector Epoch {epoch+1}: Loss: {total_loss/len(train_loader):.6f}")
        
        # Train position sizer
        print(f"\n📈 Training MAREA Dynamic Position Sizing System...")
        self.position_sizer = PositionSizer(input_size=min(20, self.X.shape[2])).to(self.device)
        
        optimizer = torch.optim.Adam(self.position_sizer.parameters(), lr=0.0015)
        
        for epoch in range(DIVERSE_TRAINING_PARAMS['position_sizer_epochs']):
            self.position_sizer.train()
            total_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                
                position_features = batch_x[:, -1, :min(20, self.X.shape[2])]
                position_multiplier = self.position_sizer(position_features)
                
                target_positions = torch.sigmoid(batch_y.abs() * 3) * 1.2
                
                loss = torch.nn.MSELoss()(position_multiplier.squeeze(), target_positions)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 15 == 0:
                print(f"     📊 Position Sizer Epoch {epoch+1}: Loss: {total_loss/len(train_loader):.6f}")
        
        print(f"\n🏆 MAREA DIVERSE ARCHITECTURE TRAINING COMPLETE!")
        print(f"   ✅ {len(self.models)} diverse architecture models trained")
        print(f"   ✅ Regime detection system active")
        print(f"   ✅ Dynamic position sizing enabled")
        print(f"   🎯 System optimized for MAXIMUM returns with architecture diversity")
        
        return self.models
    
    def generate_diverse_signals(self, start_idx=None, end_idx=None):
        """Generate signals using diverse architecture ensemble"""
        print(f"🎯 Generating MAREA Diverse Architecture Signals...")
        
        if not self.models:
            raise ValueError("Models not trained. Run train_diverse_ensemble() first.")
        
        # Get diverse regime weights
        diverse_regime_weights = get_diverse_regime_weights()
        
        # Prepare data
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(self.X)
        
        X_test = self.X[start_idx:end_idx].to(self.device)
        
        # Get model predictions
        model_predictions = []
        for i, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                predictions = model(X_test)
                predictions = predictions.view(-1).cpu().numpy()
                model_predictions.append(predictions)
        model_predictions = np.stack(model_predictions)  # Shape: (n_models, n_samples)
        
        # Get regime predictions
        self.regime_detector.eval()
        with torch.no_grad():
            regime_probs = self.regime_detector(X_test)
            regime_predictions = torch.argmax(regime_probs, dim=1).cpu().numpy()
        
        # Get position sizing
        self.position_sizer.eval()
        with torch.no_grad():
            position_features = X_test[:, -1, :min(20, self.X.shape[2])]
            position_multipliers = self.position_sizer(position_features).cpu().numpy()
        
        # Generate ensemble signals with diverse weighting
        ensemble_signals = []
        
        for i in range(len(X_test)):
            current_regime = regime_predictions[i]
            regime_weights = diverse_regime_weights[current_regime]
            # Weighted ensemble prediction
            if model_predictions.ndim == 2:
                weighted_prediction = np.sum(model_predictions[:, i] * regime_weights)
            else:
                weighted_prediction = np.sum(model_predictions[i] * regime_weights)
            # Apply position sizing
            position_multiplier = position_multipliers[i, 0]
            final_signal = weighted_prediction * position_multiplier
            # Ultra-aggressive amplification
            final_signal = np.clip(final_signal * self.return_boost_factor, -1.2, 1.2)
            ensemble_signals.append(final_signal)
        ensemble_signals = np.array(ensemble_signals)
        
        print(f"   ✅ Generated {len(ensemble_signals)} diverse architecture signals")
        print(f"   📊 Signal range: [{ensemble_signals.min():.3f}, {ensemble_signals.max():.3f}]")
        print(f"   🎯 Average signal magnitude: {np.abs(ensemble_signals).mean():.3f}")
        
        return ensemble_signals

    def backtest_signals(self, signals):
        """Alias for backtest_novel_signals to ensure compatibility with runner script."""
        return self.backtest_novel_signals(signals)

def main():
    """Main function to run MAREA Diverse Ensemble on AAPL"""
    print("🚀 Starting MAREA Diverse Ensemble Runner...")
    
    import argparse
    
    parser = argparse.ArgumentParser(description='MAREA Diverse Ensemble Trading System')
    parser.add_argument('stock_symbol', type=str, help='Stock symbol (e.g., AAPL)')
    parser.add_argument('--mode', type=str, default='ultra-aggressive', 
                       choices=['ultra-aggressive', 'balanced'],
                       help='Trading mode')
    parser.add_argument('--balance', type=float, default=100000, help='Initial balance')
    parser.add_argument('--boost', type=float, default=1.25, help='Return boost factor')
    
    args = parser.parse_args()
    
    print("🧠 MAREA-Diverse-Ensemble Trading System")
    print("=" * 50)
    print(f"📈 Stock: {args.stock_symbol}")
    print(f"🎯 Mode: {args.mode}")
    print(f"💰 Initial Balance: ${args.balance:,.2f}")
    print(f"🚀 Return Boost: {args.boost}x")
    print()
    
    # Initialize diverse ensemble system
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    system = MAREADiverseEnsembleSystem(
        sequence_length=60,
        initial_balance=args.balance,
        device=device,
        return_boost_factor=args.boost,
        ultra_aggressive_mode=(args.mode == 'ultra-aggressive')
    )
    
    try:
        # Load and prepare data
        print("📊 Loading and preparing data...")
        system.load_and_prepare_data(stock_symbol=args.stock_symbol)
        system.create_enhanced_technical_indicators()
        system.prepare_sequences()
        
        # Train diverse ensemble
        print("\n🔥 Training diverse architecture ensemble...")
        system.train_diverse_ensemble(n_models=5, epochs=250, batch_size=96)
        
        # Generate signals
        print("\n🎯 Generating diverse architecture signals...")
        signals = system.generate_diverse_signals()
        
        # Backtest
        print("\n📈 Running backtest...")
        results = system.backtest_signals(signals)
        
        # Analyze performance
        print("\n🏆 Performance Analysis:")
        print("=" * 30)
        
        # Use the built-in analyze_performance method instead of PerformanceEstimator
        system.analyze_performance(results)
        
        print(f"\n🎯 MAREA Diverse Architecture Results:")
        print(f"   🧠 5 Different Neural Network Architectures")
        print(f"   🔄 Regime-Aware Adaptive Weighting")
        print(f"   📊 Dynamic Position Sizing")
        print(f"   🚀 Ultra-Aggressive Return Optimization")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    main() 