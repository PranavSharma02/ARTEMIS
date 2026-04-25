#!/usr/bin/env python3
"""
ATHENA Core Ensemble System

Inherits the same base as ARTEMIS (PyTorchNovelTradingSystem) and wires
together all ATHENA modifications:
  Mod 1  – Modified neural networks (residual + positional encoding)
  Mod 2  – Wavelet feature engineering
  Mod 3  – Cross-attention fusion
  Mod 4  – Dual-horizon signal generation
"""

import sys
import os

# Allow imports from parent directory (ARTEMIS root)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from pytorch_trading_system import PyTorchNovelTradingSystem

from athena_features import WaveletFeatureExtractor
from athena_networks import (
    ATHENAUltra1Model, ATHENAMomentumModel, ATHENAReturnModel,
    ATHENATrendModel, ATHENAHFModel, get_athena_model_configs, create_athena_model,
)
from athena_fusion import CrossAttentionFusion, SimpleFusion
from athena_signals import DualHorizonSignalGenerator, DualHorizonLoss
from athena_utils import (
    ATHENAReturnBoostLoss, ATHENARegimeDetector, ATHENAPositionSizer,
    create_athena_model_configs, get_athena_regime_weights,
    ATHENA_TRAINING_PARAMS,
)


class ATHENAEnsembleSystem(PyTorchNovelTradingSystem):
    """
    ATHENA: Adaptive Trading with Hierarchical Ensemble and Neural Attention

    Key differences from ARTEMIS:
      - Wavelet-enhanced feature engineering
      - Modified neural architectures (residual connections, positional encoding)
      - Cross-attention fusion instead of weighted average
      - Dual-horizon signal generation with learned gating
    """

    def __init__(self, sequence_length=60, initial_balance=100000, device=None,
                 return_boost_factor=1.0, ultra_aggressive_mode=True):
        super().__init__(sequence_length, initial_balance, device)

        self.return_boost_factor = return_boost_factor
        self.ultra_aggressive_mode = ultra_aggressive_mode
        self.framework_name = "ATHENA-Ensemble"
        self.version = "1.0"

        self.wavelet_extractor = WaveletFeatureExtractor()
        self.regime_detector = None
        self.position_sizer = None
        self.fusion_module = None
        self.signal_generator = None
        self.models = []

        print(f"[ATHENA] {self.framework_name} v{self.version} initialized")
        print(f"   Return boost factor: {return_boost_factor}")
        print(f"   Ultra-aggressive mode: {ultra_aggressive_mode}")

        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

    # ------------------------------------------------------------------
    # Feature Engineering  (Mod 2: Wavelet features)
    # ------------------------------------------------------------------

    def create_enhanced_technical_indicators(self):
        df = super().create_advanced_technical_indicators()

        # ATHENA-specific features (same as ARTEMIS additions)
        for period in [1, 2, 3, 5, 8, 13]:
            df[f'ATHENA_Momentum_Return_{period}'] = df['Returns'].rolling(period).sum()
            rs = df['Returns'].rolling(period).std().fillna(df['Returns'].std())
            df[f'ATHENA_Momentum_Strength_{period}'] = (
                df['Returns'].rolling(period).sum() / (rs + 1e-8)
            )

        for period in [5, 10, 15]:
            vol = df['Returns'].rolling(period).std().fillna(df['Returns'].std())
            df[f'ATHENA_Vol_Adj_Return_{period}'] = df['Returns'] / (vol + 1e-8)
            df[f'ATHENA_Vol_Scaled_Momentum_{period}'] = (
                df['Returns'].rolling(period).mean() / (vol + 1e-8)
            )

        df['ATHENA_Price_Acceleration'] = df['Returns'].diff().fillna(0)
        df['ATHENA_Price_Jerk'] = df['ATHENA_Price_Acceleration'].diff().fillna(0)

        for window in [10, 20, 30]:
            rmax = df['High'].rolling(window).max()
            rmin = df['Low'].rolling(window).min()
            df[f'ATHENA_Resistance_Break_{window}'] = (df['Close'] > rmax.shift(1)).astype(int)
            df[f'ATHENA_Support_Break_{window}'] = (df['Close'] < rmin.shift(1)).astype(int)

        for col, w in [('SMA_20', 20), ('SMA_50', 50), ('SMA_100', 100)]:
            if col not in df.columns:
                df[col] = df['Close'].rolling(w).mean()

        df['ATHENA_Bull_Strength'] = (
            (df['SMA_20'] > df['SMA_50']).fillna(False).astype(int) *
            (df['SMA_50'] > df['SMA_100']).fillna(False).astype(int) *
            (df['Close'] > df['SMA_20']).fillna(False).astype(int)
        )
        df['ATHENA_Bear_Strength'] = (
            (df['SMA_20'] < df['SMA_50']).fillna(False).astype(int) *
            (df['SMA_50'] < df['SMA_100']).fillna(False).astype(int) *
            (df['Close'] < df['SMA_20']).fillna(False).astype(int)
        )

        if 'Volume' in df.columns:
            vsma = df['Volume'].rolling(20).mean().fillna(df['Volume'].mean())
            df['ATHENA_Volume_Price_Trend'] = (
                np.sign(df['Returns']) * (df['Volume'] / (vsma + 1e-8))
            ).fillna(0)

        df['ATHENA_Return_Sign'] = np.sign(df['Returns']).fillna(0)
        df['ATHENA_Return_Magnitude'] = np.abs(df['Returns']).fillna(0)

        for p in [3, 5, 8]:
            df[f'ATHENA_Trend_{p}'] = df['Close'].rolling(p).mean() / df['Close'].shift(p) - 1

        # Wavelet features (Mod 2)
        df = self.wavelet_extractor.extract(df)

        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        self.features_df = df
        print(f"   [ATHENA] Enhanced features: {len(df.columns)} columns")
        return df

    # ------------------------------------------------------------------
    # Training  (Mods 1, 3, 4)
    # ------------------------------------------------------------------

    def train_athena_ensemble(self, n_models=5, epochs=250, batch_size=96):
        print(f"      [ATHENA] Training {n_models} models...")

        configs = get_athena_model_configs()[:n_models]
        self.models = []
        criterion = ATHENAReturnBoostLoss(alpha=0.03, return_weight=0.95)
        params = ATHENA_TRAINING_PARAMS

        dataset = torch.utils.data.TensorDataset(self.X, self.y)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        lrs = params['learning_rates']

        for i, cfg in enumerate(configs):
            model = create_athena_model(cfg, self.X.shape[2]).to(self.device)
            lr = lrs[i] if i < len(lrs) else lrs[-1]
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=params['weight_decay'])
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)

            best_val = float('inf')
            patience_ctr = 0

            for epoch in range(epochs):
                model.train()
                train_loss = 0
                for bx, by in train_loader:
                    optimizer.zero_grad()
                    out = model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), params['gradient_clip'])
                    optimizer.step()
                    train_loss += loss.item()

                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for bx, by in val_loader:
                        val_loss += criterion(model(bx), by).item()

                scheduler.step()
                avg_train = train_loss / len(train_loader)
                avg_val = val_loss / len(val_loader)

                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"         Model {i+1}/{len(configs)} | Epoch {epoch+1}/{epochs} | "
                          f"Train: {avg_train:.6f} | Val: {avg_val:.6f}", flush=True)

                if avg_val < best_val:
                    best_val = avg_val
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                if patience_ctr >= params['patience']:
                    print(f"         Model {i+1} early stopped at epoch {epoch+1}", flush=True)
                    break

            print(f"      ✅ Model {i+1}/{len(configs)} trained (best val: {best_val:.6f})", flush=True)
            self.models.append(model)

        # Regime detector
        print("      [ATHENA] Training regime detector...")
        self.regime_detector = ATHENARegimeDetector(self.X.shape[2]).to(self.device)
        regime_labels = self._create_athena_regime_labels()
        opt_r = optim.AdamW(self.regime_detector.parameters(), lr=0.001)
        ce = nn.CrossEntropyLoss()
        for _ in range(params['regime_detector_epochs']):
            self.regime_detector.train()
            for bx, _ in train_loader:
                opt_r.zero_grad()
                probs = self.regime_detector(bx)
                bl = regime_labels[train_ds.indices][:len(bx)]
                bl = torch.LongTensor(bl).to(self.device)
                ce(probs, bl).backward()
                opt_r.step()

        # Position sizer
        print("      [ATHENA] Training position sizer...")
        self.position_sizer = ATHENAPositionSizer(
            input_size=min(20, self.X.shape[2])
        ).to(self.device)
        opt_p = optim.Adam(self.position_sizer.parameters(), lr=0.0015)
        for _ in range(params['position_sizer_epochs']):
            self.position_sizer.train()
            for bx, by in train_loader:
                opt_p.zero_grad()
                pf = bx[:, -1, :min(20, self.X.shape[2])]
                pm = self.position_sizer(pf)
                target = torch.sigmoid(by.abs() * 3) * 1.2
                nn.MSELoss()(pm.squeeze(), target).backward()
                opt_p.step()

        # Cross-attention fusion (Mod 3)
        print("      [ATHENA] Training cross-attention fusion...")
        market_dim = min(20, self.X.shape[2])
        self.fusion_module = SimpleFusion(
            n_models=len(self.models), market_dim=market_dim
        ).to(self.device)
        opt_f = optim.Adam(self.fusion_module.parameters(), lr=0.001)
        fusion_criterion = ATHENAReturnBoostLoss(alpha=0.03, return_weight=0.95)

        for _ in range(params['fusion_epochs']):
            self.fusion_module.train()
            for bx, by in train_loader:
                opt_f.zero_grad()
                preds = []
                for m in self.models:
                    m.eval()
                    with torch.no_grad():
                        preds.append(m(bx).squeeze(-1))
                preds_t = torch.stack(preds, dim=1)
                mf = bx[:, -1, :market_dim]
                fused, _ = self.fusion_module(preds_t, mf)
                fusion_criterion(fused, by).backward()
                opt_f.step()

        # Dual-horizon signal generator (Mod 4)
        print("      [ATHENA] Training dual-horizon signal generator...")
        self.signal_generator = DualHorizonSignalGenerator(
            input_dim=len(self.models) + market_dim, hidden_dim=64
        ).to(self.device)
        opt_s = optim.Adam(self.signal_generator.parameters(), lr=0.001)
        sig_criterion = ATHENAReturnBoostLoss(alpha=0.03, return_weight=0.95)

        for _ in range(params['signal_generator_epochs']):
            self.signal_generator.train()
            for bx, by in train_loader:
                opt_s.zero_grad()
                preds = []
                for m in self.models:
                    m.eval()
                    with torch.no_grad():
                        preds.append(m(bx).squeeze(-1))
                preds_t = torch.stack(preds, dim=1)
                mf = bx[:, -1, :market_dim]
                sig_input = torch.cat([preds_t, mf], dim=1)
                blended, gate = self.signal_generator(sig_input)
                sig_criterion(blended, by).backward()
                opt_s.step()

        print("      [ATHENA] Ensemble training complete!")
        return self.models

    def _create_athena_regime_labels(self):
        n = len(self.X)
        labels = np.zeros(n)
        returns = self.features_df['Returns'].dropna()
        vol_col = 'Volatility_20' if 'Volatility_20' in self.features_df.columns else None
        if vol_col:
            volatility = self.features_df[vol_col].dropna()
        else:
            volatility = returns.rolling(20).std().dropna()

        momentum_col = 'ATHENA_Momentum_Return_5'
        if momentum_col in self.features_df.columns:
            momentum = self.features_df[momentum_col].dropna()
        else:
            momentum = returns.rolling(5).sum().dropna()

        min_len = min(n, len(returns), len(volatility), len(momentum))
        returns = returns.iloc[-min_len:]
        volatility = volatility.iloc[-min_len:]
        momentum = momentum.iloc[-min_len:]

        vol_thresh = volatility.median()
        mom_thresh = momentum.quantile(0.7)

        for i in range(min_len):
            if i < 20:
                labels[i] = 2
                continue
            recent = returns.iloc[max(0, i - 20):i]
            trend = recent.mean()
            cv = volatility.iloc[i]
            cm = momentum.iloc[i]
            if cm > mom_thresh:
                labels[i] = 4
            elif trend > 0.003 and cv < vol_thresh:
                labels[i] = 0
            elif trend < -0.003 and cv < vol_thresh:
                labels[i] = 1
            elif cv >= vol_thresh:
                labels[i] = 3
            else:
                labels[i] = 2

        return labels[-n:]

    # ------------------------------------------------------------------
    # Signal Generation  (Mods 3, 4 combined)
    # ------------------------------------------------------------------

    def generate_athena_signals(self, start_idx=None, end_idx=None):
        if not self.models:
            raise ValueError("ATHENA models must be trained first!")

        if start_idx is None:
            start_idx = self.sequence_length
        if end_idx is None:
            end_idx = len(self.features_df)

        feature_cols = self.feature_names
        data_sub = self.features_df.iloc[start_idx - self.sequence_length:end_idx]
        X_data = data_sub[feature_cols].values
        X_scaled = self.scaler.transform(X_data)

        X_seq = []
        for i in range(self.sequence_length, len(X_scaled)):
            X_seq.append(X_scaled[i - self.sequence_length:i])
        X_seq = torch.FloatTensor(np.array(X_seq)).to(self.device)

        # Ensemble predictions
        base_preds = []
        for m in self.models:
            m.eval()
            with torch.no_grad():
                base_preds.append(m(X_seq).cpu().numpy().flatten())
        base_preds = np.array(base_preds).T  # (T, n_models)

        # Regime detection
        if self.regime_detector:
            self.regime_detector.eval()
            with torch.no_grad():
                regime_probs = self.regime_detector(X_seq)
                dominant_regime = torch.argmax(regime_probs, dim=1).cpu().numpy()
        else:
            dominant_regime = np.zeros(len(base_preds))

        # Position sizing
        pos_mults = np.ones(len(base_preds))
        if self.position_sizer:
            self.position_sizer.eval()
            with torch.no_grad():
                pf = X_seq[:, -1, :min(20, X_seq.shape[2])]
                pos_mults = self.position_sizer(pf).cpu().numpy().flatten() * 1.3

        market_dim = min(20, X_seq.shape[2])
        final_signals = np.zeros(len(base_preds))

        # Use fusion + dual-horizon if trained, else fallback to weighted avg
        use_fusion = self.fusion_module is not None
        use_signal_gen = self.signal_generator is not None

        if use_fusion or use_signal_gen:
            preds_t = torch.FloatTensor(base_preds).to(self.device)
            mf_t = X_seq[:, -1, :market_dim]

            if use_signal_gen:
                self.signal_generator.eval()
                with torch.no_grad():
                    sig_input = torch.cat([preds_t, mf_t], dim=1)
                    blended, _ = self.signal_generator(sig_input)
                    raw_signals = blended.cpu().numpy().flatten()
            elif use_fusion:
                self.fusion_module.eval()
                with torch.no_grad():
                    fused, _ = self.fusion_module(preds_t, mf_t)
                    raw_signals = fused.cpu().numpy().flatten()

            for i in range(len(raw_signals)):
                multiplier = pos_mults[i] * self.return_boost_factor * 1.4
                final_signals[i] = np.clip(raw_signals[i] * multiplier, -1.2, 1.2)
        else:
            # Weighted-average fallback (similar to ARTEMIS)
            regime_weights = get_athena_regime_weights()
            for i in range(len(base_preds)):
                regime = int(dominant_regime[i])
                rw = np.array(regime_weights.get(regime, [0.2] * len(self.models))[:len(self.models)])
                rw = rw / rw.sum()
                base_sig = np.average(base_preds[i], weights=rw)
                mult = pos_mults[i] * self.return_boost_factor * 1.4
                final_signals[i] = np.clip(base_sig * mult, -1.2, 1.2)

        return final_signals
