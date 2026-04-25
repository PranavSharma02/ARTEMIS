#!/usr/bin/env python3
"""
ATHENA Hybrid RL System

Combines supervised ATHENA ensemble with reinforcement learning.
Modifications 5 & 6 vs ARTEMIS:
  Mod 5 – SAC (Soft Actor-Critic) replaces TD3
  Mod 6 – Adaptive Confidence Gate replaces fixed 70/30 blend
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from athena_core import ATHENAEnsembleSystem
from athena_algorithms import (
    ATHENAMultiAgentSystem,
    AdaptiveConfidenceGate,
)


class HybridRLATHENASystem(ATHENAEnsembleSystem):
    """
    Hybrid RL-ATHENA System

    Training pipeline:
      1. Train supervised ATHENA ensemble (Mods 1-4)
      2. Establish supervised baseline
      3. Initialise SAC multi-agent RL system (Mod 5)
      4. Train RL agents conservatively
      5. Train adaptive confidence gate (Mod 6)
      6. Generate hybrid signals using learned gate
    """

    def __init__(self, sequence_length=60, initial_balance=100000, device=None,
                 return_boost_factor=1.25, ultra_aggressive_mode=True,
                 rl_learning_rate=1e-4):
        super().__init__(sequence_length, initial_balance, device,
                         return_boost_factor, ultra_aggressive_mode)

        self.framework_name = "Hybrid-RL-ATHENA"
        self.version = "1.0"
        self.rl_learning_rate = rl_learning_rate

        self.multi_agent_system = None
        self.confidence_gate = None
        self.rl_agents_initialized = False

        self.supervised_baseline_performance = None
        self.rl_performance_history = []
        self.performance_violations = 0
        self.max_violations = 3
        self.performance_threshold = 0.95
        self.fallback_mode = False

        print(f"   [{self.framework_name}] v{self.version} initialized")
        print(f"   SAC RL agent + Adaptive Confidence Gate")

        if self.device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------

    def train_hybrid_system(self, n_supervised_models=5, supervised_epochs=250,
                            rl_episodes=500, evaluation_frequency=50):
        print("   Phase 2.1: Training supervised ensemble...")
        self.train_athena_ensemble(n_models=n_supervised_models, epochs=supervised_epochs)

        print("   Phase 2.2: Baseline evaluation...")
        baseline_signals = self.generate_athena_signals()
        baseline_results = self.backtest_signals(baseline_signals)
        self.supervised_baseline_performance = {
            'annual_return': baseline_results['annual_return'],
            'sharpe_ratio': baseline_results['sharpe_ratio'],
            'max_drawdown': baseline_results['max_drawdown'],
        }

        print("   Phase 2.3: Initializing SAC RL agents...")
        self._initialize_rl_agents()

        print(f"   Phase 2.4: RL training ({rl_episodes} episodes)...")
        self._train_rl_agents_conservatively(rl_episodes, evaluation_frequency)

        print("   Phase 2.5: Training adaptive confidence gate...")
        self._train_confidence_gate()

        print("   Model training complete!")

    def _initialize_rl_agents(self):
        state_size = self.X.shape[2] if hasattr(self, 'X') and self.X is not None else 86
        n_models = len(self.models) if self.models else 5
        market_features = min(20, state_size)

        self.multi_agent_system = ATHENAMultiAgentSystem(
            state_size=state_size,
            n_models=n_models,
            market_features=market_features,
        )

        self.confidence_gate = AdaptiveConfidenceGate(
            market_dim=market_features, hidden_size=64,
        ).to(self.device)

        self.rl_agents_initialized = True

    def _train_rl_agents_conservatively(self, episodes, eval_freq):
        if not self.rl_agents_initialized:
            raise ValueError("RL agents must be initialized first")

        for ep in range(episodes):
            self._run_rl_episode()

            if (ep + 1) % eval_freq == 0:
                perf = self._evaluate_hybrid_performance()
                baseline_sharpe = self.supervised_baseline_performance['sharpe_ratio']
                if perf < baseline_sharpe * self.performance_threshold:
                    self.performance_violations += 1
                    if self.performance_violations >= self.max_violations:
                        self.fallback_mode = True
                        break
                else:
                    self.performance_violations = max(0, self.performance_violations - 1)

    def _run_rl_episode(self):
        if not hasattr(self, 'X') or self.X is None:
            return 0.0

        total_reward = 0.0
        seq_len = min(100, len(self.X))

        for i in range(seq_len - 1):
            state = self.X[i, -1, :].cpu().numpy()
            model_preds = []
            for m in self.models:
                m.eval()
                with torch.no_grad():
                    model_preds.append(m(self.X[i:i + 1]).cpu().item())

            mf = state[-20:] if len(state) >= 20 else np.pad(state, (0, 20 - len(state)))
            action = self.multi_agent_system.act(state, model_preds, mf)

            next_ret = self.y[i + 1].cpu().item() if i + 1 < len(self.y) else 0.0
            reward = action * next_ret * 100

            next_state = self.X[i + 1, -1, :].cpu().numpy() if i + 1 < len(self.X) else state
            done = (i == seq_len - 2)

            self.multi_agent_system.update(
                state, action, reward, next_state, done, model_preds, mf
            )
            total_reward += reward

        return total_reward

    def _train_confidence_gate(self):
        """Train the adaptive confidence gate on recent signal quality."""
        if self.confidence_gate is None or not hasattr(self, 'X'):
            return

        gate_opt = optim.Adam(self.confidence_gate.parameters(), lr=0.001)
        market_dim = min(20, self.X.shape[2])

        for _ in range(30):
            self.confidence_gate.train()
            indices = np.random.permutation(min(200, len(self.X)))
            total_loss = 0.0

            for idx in indices:
                mf = self.X[idx, -1, :market_dim].unsqueeze(0)
                gate_val = self.confidence_gate(mf)

                # Simulate: supervised signal vs rl signal quality
                with torch.no_grad():
                    sup_preds = []
                    for m in self.models:
                        m.eval()
                        sup_preds.append(m(self.X[idx:idx + 1]).item())
                    sup_signal = np.mean(sup_preds)

                state = self.X[idx, -1, :].cpu().numpy()
                mf_np = state[-20:] if len(state) >= 20 else np.pad(state, (0, 20 - len(state)))
                rl_signal = self.multi_agent_system.act(state, sup_preds, mf_np)

                target_ret = self.y[idx].item()
                sup_pnl = sup_signal * target_ret
                rl_pnl = rl_signal * target_ret

                # Gate should be high when supervised is better
                optimal_gate = 1.0 if sup_pnl >= rl_pnl else 0.0
                target_t = torch.tensor([optimal_gate], device=self.device)

                loss = nn.MSELoss()(gate_val, target_t)
                gate_opt.zero_grad()
                loss.backward()
                gate_opt.step()
                total_loss += loss.item()

    def _evaluate_hybrid_performance(self):
        try:
            signals = self.generate_hybrid_rl_signals()
            if len(signals) > 100:
                recent = signals[-100:]
                rets = self.y[-100:].cpu().numpy() if hasattr(self, 'y') else np.random.randn(100) * 0.01
                pnl = recent * rets
                sharpe = np.mean(pnl) / (np.std(pnl) + 1e-8) * np.sqrt(252)
                self.rl_performance_history.append(sharpe)
                return sharpe
        except Exception:
            pass
        return 0.0

    # ------------------------------------------------------------------
    # Hybrid Signal Generation  (Mods 5, 6)
    # ------------------------------------------------------------------

    def generate_hybrid_rl_signals(self, start_idx=None, end_idx=None):
        if self.fallback_mode or not self.rl_agents_initialized:
            return self.generate_athena_signals(start_idx, end_idx)

        base_signals = self.generate_athena_signals(start_idx, end_idx)
        X_seq = self.X
        market_dim = min(20, X_seq.shape[2])
        enhanced = np.zeros(len(base_signals))

        for i in range(len(base_signals)):
            try:
                if i < len(X_seq):
                    state = X_seq[i, -1, :].cpu().numpy()
                    model_preds = []
                    for m in self.models:
                        m.eval()
                        with torch.no_grad():
                            st = torch.FloatTensor(X_seq[i:i + 1]).to(self.device)
                            model_preds.append(m(st).cpu().item())

                    mf = state[-20:] if len(state) >= 20 else np.pad(state, (0, 20 - len(state)))
                    rl_action = self.multi_agent_system.act(state, model_preds, mf)

                    # Adaptive gate (Mod 6) instead of fixed 0.7
                    if self.confidence_gate is not None:
                        self.confidence_gate.eval()
                        with torch.no_grad():
                            mf_t = torch.FloatTensor(mf).unsqueeze(0).to(self.device)
                            gate = self.confidence_gate(mf_t).item()
                    else:
                        gate = 0.7

                    enhanced[i] = gate * base_signals[i] + (1 - gate) * rl_action
                else:
                    enhanced[i] = base_signals[i]
            except Exception:
                enhanced[i] = base_signals[i]

        return np.clip(enhanced, -0.8, 0.8)

    # ------------------------------------------------------------------
    # Backtesting
    # ------------------------------------------------------------------

    def backtest_signals(self, signals):
        if hasattr(self, 'features_df') and 'Returns' in self.features_df.columns:
            returns = self.features_df['Returns'].dropna()
            min_len = min(len(signals), len(returns))
            signals = signals[-min_len:]
            returns = returns.iloc[-min_len:]

            pnl = signals * returns.values
            total_ret = (1 + pnl).prod() - 1
            annual_ret = (1 + total_ret) ** (252 / len(pnl)) - 1
            vol = pnl.std() * np.sqrt(252)
            sharpe = annual_ret / vol if vol > 0 else 0

            import pandas as pd
            cum = pd.Series(1 + pnl).cumprod()
            rmax = cum.expanding().max()
            dd = (cum - rmax) / rmax
            max_dd = dd.min()

            return {
                'total_return': total_ret,
                'annual_return': annual_ret,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'volatility': vol,
            }

        return {k: 0.0 for k in
                ['total_return', 'annual_return', 'sharpe_ratio', 'max_drawdown', 'volatility']}

    def get_performance_comparison(self):
        if not self.supervised_baseline_performance:
            return None
        return {
            'supervised_baseline': self.supervised_baseline_performance,
            'rl_violations': self.performance_violations,
            'rl_performance_history': self.rl_performance_history[-10:],
            'fallback_mode': self.fallback_mode,
        }
