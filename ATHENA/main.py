#!/usr/bin/env python3
"""
ATHENA Trading System – Main Entry Point

Usage:
    python ATHENA/main.py GOOGL --mode supervised --epochs 100
    python ATHENA/main.py GOOGL --mode hybrid --episodes 50
"""

import sys
import os
import argparse
import json
from datetime import datetime

import torch
import numpy as np
import pandas as pd

# Allow imports from ATHENA package directory
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from athena_rl_system import HybridRLATHENASystem
from athena_core import ATHENAEnsembleSystem


class ATHENARunner:
    def __init__(self, initial_balance=100000, sequence_length=60):
        self.initial_balance = initial_balance
        self.sequence_length = sequence_length
        self.results_dir = "athena_results"
        os.makedirs(self.results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Supervised mode
    # ------------------------------------------------------------------

    def run_supervised(self, stock_symbol, n_models=5, epochs=250):
        print(f"[ATHENA] Running Supervised mode for {stock_symbol}")

        system = ATHENAEnsembleSystem(
            sequence_length=self.sequence_length,
            initial_balance=self.initial_balance,
            return_boost_factor=3.0,
            ultra_aggressive_mode=True,
        )

        system.load_and_prepare_data(stock_symbol=stock_symbol)
        system.create_enhanced_technical_indicators()
        system.prepare_sequences()

        system.train_athena_ensemble(n_models=n_models, epochs=epochs)
        signals = system.generate_athena_signals()
        results = system.backtest_signals(signals)

        return {'type': 'supervised', 'stock': stock_symbol,
                'results': results, 'system': system}

    # ------------------------------------------------------------------
    # Hybrid RL mode
    # ------------------------------------------------------------------

    def run_hybrid(self, stock_symbol, n_models=5, supervised_epochs=250,
                   rl_episodes=500, rl_learning_rate=1e-4):
        print(f"[ATHENA] Running Hybrid RL mode for {stock_symbol}")

        print("Phase 0: System initialization...")
        system = HybridRLATHENASystem(
            sequence_length=self.sequence_length,
            initial_balance=self.initial_balance,
            return_boost_factor=3.0,
            ultra_aggressive_mode=True,
            rl_learning_rate=rl_learning_rate,
        )

        print("Phase 1: Data preparation...")
        system.load_and_prepare_data(stock_symbol=stock_symbol)
        system.create_enhanced_technical_indicators()
        system.prepare_sequences()

        print("Phase 2: Model training...")
        system.train_hybrid_system(
            n_supervised_models=n_models,
            supervised_epochs=supervised_epochs,
            rl_episodes=rl_episodes,
            evaluation_frequency=50,
        )

        print("Phase 3: Signal generation...")
        signals = system.generate_hybrid_rl_signals()

        print("Phase 4: Performance evaluation...")
        results = system.backtest_signals(signals)

        return {
            'type': 'hybrid_rl', 'stock': stock_symbol,
            'results': results, 'system': system,
            'performance_comparison': system.get_performance_comparison(),
        }

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def display_results(self, data):
        results = data['results']
        stock = data['stock']
        mode = data['type']

        print(f"\n{'=' * 60}")
        print(f"  ATHENA {'RL-' if mode == 'hybrid_rl' else ''}PERFORMANCE RESULTS – {stock}")
        print(f"{'=' * 60}")

        print(f"  Total Return:        {results['total_return']:.2%}")
        print(f"  Annual Return:       {results['annual_return']:.2%}")
        print(f"  Sharpe Ratio:        {results['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown:        {results['max_drawdown']:.2%}")
        vol = results.get('volatility', None)
        if vol is None and 'portfolio_returns' in results:
            vol = float(np.std(results['portfolio_returns']) * np.sqrt(252))
        print(f"  Volatility:          {(vol or 0):.2%}")

        if mode == 'hybrid_rl':
            comp = data.get('performance_comparison')
            if comp and comp.get('supervised_baseline'):
                bl = comp['supervised_baseline']
                print(f"\n  Supervised Baseline Sharpe: {bl['sharpe_ratio']:.3f}")
                rl_delta = results['sharpe_ratio'] - bl['sharpe_ratio']
                print(f"  RL Enhancement:            {rl_delta:+.3f}")
                print(f"  Fallback Mode:             {comp.get('fallback_mode', False)}")

        print(f"\n  ATHENA System Highlights:")
        print(f"    - 5 Modified Neural Network Architectures")
        print(f"    - Wavelet-Enhanced Feature Engineering")
        print(f"    - Cross-Attention Fusion")
        print(f"    - Dual-Horizon Signal Generation")
        if mode == 'hybrid_rl':
            print(f"    - SAC Reinforcement Learning")
            print(f"    - Adaptive Confidence Gate")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_results(self, data):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = os.path.join(
            self.results_dir,
            f"athena_{data['stock']}_{data['type']}_{ts}.json"
        )
        out = {
            'stock': data['stock'],
            'type': data['type'],
            'timestamp': ts,
            'results': {
                k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v))
                for k, v in data['results'].items()
                if not isinstance(v, (np.ndarray, pd.Series, list))
            },
        }
        with open(fname, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"  Results saved to: {fname}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="ATHENA Trading System")
    parser.add_argument('stock', type=str, help='Stock symbol (e.g. GOOGL)')
    parser.add_argument('--mode', choices=['supervised', 'hybrid'], default='hybrid')
    parser.add_argument('--balance', type=int, default=100000)
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--models', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=250)
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{'=' * 60}")
    print(f"  ATHENA Trading System – {args.stock}")
    print(f"{'=' * 60}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    else:
        print(f"  Using CPU")

    runner = ATHENARunner(initial_balance=args.balance, sequence_length=60)

    try:
        if args.mode == 'supervised':
            data = runner.run_supervised(
                args.stock, n_models=args.models, epochs=args.epochs,
            )
        else:
            data = runner.run_hybrid(
                args.stock, n_models=args.models,
                supervised_epochs=args.epochs, rl_episodes=args.episodes,
                rl_learning_rate=args.lr,
            )

        runner.display_results(data)
        runner.save_results(data)

        print(f"\n  {args.mode.title()} analysis completed successfully!")

    except Exception as e:
        print(f"\n  Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
