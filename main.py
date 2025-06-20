#!/usr/bin/env python3

import sys
import argparse
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import json
import os
from scipy import stats
from tabulate import tabulate
import matplotlib.pyplot as plt

from artemis_rl_system import HybridRLARTEMISSystem
from artemis_core import ARTEMISEnsembleSystem
from artemis_utils import PerformanceEstimator

class HybridRLARTEMISRunner:
    def __init__(self, initial_balance=100000, sequence_length=60):
        self.initial_balance = initial_balance
        self.sequence_length = sequence_length
        self.results_dir = "rl_artemis_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def run_supervised_baseline(self, stock_symbol, n_models=5, epochs=250):
        print(f"🎯 Running Supervised ARTEMIS Baseline for {stock_symbol}")
        
        system = ARTEMISEnsembleSystem(
            sequence_length=self.sequence_length,
            initial_balance=self.initial_balance,
            return_boost_factor=3.0,  # 🌟 LEGENDARY boost factor
            ultra_aggressive_mode=True
        )
        
        system.load_and_prepare_data(stock_symbol=stock_symbol)
        system.create_enhanced_technical_indicators()
        system.prepare_sequences()
        
        system.train_artemis_ultra_aggressive_ensemble(n_models=n_models, epochs=epochs)
        
        signals = system.generate_artemis_ultra_aggressive_signals()
        results = system.backtest_signals(signals)
        
        return {
            'type': 'supervised',
            'stock': stock_symbol,
            'results': results,
            'system': system
        }
    
    def run_hybrid_rl_system(self, stock_symbol, n_models=5, supervised_epochs=250, 
                           rl_episodes=500, rl_learning_rate=1e-4):
        print(f"🤖 Running Hybrid RL-ARTEMIS for {stock_symbol}")
        
        print("🔧 Phase 0: System initialization...")
        system = HybridRLARTEMISSystem(
            sequence_length=self.sequence_length,
            initial_balance=self.initial_balance,
            return_boost_factor=3.0,  # 🌟 LEGENDARY boost factor
            ultra_aggressive_mode=True,
            rl_learning_rate=rl_learning_rate
        )
        
        print("📊 Phase 1: Data preparation...")
        system.load_and_prepare_data(stock_symbol=stock_symbol)
        system.create_enhanced_technical_indicators()
        system.prepare_sequences()
        
        print("🧠 Phase 2: Model training...")
        system.train_hybrid_system(
            n_supervised_models=n_models,
            supervised_epochs=supervised_epochs,
            rl_episodes=rl_episodes,
            evaluation_frequency=50
        )
        
        print("🎯 Phase 3: Signal generation...")
        signals = system.generate_hybrid_rl_signals()
        
        print("📈 Phase 4: Performance evaluation...")
        results = system.backtest_signals(signals)
        
        return {
            'type': 'hybrid_rl',
            'stock': stock_symbol,
            'results': results,
            'system': system,
            'performance_comparison': system.get_performance_comparison()
        }
    
    def run_comprehensive_comparison(self, stock_symbol, trials=3, rl_episodes=300):
        print(f"📊 Running Comprehensive Comparison Study for {stock_symbol}")
        print(f"   Trials: {trials}, RL Episodes per trial: {rl_episodes}")
        
        supervised_results = []
        hybrid_results = []
        
        for trial in range(trials):
            print(f"\n   Trial {trial + 1}/{trials}")
            
            supervised = self.run_supervised_baseline(
                stock_symbol, 
                epochs=200
            )
            supervised_results.append(supervised['results'])
            
            hybrid = self.run_hybrid_rl_system(
                stock_symbol,
                supervised_epochs=200,
                rl_episodes=rl_episodes
            )
            hybrid_results.append(hybrid['results'])
        
        comparison_stats = self._analyze_comparison_statistics(
            supervised_results, hybrid_results
        )
        
        self._save_comparison_results(stock_symbol, comparison_stats, trials)
        
        return comparison_stats
    
    def _analyze_comparison_statistics(self, supervised_results, hybrid_results):
        supervised_metrics = {
            'annual_returns': [r['annual_return'] for r in supervised_results],
            'sharpe_ratios': [r['sharpe_ratio'] for r in supervised_results],
            'max_drawdowns': [r['max_drawdown'] for r in supervised_results],
            'win_rates': [r['win_rate'] for r in supervised_results]
        }
        
        hybrid_metrics = {
            'annual_returns': [r['annual_return'] for r in hybrid_results],
            'sharpe_ratios': [r['sharpe_ratio'] for r in hybrid_results],
            'max_drawdowns': [r['max_drawdown'] for r in hybrid_results],
            'win_rates': [r['win_rate'] for r in hybrid_results]
        }
        
        stats_results = {}
        for metric in supervised_metrics.keys():
            supervised_vals = supervised_metrics[metric]
            hybrid_vals = hybrid_metrics[metric]
            
            t_stat, p_value = stats.ttest_ind(hybrid_vals, supervised_vals)
            
            pooled_std = np.sqrt(((np.std(supervised_vals)**2 + np.std(hybrid_vals)**2) / 2))
            cohens_d = (np.mean(hybrid_vals) - np.mean(supervised_vals)) / pooled_std if pooled_std > 0 else 0
            
            stats_results[metric] = {
                'supervised_mean': np.mean(supervised_vals),
                'supervised_std': np.std(supervised_vals),
                'hybrid_mean': np.mean(hybrid_vals),
                'hybrid_std': np.std(hybrid_vals),
                'improvement': np.mean(hybrid_vals) - np.mean(supervised_vals),
                'improvement_pct': ((np.mean(hybrid_vals) - np.mean(supervised_vals)) / abs(np.mean(supervised_vals))) * 100 if np.mean(supervised_vals) != 0 else 0,
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significant': p_value < 0.05
            }
        
        return stats_results
    
    def _save_comparison_results(self, stock_symbol, stats_results, trials):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.results_dir, f"comparison_{stock_symbol}_{trials}trials_{timestamp}.json")
        
        results_data = {
            'stock': stock_symbol,
            'trials': trials,
            'timestamp': timestamp,
            'statistics': stats_results
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        print(f"   💾 Results saved to: {filename}")
    
    def display_results(self, results_data):
        if results_data['type'] == 'supervised':
            self._display_supervised_results(results_data)
        elif results_data['type'] == 'hybrid_rl':
            self._display_hybrid_results(results_data)
    
    def _display_supervised_results(self, data):
        results = data['results']
        stock = data['stock']
        
        print(f"\n{'='*50}")
        print(f"🏆 ARTEMIS ULTRA-AGGRESSIVE PERFORMANCE RESULTS - {stock}")
        print(f"{'='*50}")
        
        print(f"📊 Portfolio Metrics:")
        print(f"   Total Return:        {results['total_return']:.2%}")
        print(f"   Cumulative Return:   {results['total_return']:.2%}")
        print(f"   Annual Return:       {results['annual_return']:.2%}")
        print(f"   Sharpe Ratio:        {results['sharpe_ratio']:.3f}")
        print(f"   Max Drawdown:        {results['max_drawdown']:.2%}")
        print(f"   Volatility:          {results.get('volatility', 0):.2%}")
        
        # Skip missing fields that aren't calculated in the backtest
        if 'win_rate' in results:
            print(f"   Win Rate:            {results['win_rate']:.1%}")
        if 'total_trades' in results:
            print(f"   Total Trades:        {results['total_trades']:,}")
        
        # Skip date fields if not available
        if 'start_date' in results and 'end_date' in results:
            print(f"\n📅 Trading Period:")
            print(f"   Start Date:          {results['start_date'].strftime('%Y-%m-%d')}")
            print(f"   End Date:            {results['end_date'].strftime('%Y-%m-%d')}")
            if 'trading_days' in results:
                print(f"   Trading Days:        {results['trading_days']:,}")
            if 'years' in results:
                print(f"   Years:               {results['years']:,.2f}")
        
        # Skip portfolio returns analysis if not available
        if 'portfolio_returns' in results:
            daily_returns = results['portfolio_returns']
            print(f"\n📈 Daily Performance:")
            print(f"   Avg Daily Return:    {daily_returns.mean():.4f}")
            print(f"   Daily Volatility:    {daily_returns.std():.4f}")
            print(f"   Best Day:            {daily_returns.max():.4f}")
            print(f"   Worst Day:           {daily_returns.min():.4f}")
        
        # Skip benchmark comparison if not available
        if 'buy_hold_total_return' in results and 'buy_hold_annual_return' in results:
            print(f"\n🏪 Benchmark Comparison:")
            print(f"   Buy & Hold Total:    {results['buy_hold_total_return']:.2%}")
            print(f"   Buy & Hold Annual:   {results['buy_hold_annual_return']:.2%}")
            print(f"   Strategy vs B&H:     {results['total_return'] - results['buy_hold_total_return']:.2%}")
            print(f"   Annual Alpha:        {results['annual_return'] - results['buy_hold_annual_return']:.2%}")
        
        print(f"\n🎯 ARTEMIS Supervised System Features:")
        print(f"   🧠 5 Diverse Neural Network Architectures")
        print(f"   🔄 Regime-Aware Adaptive Weighting") 
        print(f"   📊 Dynamic Position Sizing")
        print(f"   🚀 Ultra-Aggressive Return Optimization")
    
    def _display_hybrid_results(self, data):
        results = data['results']
        stock = data['stock']
        comparison = data.get('performance_comparison')
        
        print(f"\n{'='*50}")
        print(f"🏆 RL-ARTEMIS PERFORMANCE RESULTS - {stock}")
        print(f"{'='*50}")
        
        print(f"🚀 RL-ARTEMIS Portfolio Performance:")
        print(f"   Total Return:        {results['total_return']:.2%}")
        print(f"   Annual Return:       {results['annual_return']:.2%}")
        print(f"   Sharpe Ratio:        {results['sharpe_ratio']:.3f}")
        print(f"   Max Drawdown:        {results['max_drawdown']:.2%}")
        print(f"   Volatility:          {results.get('volatility', 0):.2%}")
        
        # Additional performance metrics
        if 'win_rate' in results:
            print(f"   Win Rate:            {results['win_rate']:.1%}")
        if 'total_trades' in results:
            print(f"   Total Trades:        {results['total_trades']:,}")
        
        # Skip date fields if not available
        if 'start_date' in results and 'end_date' in results:
            print(f"\n📅 Trading Period:")
            print(f"   Start Date:          {results['start_date'].strftime('%Y-%m-%d')}")
            print(f"   End Date:            {results['end_date'].strftime('%Y-%m-%d')}")
            if 'trading_days' in results:
                print(f"   Trading Days:        {results['trading_days']:,}")
            if 'years' in results:
                print(f"   Years:               {results['years']:,.2f}")
        
        # Skip portfolio returns analysis if not available
        if 'portfolio_returns' in results:
            daily_returns = results['portfolio_returns']
            print(f"\n📈 Daily Performance:")
            print(f"   Avg Daily Return:    {daily_returns.mean():.4f}")
            print(f"   Daily Volatility:    {daily_returns.std():.4f}")
            print(f"   Best Day:            {daily_returns.max():.4f}")
            print(f"   Worst Day:           {daily_returns.min():.4f}")
        
        # Skip benchmark comparison if not available
        if 'buy_hold_total_return' in results and 'buy_hold_annual_return' in results:
            print(f"\n🏪 Benchmark Comparison:")
            print(f"   Buy & Hold Total:    {results['buy_hold_total_return']:.2%}")
            print(f"   Buy & Hold Annual:   {results['buy_hold_annual_return']:.2%}")
            print(f"   Strategy vs B&H:     {results['total_return'] - results['buy_hold_total_return']:.2%}")
            print(f"   Annual Alpha:        {results['annual_return'] - results['buy_hold_annual_return']:.2%}")
        
        print(f"\n🎯 RL-ARTEMIS System Features:")
        print(f"   🧠 5 Diverse Neural Network Architectures")
        print(f"   🤖 Advanced TD3 Reinforcement Learning")
        print(f"   🔄 Conservative RL Integration (Performance Preservation)")
        print(f"   📊 Dynamic Position Sizing & Risk Management")
        print(f"   🚀 GPU-Accelerated Training & Inference")
    
    def display_comparison_statistics(self, stats_results, stock_symbol):
        print(f"\n{'='*80}")
        print(f"📊 STATISTICAL COMPARISON ANALYSIS - {stock_symbol}")
        print(f"{'='*80}")
        
        for metric, stats in stats_results.items():
            print(f"\n📈 {metric.replace('_', ' ').title()}")
            print(f"   Supervised:     {stats['supervised_mean']:.4f} ± {stats['supervised_std']:.4f}")
            print(f"   Hybrid RL:      {stats['hybrid_mean']:.4f} ± {stats['hybrid_std']:.4f}")
            print(f"   Improvement:    {stats['improvement']:+.4f} ({stats['improvement_pct']:+.2f}%)")
            print(f"   Significance:   p = {stats['p_value']:.4f} {'✓' if stats['significant'] else '✗'}")
            print(f"   Effect Size:    Cohen's d = {stats['cohens_d']:.3f}")
        
        significant_improvements = sum(1 for stats in stats_results.values() if stats['significant'] and stats['improvement'] > 0)
        total_metrics = len(stats_results)
        
        print(f"\n🏆 OVERALL ASSESSMENT")
        print(f"   Significant improvements: {significant_improvements}/{total_metrics} metrics")
        
        if significant_improvements >= total_metrics * 0.5:
            print(f"   ✅ RL enhancement shows meaningful improvement")
        else:
            print(f"   ⚠️  RL enhancement shows mixed results")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Hybrid RL-ARTEMIS Trading System Runner")
    
    parser.add_argument('stock', type=str, help='Stock symbol to trade (e.g., AAPL, GOOGL)')
    parser.add_argument('--mode', type=str, choices=['supervised', 'hybrid', 'comparison'], 
                       default='hybrid', help='Running mode')
    parser.add_argument('--balance', type=int, default=100000, help='Initial balance')
    parser.add_argument('--episodes', type=int, default=500, help='RL training episodes')
    parser.add_argument('--trials', type=int, default=3, help='Number of trials for comparison')
    parser.add_argument('--lr', type=float, default=1e-4, help='RL learning rate')
    parser.add_argument('--models', type=int, default=5, help='Number of ensemble models')
    parser.add_argument('--epochs', type=int, default=250, help='Supervised training epochs')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    print(f"🚀 ARTEMIS Trading System - {args.stock}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"   🚀 GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    else:
        print(f"   ⚠️  Using CPU")
    
    runner = HybridRLARTEMISRunner(
        initial_balance=args.balance,
        sequence_length=60
    )
    
    try:
        if args.mode == 'supervised':
            results = runner.run_supervised_baseline(
                args.stock, 
                n_models=args.models, 
                epochs=args.epochs
            )
            runner.display_results(results)
            
        elif args.mode == 'hybrid':
            results = runner.run_hybrid_rl_system(
                args.stock,
                n_models=args.models,
                supervised_epochs=args.epochs,
                rl_episodes=args.episodes,
                rl_learning_rate=args.lr
            )
            runner.display_results(results)
            
        elif args.mode == 'comparison':
            stats_results = runner.run_comprehensive_comparison(
                args.stock,
                trials=args.trials,
                rl_episodes=args.episodes
            )
            runner.display_comparison_statistics(stats_results, args.stock)
        
        print(f"\n✅ {args.mode.title()} analysis completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 