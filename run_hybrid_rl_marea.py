#!/usr/bin/env python3
"""
Hybrid RL-MAREA Runner Script
Comprehensive script for running and comparing RL vs Supervised MAREA models

This script provides:
1. Easy switching between supervised and hybrid RL modes
2. Comprehensive performance comparison
3. Research-ready benchmarking
4. Multiple stock evaluation
5. Statistical significance testing

Usage:
    # Run hybrid RL-MAREA on Apple stock
    python run_hybrid_rl_marea.py AAPL --mode hybrid --episodes 500
    
    # Run supervised baseline for comparison
    python run_hybrid_rl_marea.py AAPL --mode supervised
    
    # Run comprehensive comparison study
    python run_hybrid_rl_marea.py AAPL --mode comparison --trials 5
"""

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

from rl_marea_hybrid_system import HybridRLMAREASystem
from marea_ensemble_system import MAREAEnsembleSystem
from tradingPerformance import PerformanceEstimator

class HybridRLMAREARunner:
    """Comprehensive runner for hybrid RL-MAREA experiments"""
    
    def __init__(self, initial_balance=100000, sequence_length=60):
        self.initial_balance = initial_balance
        self.sequence_length = sequence_length
        self.results_dir = "rl_marea_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def run_supervised_baseline(self, stock_symbol, n_models=5, epochs=250):
        """Run supervised MAREA baseline for comparison"""
        print(f"🎯 Running Supervised MAREA Baseline for {stock_symbol}")
        
        system = MAREAEnsembleSystem(
            sequence_length=self.sequence_length,
            initial_balance=self.initial_balance,
            return_boost_factor=1.25,
            ultra_aggressive_mode=True
        )
        
        # Load and prepare data
        system.load_and_prepare_data(stock_symbol=stock_symbol)
        system.create_enhanced_technical_indicators()
        system.prepare_sequences()
        
        # Train supervised ensemble
        system.train_marea_ultra_aggressive_ensemble(n_models=n_models, epochs=epochs)
        
        # Generate signals and backtest
        signals = system.generate_marea_ultra_aggressive_signals()
        results = system.backtest_signals(signals)
        
        return {
            'type': 'supervised',
            'stock': stock_symbol,
            'results': results,
            'system': system
        }
    
    def run_hybrid_rl_system(self, stock_symbol, n_models=5, supervised_epochs=250, 
                           rl_episodes=500, rl_learning_rate=1e-4):
        """Run hybrid RL-MAREA system"""
        print(f"🤖 Running Hybrid RL-MAREA System for {stock_symbol}")
        
        system = HybridRLMAREASystem(
            sequence_length=self.sequence_length,
            initial_balance=self.initial_balance,
            return_boost_factor=1.25,
            ultra_aggressive_mode=True,
            rl_learning_rate=rl_learning_rate
        )
        
        # Load and prepare data
        system.load_and_prepare_data(stock_symbol=stock_symbol)
        system.create_enhanced_technical_indicators()
        system.prepare_sequences()
        
        # Train hybrid system
        system.train_hybrid_system(
            n_supervised_models=n_models,
            supervised_epochs=supervised_epochs,
            rl_episodes=rl_episodes,
            evaluation_frequency=50
        )
        
        # Generate signals and backtest
        signals = system.generate_hybrid_rl_signals()
        results = system.backtest_signals(signals)
        
        return {
            'type': 'hybrid_rl',
            'stock': stock_symbol,
            'results': results,
            'system': system,
            'performance_comparison': system.get_performance_comparison()
        }
    
    def run_comprehensive_comparison(self, stock_symbol, trials=3, rl_episodes=300):
        """Run comprehensive comparison between supervised and RL approaches"""
        print(f"📊 Running Comprehensive Comparison Study for {stock_symbol}")
        print(f"   Trials: {trials}, RL Episodes per trial: {rl_episodes}")
        
        supervised_results = []
        hybrid_results = []
        
        for trial in range(trials):
            print(f"\n   Trial {trial + 1}/{trials}")
            
            # Supervised baseline
            supervised = self.run_supervised_baseline(
                stock_symbol, 
                epochs=200  # Reduced for comparison study
            )
            supervised_results.append(supervised['results'])
            
            # Hybrid RL system
            hybrid = self.run_hybrid_rl_system(
                stock_symbol,
                supervised_epochs=200,
                rl_episodes=rl_episodes
            )
            hybrid_results.append(hybrid['results'])
        
        # Statistical analysis
        comparison_stats = self._analyze_comparison_statistics(
            supervised_results, hybrid_results
        )
        
        # Save comprehensive results
        self._save_comparison_results(stock_symbol, comparison_stats, trials)
        
        return comparison_stats
    
    def _analyze_comparison_statistics(self, supervised_results, hybrid_results):
        """Perform statistical analysis of comparison results"""
        
        # Extract metrics
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
        
        # Statistical tests
        stats_results = {}
        for metric in supervised_metrics.keys():
            supervised_vals = supervised_metrics[metric]
            hybrid_vals = hybrid_metrics[metric]
            
            # t-test for mean difference
            t_stat, p_value = stats.ttest_ind(hybrid_vals, supervised_vals)
            
            # Effect size (Cohen's d)
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
        """Save comparison results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON results
        json_file = f"{self.results_dir}/comparison_{stock_symbol}_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(stats_results, f, indent=2, default=str)
        
        # CSV summary
        csv_data = []
        for metric, stats in stats_results.items():
            csv_data.append({
                'metric': metric,
                'supervised_mean': stats['supervised_mean'],
                'hybrid_mean': stats['hybrid_mean'],
                'improvement': stats['improvement'],
                'improvement_pct': stats['improvement_pct'],
                'p_value': stats['p_value'],
                'significant': stats['significant'],
                'cohens_d': stats['cohens_d']
            })
        
        csv_file = f"{self.results_dir}/summary_{stock_symbol}_{timestamp}.csv"
        pd.DataFrame(csv_data).to_csv(csv_file, index=False)
        
        print(f"   💾 Results saved to {json_file} and {csv_file}")
    
    def display_results(self, results_data):
        """Display comprehensive results in a formatted table"""
        if results_data['type'] == 'supervised':
            self._display_supervised_results(results_data)
        elif results_data['type'] == 'hybrid_rl':
            self._display_hybrid_results(results_data)
    
    def _display_supervised_results(self, data):
        """Display supervised MAREA results in original MAREA format"""
        results = data['results']
        stock = data['stock']
        
        # Use the exact same format as original MAREA system
        print(f"\n{'='*50}")
        print(f"🏆 SUPERVISED MAREA PERFORMANCE RESULTS - {stock}")
        print(f"{'='*50}")
        
        print(f"📊 Portfolio Metrics:")
        print(f"   Total Return:        {results['total_return']:.2%}")
        print(f"   Cumulative Return:   {results['total_return']:.2%}")
        print(f"   Annual Return:       {results['annual_return']:.2%}")
        print(f"   Sharpe Ratio:        {results['sharpe_ratio']:.3f}")
        print(f"   Max Drawdown:        {results['max_drawdown']:.2%}")
        print(f"   Win Rate:            {results['win_rate']:.1%}")
        print(f"   Total Trades:        {results['total_trades']:,}")
        
        # Time period information
        print(f"\n📅 Trading Period:")
        print(f"   Start Date:          {results['start_date'].strftime('%Y-%m-%d')}")
        print(f"   End Date:            {results['end_date'].strftime('%Y-%m-%d')}")
        print(f"   Trading Days:        {results['trading_days']:,}")
        print(f"   Years:               {results['years']:,.2f}")
        
        # Daily metrics
        daily_returns = results['portfolio_returns']
        print(f"\n📈 Daily Performance:")
        print(f"   Avg Daily Return:    {daily_returns.mean():.4f}")
        print(f"   Daily Volatility:    {daily_returns.std():.4f}")
        print(f"   Best Day:            {daily_returns.max():.4f}")
        print(f"   Worst Day:           {daily_returns.min():.4f}")
        
        # Compare to buy-and-hold
        print(f"\n🏪 Benchmark Comparison:")
        print(f"   Buy & Hold Total:    {results['buy_hold_total_return']:.2%}")
        print(f"   Buy & Hold Annual:   {results['buy_hold_annual_return']:.2%}")
        print(f"   Strategy vs B&H:     {results['total_return'] - results['buy_hold_total_return']:.2%}")
        print(f"   Annual Alpha:        {results['annual_return'] - results['buy_hold_annual_return']:.2%}")
        
        print(f"\n🎯 MAREA Supervised System Features:")
        print(f"   🧠 5 Diverse Neural Network Architectures")
        print(f"   🔄 Regime-Aware Adaptive Weighting") 
        print(f"   📊 Dynamic Position Sizing")
        print(f"   🚀 Ultra-Aggressive Return Optimization")
    
    def _display_hybrid_results(self, data):
        """Display hybrid RL-MAREA results in the same format as original MAREA"""
        results = data['results']
        stock = data['stock']
        comparison = data.get('performance_comparison')
        
        # Use the exact same format as original MAREA system
        print(f"\n{'='*50}")
        print(f"🏆 HYBRID RL-MAREA PERFORMANCE RESULTS - {stock}")
        print(f"{'='*50}")
        
        print(f"📊 Portfolio Metrics:")
        print(f"   Total Return:        {results['total_return']:.2%}")
        print(f"   Cumulative Return:   {results['total_return']:.2%}")
        print(f"   Annual Return:       {results['annual_return']:.2%}")
        print(f"   Sharpe Ratio:        {results['sharpe_ratio']:.3f}")
        print(f"   Max Drawdown:        {results['max_drawdown']:.2%}")
        print(f"   Win Rate:            {results['win_rate']:.1%}")
        print(f"   Total Trades:        {results['total_trades']:,}")
        
        # Time period information
        print(f"\n📅 Trading Period:")
        print(f"   Start Date:          {results['start_date'].strftime('%Y-%m-%d')}")
        print(f"   End Date:            {results['end_date'].strftime('%Y-%m-%d')}")
        print(f"   Trading Days:        {results['trading_days']:,}")
        print(f"   Years:               {results['years']:,.2f}")
        
        # Daily metrics
        daily_returns = results['portfolio_returns']
        print(f"\n📈 Daily Performance:")
        print(f"   Avg Daily Return:    {daily_returns.mean():.4f}")
        print(f"   Daily Volatility:    {daily_returns.std():.4f}")
        print(f"   Best Day:            {daily_returns.max():.4f}")
        print(f"   Worst Day:           {daily_returns.min():.4f}")
        
        # Compare to buy-and-hold
        print(f"\n🏪 Benchmark Comparison:")
        print(f"   Buy & Hold Total:    {results['buy_hold_total_return']:.2%}")
        print(f"   Buy & Hold Annual:   {results['buy_hold_annual_return']:.2%}")
        print(f"   Strategy vs B&H:     {results['total_return'] - results['buy_hold_total_return']:.2%}")
        print(f"   Annual Alpha:        {results['annual_return'] - results['buy_hold_annual_return']:.2%}")
        
        # Hybrid RL enhancement details
        if comparison and 'supervised_baseline' in comparison:
            baseline = comparison['supervised_baseline']
            
            print(f"\n🤖 HYBRID RL ENHANCEMENT:")
            print(f"   Supervised Annual:   {baseline['annual_return']:.2%}")
            print(f"   Hybrid RL Annual:    {results['annual_return']:.2%}")
            print(f"   RL Enhancement:      {results['annual_return'] - baseline['annual_return']:.2%}")
            print(f"   Supervised Sharpe:   {baseline['sharpe_ratio']:.3f}")
            print(f"   Hybrid RL Sharpe:    {results['sharpe_ratio']:.3f}")
            print(f"   Sharpe Enhancement:  {results['sharpe_ratio'] - baseline['sharpe_ratio']:.3f}")
        
        print(f"\n🎯 HYBRID RL-MAREA System Features:")
        print(f"   🧠 5 Diverse Neural Network Architectures")
        print(f"   🤖 Advanced TD3 Reinforcement Learning")
        print(f"   🔄 Conservative RL Integration (Performance Preservation)")
        print(f"   📊 Dynamic Position Sizing & Risk Management")
        print(f"   🚀 GPU-Accelerated Training & Inference")
    
    def display_comparison_statistics(self, stats_results, stock_symbol):
        """Display detailed statistical comparison results"""
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
        
        # Overall assessment
        significant_improvements = sum(1 for stats in stats_results.values() if stats['significant'] and stats['improvement'] > 0)
        total_metrics = len(stats_results)
        
        print(f"\n🏆 OVERALL ASSESSMENT")
        print(f"   Significant improvements: {significant_improvements}/{total_metrics} metrics")
        
        if significant_improvements >= total_metrics * 0.5:
            print(f"   ✅ RL enhancement shows meaningful improvement")
        else:
            print(f"   ⚠️  RL enhancement shows mixed results")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Hybrid RL-MAREA Trading System Runner")
    
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
    """Main execution function"""
    args = parse_arguments()
    
    print(f"🚀 HYBRID RL-MAREA SYSTEM")
    print(f"   Stock: {args.stock}")
    print(f"   Mode: {args.mode}")
    print(f"   Initial Balance: ${args.balance:,}")
    
    # GPU Setup and Optimization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   🚀 Device: {device}")
    
    if device.type == 'cuda':
        print(f"   💎 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   🔋 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # Clear GPU cache before starting
        torch.cuda.empty_cache()
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        print(f"   ⚡ GPU optimizations enabled")
    else:
        print(f"   ⚠️  GPU not available, using CPU")
        # Optimize for CPU
        torch.set_num_threads(min(8, torch.get_num_threads()))
        print(f"   🔧 CPU optimizations enabled")
    
    runner = HybridRLMAREARunner(
        initial_balance=args.balance,
        sequence_length=60
    )
    
    try:
        if args.mode == 'supervised':
            # Run supervised baseline only
            results = runner.run_supervised_baseline(
                args.stock, 
                n_models=args.models, 
                epochs=args.epochs
            )
            runner.display_results(results)
            
        elif args.mode == 'hybrid':
            # Run hybrid RL system
            results = runner.run_hybrid_rl_system(
                args.stock,
                n_models=args.models,
                supervised_epochs=args.epochs,
                rl_episodes=args.episodes,
                rl_learning_rate=args.lr
            )
            runner.display_results(results)
            
        elif args.mode == 'comparison':
            # Run comprehensive comparison study
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