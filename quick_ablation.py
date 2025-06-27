#!/usr/bin/env python3

from ablation_study import ARTEMISAblationStudy
import json

def run_quick_ablation():
    """Run quick ablation study and get real results"""
    
    print("🔬 Running Quick Ablation Study for AAPL...")
    study = ARTEMISAblationStudy('AAPL')
    
    print("\n1️⃣ Testing Single Model...")
    single_model = study._test_single_model()
    print(f"Single Model - AR: {single_model['annual_return']:.2%}, SR: {single_model['sharpe_ratio']:.3f}, DD: {single_model['max_drawdown']:.2%}")
    
    print("\n2️⃣ Testing Basic Ensemble...")
    basic_ensemble = study._test_basic_ensemble()
    print(f"Basic Ensemble - AR: {basic_ensemble['annual_return']:.2%}, SR: {basic_ensemble['sharpe_ratio']:.3f}, DD: {basic_ensemble['max_drawdown']:.2%}")
    
    print("\n3️⃣ Testing Diverse Ensemble...")
    diverse_ensemble = study._test_diverse_ensemble()
    print(f"Diverse Ensemble - AR: {diverse_ensemble['annual_return']:.2%}, SR: {diverse_ensemble['sharpe_ratio']:.3f}, DD: {diverse_ensemble['max_drawdown']:.2%}")
    
    # Compile results
    results = {
        'single_model': single_model,
        'basic_ensemble': basic_ensemble,
        'diverse_ensemble': diverse_ensemble
    }
    
    # Save to file
    with open('quick_ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to quick_ablation_results.json")
    return results

if __name__ == "__main__":
    run_quick_ablation() 