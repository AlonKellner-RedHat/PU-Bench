#!/usr/bin/env python3
"""Compare 4 methods vs baseline with all metrics

Baseline:
- vpu_nomixup

Methods to compare:
1. vpu (classic VPU with mixup)
2. vpu_nomixup_mean_prior (auto)
3. vpu_nomixup_mean_prior (0.5)
4. vpu_mean_prior (0.5) with mixup
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats


def load_method_results(results_dir="results_cartesian", method_name=None, method_prior_filter=None):
    """Load results for a specific method configuration"""
    results = []

    results_path = Path(results_dir)
    json_files = list(results_path.glob("seed_*/*.json"))

    for json_file in json_files:
        try:
            with open(json_file) as f:
                data = json.load(f)

            # Check if this file contains the target method
            if method_name not in data.get('runs', {}):
                continue

            method_data = data['runs'][method_name]
            hyperparams = method_data.get('hyperparameters', {})
            dataset_info = method_data.get('dataset', {})
            best_metrics = method_data.get('best', {}).get('metrics', {})

            # Filter by method_prior if specified
            if method_prior_filter is not None:
                actual_method_prior = hyperparams.get('method_prior')
                if method_prior_filter == "auto" and actual_method_prior is not None:
                    continue
                elif method_prior_filter != "auto" and actual_method_prior != method_prior_filter:
                    continue

            dataset = hyperparams.get('dataset_class')
            seed = hyperparams.get('seed')
            c = hyperparams.get('labeled_ratio')
            true_prior_actual = dataset_info.get('train', {}).get('prior')

            if all([dataset, seed, c, true_prior_actual]):
                results.append({
                    'method': method_name,
                    'dataset': dataset,
                    'seed': seed,
                    'c': c,
                    'true_prior': true_prior_actual,
                    # All metrics
                    'test_ap': best_metrics.get('test_ap'),
                    'test_f1': best_metrics.get('test_f1'),
                    'test_max_f1': best_metrics.get('test_max_f1'),
                    'test_auc': best_metrics.get('test_auc'),
                    'test_accuracy': best_metrics.get('test_accuracy'),
                    'test_error': best_metrics.get('test_error'),
                    'test_precision': best_metrics.get('test_precision'),
                    'test_recall': best_metrics.get('test_recall'),
                    'test_ece': best_metrics.get('test_ece'),
                    'test_mce': best_metrics.get('test_mce'),
                    'test_brier': best_metrics.get('test_brier'),
                    'test_anice': best_metrics.get('test_anice'),
                    'test_snice': best_metrics.get('test_snice'),
                    'test_oracle_ce': best_metrics.get('test_oracle_ce'),
                    'test_risk': best_metrics.get('test_risk'),
                    'convergence_epoch': method_data.get('best', {}).get('epoch'),
                })

        except Exception as e:
            continue

    return pd.DataFrame(results)


def create_comparison_table(output_dir='results_cartesian'):
    """Create detailed comparison table"""

    print("="*120)
    print("4-Way Comparison vs Baseline (Including Mixup+Prior=0.5)")
    print("="*120)
    print()

    # Load all methods
    print("Loading results...")
    df_baseline = load_method_results('results_cartesian', 'vpu_nomixup')
    df_baseline['method_label'] = 'vpu_nomixup (baseline)'
    print(f"  vpu_nomixup (baseline): {len(df_baseline)} experiments")

    df_vpu = load_method_results('results_cartesian', 'vpu')
    df_vpu['method_label'] = 'vpu (classic with mixup)'
    print(f"  vpu (classic with mixup): {len(df_vpu)} experiments")

    df_auto = load_method_results('results_cartesian', 'vpu_nomixup_mean_prior', method_prior_filter="auto")
    df_auto['method_label'] = 'vpu_nomixup_mean_prior (auto)'
    print(f"  vpu_nomixup_mean_prior (auto): {len(df_auto)} experiments")

    df_05_nomixup = load_method_results('results_cartesian', 'vpu_nomixup_mean_prior', method_prior_filter=0.5)
    df_05_nomixup['method_label'] = 'vpu_nomixup_mean_prior (0.5)'
    print(f"  vpu_nomixup_mean_prior (0.5): {len(df_05_nomixup)} experiments")

    df_05_mixup = load_method_results('results_cartesian', 'vpu_mean_prior', method_prior_filter=0.5)
    df_05_mixup['method_label'] = 'vpu_mean_prior (0.5) WITH MIXUP'
    print(f"  vpu_mean_prior (0.5) with mixup: {len(df_05_mixup)} experiments")

    # Combine
    df_all = pd.concat([df_baseline, df_vpu, df_auto, df_05_nomixup, df_05_mixup], ignore_index=True)
    print(f"\nTotal: {len(df_all)} experiments")
    print()

    # Calculate mean and std for each metric
    metrics = [
        ('test_ap', 'AP', 'higher'),
        ('test_f1', 'F1', 'higher'),
        ('test_max_f1', 'Max F1', 'higher'),
        ('test_auc', 'AUC', 'higher'),
        ('test_accuracy', 'Accuracy', 'higher'),
        ('test_precision', 'Precision', 'higher'),
        ('test_recall', 'Recall', 'higher'),
        ('test_ece', 'ECE', 'lower'),
        ('test_mce', 'MCE', 'lower'),
        ('test_brier', 'Brier', 'lower'),
        ('test_anice', 'A-NICE', 'lower'),
        ('test_snice', 'S-NICE', 'lower'),
        ('test_oracle_ce', 'Oracle CE', 'lower'),
        ('convergence_epoch', 'Convergence', 'lower'),
    ]

    # Get baseline values
    baseline_stats = {}
    for metric, label, direction in metrics:
        baseline_values = df_baseline[metric].dropna()
        if len(baseline_values) > 0:
            baseline_stats[metric] = {
                'mean': baseline_values.mean(),
                'std': baseline_values.std(),
            }

    # Create comparison table
    methods_to_compare = [
        ('vpu_nomixup (baseline)', df_baseline),
        ('vpu (classic with mixup)', df_vpu),
        ('vpu_nomixup_mean_prior (auto)', df_auto),
        ('vpu_nomixup_mean_prior (0.5)', df_05_nomixup),
        ('vpu_mean_prior (0.5) WITH MIXUP', df_05_mixup),
    ]

    print("="*120)
    print("COMPLETE COMPARISON TABLE - All Metrics")
    print("="*120)
    print()

    # Create comprehensive table
    for metric, label, direction in metrics:
        print(f"\n{label} ({'higher is better' if direction == 'higher' else 'lower is better'}):")
        print("-" * 120)

        for method_label, df_method in methods_to_compare:
            values = df_method[metric].dropna()

            if len(values) == 0:
                print(f"  {method_label:45s}: N/A")
                continue

            mean_val = values.mean()
            std_val = values.std()

            if method_label == 'vpu_nomixup (baseline)':
                print(f"  {method_label:45s}: {mean_val:.4f} ± {std_val:.4f}")
            else:
                if metric in baseline_stats:
                    baseline_mean = baseline_stats[metric]['mean']
                    diff = mean_val - baseline_mean

                    if direction == 'lower':
                        pct_change = (baseline_mean - mean_val) / baseline_mean * 100 if baseline_mean != 0 else 0
                    else:
                        pct_change = (mean_val - baseline_mean) / baseline_mean * 100 if baseline_mean != 0 else 0

                    sign = '+' if pct_change >= 0 else ''
                    print(f"  {method_label:45s}: {mean_val:.4f} ± {std_val:.4f}  ({sign}{pct_change:6.2f}%)")

    # Find overall winner
    print("\n" + "="*120)
    print("WINNERS BY METRIC")
    print("="*120)
    print()

    for metric, label, direction in metrics:
        best_method = None
        best_value = None

        for method_label, df_method in methods_to_compare:
            values = df_method[metric].dropna()
            if len(values) == 0:
                continue

            mean_val = values.mean()

            if best_value is None:
                best_method = method_label
                best_value = mean_val
            else:
                if direction == 'higher' and mean_val > best_value:
                    best_method = method_label
                    best_value = mean_val
                elif direction == 'lower' and mean_val < best_value:
                    best_method = method_label
                    best_value = mean_val

        if best_method:
            print(f"  {label:15s}: {best_method:45s} ({best_value:.4f})")

    # Statistical tests
    print("\n" + "="*120)
    print("STATISTICAL SIGNIFICANCE (vs Baseline)")
    print("="*120)
    print()

    baseline_values = {}
    for metric, label, direction in metrics:
        baseline_values[label] = df_baseline[metric].dropna()

    for method_label, df_method in methods_to_compare:
        if method_label == 'vpu_nomixup (baseline)':
            continue

        print(f"{method_label}:")
        for metric, label, direction in metrics[:7]:  # Test key metrics
            method_vals = df_method[metric].dropna()
            baseline_vals = baseline_values[label]

            if len(method_vals) > 0 and len(baseline_vals) > 0:
                t_stat, p_value = stats.ttest_ind(baseline_vals, method_vals)
                mean_diff = method_vals.mean() - baseline_vals.mean()

                sig = ''
                if p_value < 0.001:
                    sig = '***'
                elif p_value < 0.01:
                    sig = '**'
                elif p_value < 0.05:
                    sig = '*'

                print(f"  {label:12s}: Δ={mean_diff:+.4f}, t={t_stat:+6.3f}, p={p_value:.4f} {sig}")

        print()


if __name__ == "__main__":
    create_comparison_table()
