#!/usr/bin/env python3
"""
Comprehensive multi-seed analysis across all metrics including calibration.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


def load_multiseed_results_all_metrics():
    """Load results from all seeds with all metrics."""
    seeds = [42, 123, 456, 789, 2024]
    target_methods = {'oracle_bce', 'vpu', 'vpu_mean', 'vpu_nomixup', 'vpu_nomixup_mean'}
    target_datasets = {'MNIST', 'FashionMNIST', 'IMDB', '20News', 'Spambase', 'Mushrooms'}

    data = []

    for seed in seeds:
        results_dir = Path(f'results/seed_{seed}')
        if not results_dir.exists():
            continue

        for result_file in results_dir.glob('*.json'):
            dataset_name = result_file.name.split('_')[0]
            if dataset_name not in target_datasets:
                continue

            with open(result_file) as f:
                result_data = json.load(f)

            # Parse config from filename
            parts = result_file.stem.split('_')
            config = {'dataset': dataset_name, 'seed': seed}

            for i, part in enumerate(parts):
                if part.startswith('c') and i > 0:
                    try:
                        config['c'] = float(part[1:])
                    except ValueError:
                        pass
                elif part.startswith('prior'):
                    try:
                        config['prior'] = float(part[5:])
                    except ValueError:
                        pass

            # Extract all metrics
            for method_name, method_data in result_data.get('runs', {}).items():
                if method_name not in target_methods:
                    continue

                best_metrics = method_data.get('best', {}).get('metrics', {})

                row = {
                    'dataset': dataset_name,
                    'method': method_name,
                    'c': config.get('c'),
                    'prior': config.get('prior'),
                    'seed': seed,
                    # Performance metrics
                    'test_f1': best_metrics.get('test_f1'),
                    'test_auc': best_metrics.get('test_auc'),
                    'test_accuracy': best_metrics.get('test_accuracy'),
                    'test_precision': best_metrics.get('test_precision'),
                    'test_recall': best_metrics.get('test_recall'),
                    # Calibration metrics
                    'test_anice': best_metrics.get('test_anice'),
                    'test_snice': best_metrics.get('test_snice'),
                    'test_ece': best_metrics.get('test_ece'),
                    'test_mce': best_metrics.get('test_mce'),
                    'test_brier': best_metrics.get('test_brier'),
                }

                data.append(row)

    return pd.DataFrame(data)


def analyze_metric_by_dataset(df, metric_name, lower_is_better=False):
    """Analyze a single metric across datasets with statistical testing."""
    print(f"\n{'=' * 80}")
    print(f"METRIC: {metric_name.upper()} ({'lower is better' if lower_is_better else 'higher is better'})")
    print(f"{'=' * 80}\n")

    vpu_df = df[df['method'].isin(['vpu', 'vpu_mean'])].copy()

    results = []

    for dataset in sorted(vpu_df['dataset'].unique()):
        ds_data = vpu_df[vpu_df['dataset'] == dataset]

        vpu_scores = ds_data[ds_data['method'] == 'vpu'][metric_name].dropna().values
        vpu_mean_scores = ds_data[ds_data['method'] == 'vpu_mean'][metric_name].dropna().values

        if len(vpu_scores) > 1 and len(vpu_mean_scores) > 1:
            # Welch's t-test
            t_stat, p_value = stats.ttest_ind(vpu_mean_scores, vpu_scores, equal_var=False)

            # Effect size
            pooled_std = np.sqrt((np.var(vpu_scores, ddof=1) + np.var(vpu_mean_scores, ddof=1)) / 2)
            cohens_d = (np.mean(vpu_mean_scores) - np.mean(vpu_scores)) / pooled_std if pooled_std > 0 else 0

            # Determine winner based on metric direction
            if lower_is_better:
                winner = "VPU-Mean" if np.mean(vpu_mean_scores) < np.mean(vpu_scores) else "VPU"
                advantage = np.mean(vpu_scores) - np.mean(vpu_mean_scores)  # Flip for lower is better
            else:
                winner = "VPU-Mean" if np.mean(vpu_mean_scores) > np.mean(vpu_scores) else "VPU"
                advantage = np.mean(vpu_mean_scores) - np.mean(vpu_scores)

            results.append({
                'dataset': dataset,
                'vpu': np.mean(vpu_scores),
                'vpu_std': np.std(vpu_scores, ddof=1),
                'vpu_mean': np.mean(vpu_mean_scores),
                'vpu_mean_std': np.std(vpu_mean_scores, ddof=1),
                'advantage': advantage,
                'cohens_d': cohens_d,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'winner': winner
            })

    results_df = pd.DataFrame(results).sort_values('p_value')

    # Print summary
    print(f"Dataset-level comparisons:")
    print("-" * 80)
    for _, row in results_df.iterrows():
        sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else "ns"
        print(f"{row['dataset']:15s}: VPU={row['vpu']:.4f}±{row['vpu_std']:.4f}, "
              f"VPU-Mean={row['vpu_mean']:.4f}±{row['vpu_mean_std']:.4f}, "
              f"Adv={row['advantage']:+.4f}, p={row['p_value']:.4f} {sig} ({row['winner']})")

    # Overall comparison
    print(f"\nOverall ({metric_name}):")
    print("-" * 80)
    vpu_all = vpu_df[vpu_df['method'] == 'vpu'][metric_name].dropna().values
    vpu_mean_all = vpu_df[vpu_df['method'] == 'vpu_mean'][metric_name].dropna().values

    t_stat, p_value = stats.ttest_ind(vpu_mean_all, vpu_all, equal_var=False)

    if lower_is_better:
        overall_advantage = np.mean(vpu_all) - np.mean(vpu_mean_all)
        overall_winner = "VPU-Mean" if np.mean(vpu_mean_all) < np.mean(vpu_all) else "VPU"
    else:
        overall_advantage = np.mean(vpu_mean_all) - np.mean(vpu_all)
        overall_winner = "VPU-Mean" if np.mean(vpu_mean_all) > np.mean(vpu_all) else "VPU"

    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    print(f"VPU:      {np.mean(vpu_all):.4f} ± {np.std(vpu_all, ddof=1):.4f} (n={len(vpu_all)})")
    print(f"VPU-Mean: {np.mean(vpu_mean_all):.4f} ± {np.std(vpu_mean_all, ddof=1):.4f} (n={len(vpu_mean_all)})")
    print(f"Advantage: {overall_advantage:+.4f}, p={p_value:.4f} {sig} ({overall_winner})")

    return results_df


def analyze_calibration_metrics(df):
    """Analyze calibration metrics specifically."""
    print("\n" + "=" * 80)
    print("CALIBRATION METRICS ANALYSIS")
    print("=" * 80)

    calibration_metrics = {
        'test_anice': ('A-NICE', True),   # Lower is better
        'test_snice': ('S-NICE', True),   # Lower is better
        'test_ece': ('ECE', True),        # Lower is better
        'test_mce': ('MCE', True),        # Lower is better
        'test_brier': ('Brier Score', True),  # Lower is better
    }

    summary = []

    for metric_col, (metric_name, lower_is_better) in calibration_metrics.items():
        print(f"\n{'-' * 80}")
        print(f"{metric_name} (lower is better for calibration)")
        print(f"{'-' * 80}")

        vpu_df = df[df['method'].isin(['vpu', 'vpu_mean'])].copy()

        # Overall comparison
        vpu_scores = vpu_df[vpu_df['method'] == 'vpu'][metric_col].dropna().values
        vpu_mean_scores = vpu_df[vpu_df['method'] == 'vpu_mean'][metric_col].dropna().values

        if len(vpu_scores) > 5 and len(vpu_mean_scores) > 5:
            t_stat, p_value = stats.ttest_ind(vpu_mean_scores, vpu_scores, equal_var=False)

            vpu_better_count = np.sum(vpu_scores < vpu_mean_scores)
            vpu_mean_better_count = np.sum(vpu_mean_scores < vpu_scores)

            winner = "VPU-Mean" if np.mean(vpu_mean_scores) < np.mean(vpu_scores) else "VPU"
            improvement = (np.mean(vpu_scores) - np.mean(vpu_mean_scores)) / np.mean(vpu_scores) * 100

            print(f"VPU:      {np.mean(vpu_scores):.4f} ± {np.std(vpu_scores, ddof=1):.4f}")
            print(f"VPU-Mean: {np.mean(vpu_mean_scores):.4f} ± {np.std(vpu_mean_scores, ddof=1):.4f}")
            print(f"Improvement: {improvement:+.2f}% ({winner} is better)")
            print(f"p-value: {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
            print(f"Win rate: VPU-Mean better in {vpu_mean_better_count}/{len(vpu_scores)} cases ({vpu_mean_better_count/len(vpu_scores)*100:.1f}%)")

            summary.append({
                'metric': metric_name,
                'vpu': np.mean(vpu_scores),
                'vpu_mean': np.mean(vpu_mean_scores),
                'improvement_%': improvement,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'winner': winner
            })

    print("\n" + "=" * 80)
    print("CALIBRATION SUMMARY")
    print("=" * 80)
    summary_df = pd.DataFrame(summary)
    for _, row in summary_df.iterrows():
        sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else "ns"
        print(f"{row['metric']:15s}: {row['improvement_%']:+6.2f}% improvement, p={row['p_value']:.4f} {sig} ({row['winner']})")

    return summary_df


def analyze_performance_vs_calibration_tradeoff(df):
    """Analyze the tradeoff between performance (F1, AUC) and calibration."""
    print("\n" + "=" * 80)
    print("PERFORMANCE vs CALIBRATION TRADEOFF")
    print("=" * 80)

    vpu_df = df[df['method'].isin(['vpu', 'vpu_mean'])].copy()

    # For each dataset, check if better performance comes at calibration cost
    for dataset in sorted(vpu_df['dataset'].unique()):
        ds_data = vpu_df[vpu_df['dataset'] == dataset].copy()

        metrics_to_check = [
            ('test_f1', 'F1', False),
            ('test_auc', 'AUC', False),
            ('test_anice', 'A-NICE', True),
            ('test_ece', 'ECE', True),
        ]

        print(f"\n{dataset}:")
        print("-" * 60)

        for metric_col, metric_name, lower_is_better in metrics_to_check:
            vpu_vals = ds_data[ds_data['method'] == 'vpu'][metric_col].dropna()
            vpu_mean_vals = ds_data[ds_data['method'] == 'vpu_mean'][metric_col].dropna()

            if len(vpu_vals) > 0 and len(vpu_mean_vals) > 0:
                if lower_is_better:
                    better = "VPU-Mean" if vpu_mean_vals.mean() < vpu_vals.mean() else "VPU"
                    diff = vpu_vals.mean() - vpu_mean_vals.mean()
                else:
                    better = "VPU-Mean" if vpu_mean_vals.mean() > vpu_vals.mean() else "VPU"
                    diff = vpu_mean_vals.mean() - vpu_vals.mean()

                print(f"  {metric_name:8s}: VPU={vpu_vals.mean():.4f}, VPU-Mean={vpu_mean_vals.mean():.4f}, "
                      f"Δ={diff:+.4f} ({better})")


def comprehensive_metric_summary(df):
    """Generate a comprehensive summary across all metrics."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE METRIC SUMMARY")
    print("=" * 80)

    metrics = {
        'Performance Metrics': {
            'test_f1': ('F1 Score', False),
            'test_auc': ('AUC', False),
            'test_accuracy': ('Accuracy', False),
            'test_precision': ('Precision', False),
            'test_recall': ('Recall', False),
        },
        'Calibration Metrics': {
            'test_anice': ('A-NICE', True),
            'test_snice': ('S-NICE', True),
            'test_ece': ('ECE', True),
            'test_mce': ('MCE', True),
            'test_brier': ('Brier Score', True),
        }
    }

    vpu_df = df[df['method'].isin(['vpu', 'vpu_mean'])].copy()

    for category, metric_dict in metrics.items():
        print(f"\n{category}:")
        print("-" * 80)
        print(f"{'Metric':<20s} {'VPU':>12s} {'VPU-Mean':>12s} {'Advantage':>12s} {'p-value':>10s} {'Winner':>12s}")
        print("-" * 80)

        for metric_col, (metric_name, lower_is_better) in metric_dict.items():
            vpu_vals = vpu_df[vpu_df['method'] == 'vpu'][metric_col].dropna().values
            vpu_mean_vals = vpu_df[vpu_df['method'] == 'vpu_mean'][metric_col].dropna().values

            if len(vpu_vals) > 5 and len(vpu_mean_vals) > 5:
                t_stat, p_value = stats.ttest_ind(vpu_mean_vals, vpu_vals, equal_var=False)

                if lower_is_better:
                    winner = "VPU-Mean" if np.mean(vpu_mean_vals) < np.mean(vpu_vals) else "VPU"
                    advantage = np.mean(vpu_vals) - np.mean(vpu_mean_vals)
                else:
                    winner = "VPU-Mean" if np.mean(vpu_mean_vals) > np.mean(vpu_vals) else "VPU"
                    advantage = np.mean(vpu_mean_vals) - np.mean(vpu_vals)

                sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""

                print(f"{metric_name:<20s} {np.mean(vpu_vals):>12.4f} {np.mean(vpu_mean_vals):>12.4f} "
                      f"{advantage:>+12.4f} {p_value:>10.4f}{sig:3s} {winner:>12s}")


def main():
    print("Loading multi-seed results with all metrics...")
    df = load_multiseed_results_all_metrics()

    seeds = df['seed'].unique()
    print(f"Loaded {len(df)} results from {len(seeds)} seeds: {sorted(seeds)}")
    print(f"Datasets: {sorted(df['dataset'].unique())}")
    print()

    # Comprehensive summary first
    comprehensive_metric_summary(df)

    # Performance metrics
    print("\n" + "=" * 80)
    print("DETAILED PERFORMANCE METRICS ANALYSIS")
    print("=" * 80)

    analyze_metric_by_dataset(df, 'test_f1', lower_is_better=False)
    analyze_metric_by_dataset(df, 'test_auc', lower_is_better=False)

    # Calibration metrics
    analyze_calibration_metrics(df)

    # Tradeoff analysis
    analyze_performance_vs_calibration_tradeoff(df)


if __name__ == '__main__':
    main()
