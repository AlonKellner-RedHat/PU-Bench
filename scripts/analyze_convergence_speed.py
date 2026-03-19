#!/usr/bin/env python3
"""
Analyze convergence speed across all VPU variants.

Examines:
- Epochs to best validation metric
- Time to best metric (seconds)
- Early stopping patterns
- Convergence efficiency
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# Configuration
SEEDS = [42, 123, 456, 789, 2024]
RESULTS_DIR = Path('results')

# All VPU variants
METHODS = [
    'vpu',
    'vpu_mean',
    'vpu_mean_prior',
    'vpu_nomixup',
    'vpu_nomixup_mean',
    'vpu_nomixup_mean_prior',
]


def load_convergence_data() -> pd.DataFrame:
    """Load convergence metrics from all results."""
    records = []

    for seed in SEEDS:
        seed_dir = RESULTS_DIR / f'seed_{seed}'
        if not seed_dir.exists():
            continue

        # Find all result files
        for config_file in seed_dir.glob('*.json'):
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)
            except:
                continue

            experiment = data.get('experiment', '')

            # Parse experiment details
            parts = experiment.split('_')
            dataset = parts[0] if parts else 'unknown'

            # Extract metrics for each method
            for method_name, method_data in data.get('runs', {}).items():
                if method_name not in METHODS:
                    continue

                # Get best epoch info
                if 'best' not in method_data:
                    continue

                best_info = method_data['best']

                # Extract convergence metrics
                best_epoch = best_info.get('epoch', None)

                # Get timing info
                timing = method_data.get('timing', {})
                total_time = timing.get('duration_seconds', None)

                # Try to get time to best from best_info
                time_to_best = None
                if 'time_to_best' in best_info:
                    # Parse time_to_best (format: "97.53s" or similar)
                    time_str = best_info.get('time_to_best', '')
                    if isinstance(time_str, str) and time_str.endswith('s'):
                        try:
                            time_to_best = float(time_str[:-1])
                        except:
                            pass
                    elif isinstance(time_str, (int, float)):
                        time_to_best = float(time_str)

                # Get best metrics
                best_metrics = best_info.get('metrics', {})
                best_f1 = best_metrics.get('test_f1', None)
                best_auc = best_metrics.get('test_auc', None)

                # Get global epochs (total epochs run)
                global_epochs = method_data.get('global_epochs', None)

                record = {
                    'experiment': experiment,
                    'dataset': dataset,
                    'seed': seed,
                    'method': method_name,
                    'best_epoch': best_epoch,
                    'global_epochs': global_epochs,
                    'time_to_best': time_to_best,
                    'total_time': total_time,
                    'test_f1': best_f1,
                    'test_auc': best_auc,
                }

                records.append(record)

    df = pd.DataFrame(records)

    if len(df) == 0:
        print("WARNING: No convergence data loaded!")
        return df

    # Calculate efficiency metrics
    df['convergence_efficiency'] = df['best_epoch'] / df['global_epochs']

    print(f"Loaded convergence data for {len(df)} runs")
    print(f"Methods: {sorted(df['method'].unique())}")
    print(f"Datasets: {sorted(df['dataset'].unique())}")

    return df


def analyze_overall_convergence(df: pd.DataFrame) -> str:
    """Analyze overall convergence speed."""
    output = []
    output.append("=" * 80)
    output.append("CONVERGENCE SPEED ANALYSIS: All VPU Variants")
    output.append("=" * 80)
    output.append("")

    # Overall statistics by method
    output.append("Method Comparison (mean ± std)")
    output.append("-" * 80)
    output.append(f"{'Method':<25} {'Epochs':<15} {'Time (s)':<15} {'Efficiency':<15} {'F1':<10}")
    output.append("-" * 80)

    method_stats = []

    for method in METHODS:
        df_method = df[df['method'] == method]

        if len(df_method) == 0:
            continue

        epochs_mean = df_method['best_epoch'].mean()
        epochs_std = df_method['best_epoch'].std()

        time_mean = df_method['time_to_best'].mean()
        time_std = df_method['time_to_best'].std()

        efficiency = df_method['convergence_efficiency'].mean()

        f1_mean = df_method['test_f1'].mean()

        method_stats.append({
            'method': method,
            'epochs': epochs_mean,
            'time': time_mean,
            'efficiency': efficiency,
            'f1': f1_mean,
        })

        output.append(
            f"{method:<25} "
            f"{epochs_mean:>6.1f}±{epochs_std:>4.1f}    "
            f"{time_mean:>6.1f}±{time_std:>4.1f}    "
            f"{efficiency:>6.1%}         "
            f"{f1_mean:>6.4f}"
        )

    # Rank by speed
    output.append("\n" + "=" * 80)
    output.append("CONVERGENCE SPEED RANKING (by time to best)")
    output.append("=" * 80)

    method_stats_sorted = sorted(method_stats, key=lambda x: x['time'] if not np.isnan(x['time']) else float('inf'))

    output.append(f"\n{'Rank':<6} {'Method':<25} {'Time (s)':<12} {'Epochs':<10} {'F1':<10}")
    output.append("-" * 80)

    for i, stats in enumerate(method_stats_sorted, 1):
        if np.isnan(stats['time']):
            continue
        output.append(
            f"{i:<6} {stats['method']:<25} "
            f"{stats['time']:>8.1f}     "
            f"{stats['epochs']:>6.1f}     "
            f"{stats['f1']:>6.4f}"
        )

    return "\n".join(output)


def analyze_by_dataset(df: pd.DataFrame) -> str:
    """Analyze convergence by dataset."""
    output = []
    output.append("\n" + "=" * 80)
    output.append("CONVERGENCE SPEED BY DATASET")
    output.append("=" * 80)

    datasets = sorted(df['dataset'].unique())

    for dataset in datasets:
        df_dataset = df[df['dataset'] == dataset]

        output.append(f"\n{dataset}")
        output.append("-" * 80)
        output.append(f"{'Method':<25} {'Epochs':<12} {'Time (s)':<12} {'F1':<10}")
        output.append("-" * 80)

        for method in METHODS:
            df_method = df_dataset[df_dataset['method'] == method]

            if len(df_method) == 0:
                continue

            epochs = df_method['best_epoch'].mean()
            time = df_method['time_to_best'].mean()
            f1 = df_method['test_f1'].mean()

            output.append(
                f"{method:<25} {epochs:>8.1f}     "
                f"{time:>8.1f}     {f1:>6.4f}"
            )

    return "\n".join(output)


def analyze_with_vs_without_mixup(df: pd.DataFrame) -> str:
    """Compare convergence with and without MixUp."""
    output = []
    output.append("\n" + "=" * 80)
    output.append("CONVERGENCE: WITH vs WITHOUT MixUp")
    output.append("=" * 80)

    # Group methods
    with_mixup = ['vpu', 'vpu_mean', 'vpu_mean_prior']
    without_mixup = ['vpu_nomixup', 'vpu_nomixup_mean', 'vpu_nomixup_mean_prior']

    output.append("\nWith MixUp:")
    output.append("-" * 80)
    output.append(f"{'Method':<25} {'Epochs':<12} {'Time (s)':<12} {'F1':<10}")
    output.append("-" * 80)

    for method in with_mixup:
        df_method = df[df['method'] == method]
        if len(df_method) == 0:
            continue

        epochs = df_method['best_epoch'].mean()
        time = df_method['time_to_best'].mean()
        f1 = df_method['test_f1'].mean()

        output.append(f"{method:<25} {epochs:>8.1f}     {time:>8.1f}     {f1:>6.4f}")

    output.append("\nWithout MixUp:")
    output.append("-" * 80)
    output.append(f"{'Method':<25} {'Epochs':<12} {'Time (s)':<12} {'F1':<10}")
    output.append("-" * 80)

    for method in without_mixup:
        df_method = df[df['method'] == method]
        if len(df_method) == 0:
            continue

        epochs = df_method['best_epoch'].mean()
        time = df_method['time_to_best'].mean()
        f1 = df_method['test_f1'].mean()

        output.append(f"{method:<25} {epochs:>8.1f}     {time:>8.1f}     {f1:>6.4f}")

    # Summary
    df_with = df[df['method'].isin(with_mixup)]
    df_without = df[df['method'].isin(without_mixup)]

    output.append("\nSummary:")
    output.append("-" * 80)
    output.append(f"With MixUp:    {df_with['time_to_best'].mean():>6.1f}s avg, {df_with['best_epoch'].mean():>5.1f} epochs avg")
    output.append(f"Without MixUp: {df_without['time_to_best'].mean():>6.1f}s avg, {df_without['best_epoch'].mean():>5.1f} epochs avg")

    time_diff = df_without['time_to_best'].mean() - df_with['time_to_best'].mean()
    pct_diff = (time_diff / df_with['time_to_best'].mean()) * 100

    if time_diff > 0:
        output.append(f"\nWithout MixUp is {abs(pct_diff):.1f}% SLOWER (+{time_diff:.1f}s)")
    else:
        output.append(f"\nWithout MixUp is {abs(pct_diff):.1f}% FASTER ({time_diff:.1f}s)")

    return "\n".join(output)


def analyze_prior_impact_on_convergence(df: pd.DataFrame) -> str:
    """Analyze how prior weighting affects convergence."""
    output = []
    output.append("\n" + "=" * 80)
    output.append("IMPACT OF PRIOR WEIGHTING ON CONVERGENCE")
    output.append("=" * 80)

    comparisons = [
        ('vpu_mean_prior', 'vpu_mean', 'With MixUp'),
        ('vpu_nomixup_mean_prior', 'vpu_nomixup_mean', 'Without MixUp'),
    ]

    for method_prior, method_base, context in comparisons:
        output.append(f"\n{context}: {method_prior} vs {method_base}")
        output.append("-" * 80)

        df_prior = df[df['method'] == method_prior]
        df_base = df[df['method'] == method_base]

        if len(df_prior) == 0 or len(df_base) == 0:
            output.append("No data available")
            continue

        # Compare convergence metrics
        epochs_prior = df_prior['best_epoch'].mean()
        epochs_base = df_base['best_epoch'].mean()
        epochs_diff = epochs_prior - epochs_base

        time_prior = df_prior['time_to_best'].mean()
        time_base = df_base['time_to_best'].mean()
        time_diff = time_prior - time_base

        f1_prior = df_prior['test_f1'].mean()
        f1_base = df_base['test_f1'].mean()
        f1_diff = ((f1_prior - f1_base) / f1_base) * 100

        output.append(f"{'Metric':<20} {'Prior':<12} {'Base':<12} {'Difference':<15}")
        output.append("-" * 80)
        output.append(f"{'Epochs to best':<20} {epochs_prior:>8.1f}     {epochs_base:>8.1f}     {epochs_diff:+8.1f}")
        output.append(f"{'Time to best (s)':<20} {time_prior:>8.1f}     {time_base:>8.1f}     {time_diff:+8.1f}s")
        output.append(f"{'F1 Score':<20} {f1_prior:>8.4f}     {f1_base:>8.4f}     {f1_diff:+8.2f}%")

        # Interpretation
        if abs(time_diff) < 1.0:
            speed_verdict = "Same speed"
        elif time_diff > 0:
            speed_verdict = f"Prior is {abs(time_diff):.1f}s SLOWER"
        else:
            speed_verdict = f"Prior is {abs(time_diff):.1f}s FASTER"

        output.append(f"\nConclusion: {speed_verdict}")

    return "\n".join(output)


def main():
    """Run convergence analysis."""
    print("Loading convergence data...")
    df = load_convergence_data()

    if len(df) == 0:
        print("No data to analyze!")
        return

    print("\nGenerating analysis...")

    sections = [
        analyze_overall_convergence(df),
        analyze_with_vs_without_mixup(df),
        analyze_prior_impact_on_convergence(df),
        analyze_by_dataset(df),
    ]

    full_report = "\n\n".join(sections)

    output_file = RESULTS_DIR / 'CONVERGENCE_ANALYSIS.md'
    with open(output_file, 'w') as f:
        f.write(full_report)

    print(f"\n✓ Analysis complete!")
    print(f"  Report saved to: {output_file}")

    # Print summary
    print("\n" + analyze_overall_convergence(df))


if __name__ == '__main__':
    main()
