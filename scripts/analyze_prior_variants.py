#!/usr/bin/env python3
"""
Analyze VPU-Mean-Prior and VPU-NoMixUp-Mean-Prior variants.

Compares prior-weighted variants against their non-prior counterparts:
- VPU-Mean-Prior vs VPU-Mean
- VPU-NoMixUp-Mean-Prior vs VPU-NoMixUp-Mean

Analyzes:
- Overall performance (F1, recall, precision, AUC)
- Calibration metrics (A-NICE, S-NICE, ECE, MCE, Brier)
- Performance by dataset
- Performance by label frequency (c)
- Performance by prior value
- Statistical significance (paired t-tests)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple

# Configuration
SEEDS = [42, 123, 456, 789, 2024]
DATASETS = ['MNIST', 'FashionMNIST', 'IMDB', '20News', 'Spambase', 'Mushrooms']
RESULTS_DIR = Path('results')

# Method pairs to compare
COMPARISONS = [
    ('vpu_mean_prior', 'vpu_mean', 'With MixUp'),
    ('vpu_nomixup_mean_prior', 'vpu_nomixup_mean', 'Without MixUp'),
]

# Metrics to analyze
PERFORMANCE_METRICS = ['test_f1', 'test_recall', 'test_precision', 'test_auc', 'test_accuracy']
CALIBRATION_METRICS = ['test_anice', 'test_snice', 'test_ece', 'test_mce', 'test_brier']


def load_results() -> pd.DataFrame:
    """Load all results with prior configurations."""
    records = []

    for seed in SEEDS:
        seed_dir = RESULTS_DIR / f'seed_{seed}'
        if not seed_dir.exists():
            continue

        # Find all prior variant config files
        for config_file in seed_dir.glob('*_prior*.json'):
            with open(config_file, 'r') as f:
                data = json.load(f)

            experiment = data['experiment']

            # Parse experiment name
            parts = experiment.split('_')
            dataset = parts[0]

            # Extract c value and prior value
            c_value = None
            prior_value = None
            for part in parts:
                if part.startswith('c') and not part.startswith('case'):
                    try:
                        c_value = float(part[1:])
                    except:
                        pass
                if part.startswith('prior'):
                    try:
                        prior_value = float(part[5:])
                    except:
                        pass

            # Extract seed
            seed_val = None
            for part in parts:
                if part.startswith('seed'):
                    try:
                        seed_val = int(part[4:])
                    except:
                        pass

            # Extract metrics for each method
            for method_name, method_data in data.get('runs', {}).items():
                # Metrics are stored in best.metrics
                if 'best' not in method_data or 'metrics' not in method_data['best']:
                    continue

                metrics = method_data['best']['metrics']

                record = {
                    'experiment': experiment,
                    'dataset': dataset,
                    'seed': seed_val,
                    'c': c_value,
                    'prior': prior_value,
                    'method': method_name,
                }

                # Add all metrics
                for metric in PERFORMANCE_METRICS + CALIBRATION_METRICS:
                    record[metric] = metrics.get(metric)

                records.append(record)

    df = pd.DataFrame(records)

    if len(df) == 0:
        print("WARNING: No results loaded!")
        return df

    print(f"Loaded {len(df)} results from {len(df['experiment'].unique())} experiments")
    print(f"Methods: {sorted(df['method'].unique())}")
    print(f"Datasets: {sorted(df['dataset'].unique())}")
    print(f"Seeds: {sorted(df['seed'].unique())}")

    return df


def compute_paired_ttest(df: pd.DataFrame, method1: str, method2: str, metric: str) -> Tuple[float, float, float, int]:
    """
    Compute paired t-test between two methods on a specific metric.

    Returns: (t_statistic, p_value, cohen_d, n_pairs)
    """
    # Create pivot table to match pairs
    pivot = df.pivot_table(
        index=['dataset', 'seed', 'c', 'prior'],
        columns='method',
        values=metric
    )

    # Drop rows with missing data
    pivot = pivot.dropna()

    if len(pivot) == 0 or method1 not in pivot.columns or method2 not in pivot.columns:
        return np.nan, np.nan, np.nan, 0

    values1 = pivot[method1].values
    values2 = pivot[method2].values

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(values1, values2)

    # Cohen's d for paired samples
    diff = values1 - values2
    cohen_d = np.mean(diff) / np.std(diff, ddof=1)

    return t_stat, p_value, cohen_d, len(values1)


def format_pvalue(p: float) -> str:
    """Format p-value with significance stars."""
    if np.isnan(p):
        return 'N/A'
    stars = ''
    if p < 0.001:
        stars = ' ***'
    elif p < 0.01:
        stars = ' **'
    elif p < 0.05:
        stars = ' *'
    return f"{p:.4f}{stars}"


def format_metric_comparison(val1: float, val2: float, metric: str, p_value: float) -> str:
    """Format metric comparison with direction indicator."""
    if np.isnan(val1) or np.isnan(val2):
        return 'N/A'

    diff = val1 - val2
    pct_diff = (diff / abs(val2)) * 100 if val2 != 0 else 0

    # For calibration metrics (lower is better), flip the interpretation
    is_calibration = metric in ['test_anice', 'test_snice', 'test_ece', 'test_mce', 'test_brier']

    if is_calibration:
        better = '✓' if diff < 0 else ''
    else:
        better = '✓' if diff > 0 else ''

    sig = ''
    if not np.isnan(p_value):
        if p_value < 0.001:
            sig = ' ***'
        elif p_value < 0.01:
            sig = ' **'
        elif p_value < 0.05:
            sig = ' *'

    return f"{pct_diff:+.1f}%{sig} {better}"


def analyze_overall(df: pd.DataFrame) -> str:
    """Generate overall comparison analysis."""
    output = []
    output.append("=" * 80)
    output.append("OVERALL PERFORMANCE: Prior-Weighted vs Non-Prior Variants")
    output.append("=" * 80)
    output.append("")

    for method_prior, method_base, context in COMPARISONS:
        output.append(f"\n{context}: {method_prior.upper()} vs {method_base.upper()}")
        output.append("-" * 80)

        # Filter data
        df_comparison = df[df['method'].isin([method_prior, method_base])].copy()

        if len(df_comparison) == 0:
            output.append("No data available")
            continue

        # Overall means
        output.append(f"\n{'Metric':<15} {method_prior:<12} {method_base:<12} {'Difference':<12} {'p-value':<12} {'Winner'}")
        output.append("-" * 80)

        for metric in PERFORMANCE_METRICS + CALIBRATION_METRICS:
            mean_prior = df_comparison[df_comparison['method'] == method_prior][metric].mean()
            mean_base = df_comparison[df_comparison['method'] == method_base][metric].mean()

            t_stat, p_value, cohen_d, n_pairs = compute_paired_ttest(
                df_comparison, method_prior, method_base, metric
            )

            metric_name = metric.replace('test_', '')
            diff_str = format_metric_comparison(mean_prior, mean_base, metric, p_value)
            p_str = format_pvalue(p_value)

            # Determine winner
            is_calibration = metric in CALIBRATION_METRICS
            if np.isnan(mean_prior) or np.isnan(mean_base):
                winner = 'N/A'
            elif is_calibration:
                winner = method_prior if mean_prior < mean_base else method_base
            else:
                winner = method_prior if mean_prior > mean_base else method_base

            # Make it short
            winner = 'Prior' if winner == method_prior else ('Base' if winner == method_base else 'N/A')

            output.append(f"{metric_name:<15} {mean_prior:>12.4f} {mean_base:>12.4f} {diff_str:<12} {p_str:<12} {winner}")

        output.append(f"\nPaired comparisons based on {n_pairs} matched configurations")
        if not np.isnan(cohen_d):
            output.append(f"Effect size (Cohen's d): {cohen_d:.3f}")

    return "\n".join(output)


def analyze_by_dataset(df: pd.DataFrame) -> str:
    """Analyze performance by dataset."""
    output = []
    output.append("\n" + "=" * 80)
    output.append("PERFORMANCE BY DATASET")
    output.append("=" * 80)

    for method_prior, method_base, context in COMPARISONS:
        output.append(f"\n{context}: {method_prior.upper()} vs {method_base.upper()}")
        output.append("-" * 80)

        df_comparison = df[df['method'].isin([method_prior, method_base])].copy()

        output.append(f"\n{'Dataset':<15} {'Prior F1':<12} {'Base F1':<12} {'Diff':<12} {'p-value':<12} {'Winner'}")
        output.append("-" * 80)

        for dataset in DATASETS:
            df_dataset = df_comparison[df_comparison['dataset'] == dataset]

            if len(df_dataset) == 0:
                continue

            f1_prior = df_dataset[df_dataset['method'] == method_prior]['test_f1'].mean()
            f1_base = df_dataset[df_dataset['method'] == method_base]['test_f1'].mean()

            t_stat, p_value, cohen_d, n_pairs = compute_paired_ttest(
                df_dataset, method_prior, method_base, 'test_f1'
            )

            diff_str = format_metric_comparison(f1_prior, f1_base, 'test_f1', p_value)
            p_str = format_pvalue(p_value)
            winner = 'Prior' if f1_prior > f1_base else 'Base'

            output.append(f"{dataset:<15} {f1_prior:>12.4f} {f1_base:>12.4f} {diff_str:<12} {p_str:<12} {winner}")

    return "\n".join(output)


def analyze_by_c(df: pd.DataFrame) -> str:
    """Analyze performance by label frequency (c)."""
    output = []
    output.append("\n" + "=" * 80)
    output.append("PERFORMANCE BY LABEL FREQUENCY (c)")
    output.append("=" * 80)

    for method_prior, method_base, context in COMPARISONS:
        output.append(f"\n{context}: {method_prior.upper()} vs {method_base.upper()}")
        output.append("-" * 80)

        df_comparison = df[df['method'].isin([method_prior, method_base])].copy()
        df_comparison = df_comparison[df_comparison['c'].notna()]

        if len(df_comparison) == 0:
            output.append("No c-value data available")
            continue

        c_values = sorted(df_comparison['c'].unique())

        output.append(f"\n{'c value':<10} {'Prior F1':<12} {'Base F1':<12} {'Diff':<12} {'p-value':<12} {'Winner'}")
        output.append("-" * 80)

        for c in c_values:
            df_c = df_comparison[df_comparison['c'] == c]

            f1_prior = df_c[df_c['method'] == method_prior]['test_f1'].mean()
            f1_base = df_c[df_c['method'] == method_base]['test_f1'].mean()

            t_stat, p_value, cohen_d, n_pairs = compute_paired_ttest(
                df_c, method_prior, method_base, 'test_f1'
            )

            diff_str = format_metric_comparison(f1_prior, f1_base, 'test_f1', p_value)
            p_str = format_pvalue(p_value)
            winner = 'Prior' if f1_prior > f1_base else 'Base'

            output.append(f"{c:<10.2f} {f1_prior:>12.4f} {f1_base:>12.4f} {diff_str:<12} {p_str:<12} {winner}")

    return "\n".join(output)


def analyze_by_prior(df: pd.DataFrame) -> str:
    """Analyze performance by prior value."""
    output = []
    output.append("\n" + "=" * 80)
    output.append("PERFORMANCE BY PRIOR VALUE")
    output.append("=" * 80)

    for method_prior, method_base, context in COMPARISONS:
        output.append(f"\n{context}: {method_prior.upper()} vs {method_base.upper()}")
        output.append("-" * 80)

        df_comparison = df[df['method'].isin([method_prior, method_base])].copy()
        df_comparison = df_comparison[df_comparison['prior'].notna()]

        if len(df_comparison) == 0:
            output.append("No prior-value data available")
            continue

        prior_values = sorted(df_comparison['prior'].unique())

        output.append(f"\n{'Prior':<10} {'Prior F1':<12} {'Base F1':<12} {'Diff':<12} {'p-value':<12} {'Winner'}")
        output.append("-" * 80)

        for prior_val in prior_values:
            df_prior = df_comparison[df_comparison['prior'] == prior_val]

            f1_prior = df_prior[df_prior['method'] == method_prior]['test_f1'].mean()
            f1_base = df_prior[df_prior['method'] == method_base]['test_f1'].mean()

            t_stat, p_value, cohen_d, n_pairs = compute_paired_ttest(
                df_prior, method_prior, method_base, 'test_f1'
            )

            diff_str = format_metric_comparison(f1_prior, f1_base, 'test_f1', p_value)
            p_str = format_pvalue(p_value)
            winner = 'Prior' if f1_prior > f1_base else 'Base'

            output.append(f"{prior_val:<10.1f} {f1_prior:>12.4f} {f1_base:>12.4f} {diff_str:<12} {p_str:<12} {winner}")

    return "\n".join(output)


def generate_summary(df: pd.DataFrame) -> str:
    """Generate executive summary."""
    output = []
    output.append("=" * 80)
    output.append("EXECUTIVE SUMMARY: Impact of Prior Weighting")
    output.append("=" * 80)
    output.append("")
    output.append("Analysis of prior-weighted variants:")
    output.append("- VPU-Mean-Prior:         L = mean(φ(x)) - π·mean(log(φ_p(x))) + MixUp")
    output.append("- VPU-NoMixUp-Mean-Prior: L = mean(φ(x)) - π·mean(log(φ_p(x)))")
    output.append("")
    output.append("Compared against non-prior baselines:")
    output.append("- VPU-Mean:         L = mean(φ(x)) + MixUp")
    output.append("- VPU-NoMixUp-Mean: L = mean(φ(x))")
    output.append("")
    output.append(f"Dataset coverage: {len(df['experiment'].unique())} experiments")
    output.append(f"Seeds analyzed: {sorted(df['seed'].unique())}")
    output.append(f"Statistical tests: Paired t-tests across matched configurations")
    output.append("")

    # Quick comparison table
    for method_prior, method_base, context in COMPARISONS:
        df_comparison = df[df['method'].isin([method_prior, method_base])].copy()

        f1_prior = df_comparison[df_comparison['method'] == method_prior]['test_f1'].mean()
        f1_base = df_comparison[df_comparison['method'] == method_base]['test_f1'].mean()

        t_stat, p_value, cohen_d, n_pairs = compute_paired_ttest(
            df_comparison, method_prior, method_base, 'test_f1'
        )

        output.append(f"{context}:")
        output.append(f"  {method_prior}: F1 = {f1_prior:.4f}")
        output.append(f"  {method_base}: F1 = {f1_base:.4f}")
        output.append(f"  Difference: {((f1_prior - f1_base) / f1_base * 100):+.2f}% (p={p_value:.4f}, n={n_pairs})")
        output.append("")

    return "\n".join(output)


def main():
    """Run complete analysis."""
    print("Loading results...")
    df = load_results()

    print("\nGenerating analysis...")

    # Generate all analysis sections
    sections = [
        generate_summary(df),
        analyze_overall(df),
        analyze_by_dataset(df),
        analyze_by_c(df),
        analyze_by_prior(df),
    ]

    # Combine and save
    full_report = "\n\n".join(sections)

    output_file = RESULTS_DIR / 'PRIOR_VARIANTS_ANALYSIS.md'
    with open(output_file, 'w') as f:
        f.write(full_report)

    print(f"\n✓ Analysis complete!")
    print(f"  Report saved to: {output_file}")
    print(f"\n{generate_summary(df)}")


if __name__ == '__main__':
    main()
