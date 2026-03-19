#!/usr/bin/env python3
"""
Analyze Average Precision (AP) across all VPU variants.

AP is a threshold-independent metric (area under precision-recall curve)
that provides a more robust comparison than F1 at a fixed threshold.

Also analyzes Max F1 - the best F1 achievable with optimal threshold selection.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# Configuration
SEEDS = [42, 123, 456, 789, 2024]
DATASETS = ['MNIST', 'FashionMNIST', 'IMDB', '20News', 'Spambase', 'Mushrooms']
RESULTS_DIR = Path('results')

VPU_METHODS = [
    'oracle_bce',
    'vpu',
    'vpu_mean',
    'vpu_nomixup',
    'vpu_nomixup_mean',
]


def load_ap_results() -> pd.DataFrame:
    """Load AP and Max F1 results for all methods."""
    records = []

    for seed in SEEDS:
        seed_dir = RESULTS_DIR / f'seed_{seed}'
        if not seed_dir.exists():
            continue

        for config_file in seed_dir.glob('*.json'):
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)
            except:
                continue

            experiment = data.get('experiment', '')
            parts = experiment.split('_')
            dataset = parts[0]

            if dataset not in DATASETS:
                continue

            # Extract c and prior
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

            for method_name in VPU_METHODS:
                method_data = data.get('runs', {}).get(method_name)
                if not method_data or 'best' not in method_data:
                    continue

                if 'metrics' not in method_data['best']:
                    continue

                metrics = method_data['best']['metrics']

                record = {
                    'experiment': experiment,
                    'dataset': dataset,
                    'seed': seed,
                    'c': c_value,
                    'prior': prior_value,
                    'method': method_name,
                    'test_f1': metrics.get('test_f1'),
                    'test_ap': metrics.get('test_ap'),
                    'test_max_f1': metrics.get('test_max_f1'),
                    'test_auc': metrics.get('test_auc'),
                }

                records.append(record)

    df = pd.DataFrame(records)

    if len(df) == 0:
        print("WARNING: No AP results found!")
        print("This likely means the updated metrics haven't been computed yet.")
        print("Please re-run some experiments to generate AP/max_f1 data.")
        return df

    print(f"Loaded {len(df)} results")
    print(f"Methods: {sorted(df['method'].unique())}")
    print(f"Datasets: {sorted(df['dataset'].unique())}")

    # Check if AP exists
    ap_count = df['test_ap'].notna().sum()
    total_count = len(df)
    print(f"Results with AP: {ap_count}/{total_count} ({ap_count/total_count*100:.1f}%)")

    return df


def analyze_overall(df: pd.DataFrame) -> str:
    """Overall comparison of AP vs F1."""
    output = []
    output.append("=" * 80)
    output.append("AVERAGE PRECISION (AP): THRESHOLD-INDEPENDENT F1")
    output.append("=" * 80)
    output.append("")
    output.append("AP is the area under the precision-recall curve.")
    output.append("Unlike F1 at a fixed threshold, AP evaluates performance across ALL thresholds.")
    output.append("")

    output.append("Comparison: F1 vs AP vs Max F1")
    output.append("-" * 80)
    output.append(f"{'Method':<25} {'F1 (0.5)':<10} {'AP':<10} {'Max F1':<10} {'AP-F1 Gap':<12}")
    output.append("-" * 80)

    for method in VPU_METHODS:
        df_method = df[df['method'] == method]
        if len(df_method) == 0:
            continue

        f1_mean = df_method['test_f1'].mean()
        ap_mean = df_method['test_ap'].mean()
        max_f1_mean = df_method['test_max_f1'].mean()

        gap = ap_mean - f1_mean if not np.isnan(ap_mean) and not np.isnan(f1_mean) else float('nan')

        output.append(
            f"{method:<25} {f1_mean:>8.4f}   {ap_mean:>8.4f}   "
            f"{max_f1_mean:>8.4f}   {gap:+10.4f}"
        )

    output.append("")
    output.append("Key Insights:")
    output.append("- F1 (0.5): F1 score at fixed threshold (sigmoid(logit) >= 0.5)")
    output.append("- AP: Average Precision (integral over all thresholds)")
    output.append("- Max F1: Best F1 achievable with optimal threshold selection")
    output.append("- Gap > 0: Method could improve F1 with better threshold")
    output.append("- Gap < 0: Method is well-calibrated (fixed threshold is near-optimal)")

    return "\n".join(output)


def analyze_by_dataset(df: pd.DataFrame) -> str:
    """Compare AP vs F1 by dataset."""
    output = []
    output.append("\n" + "=" * 80)
    output.append("AP vs F1 BY DATASET")
    output.append("=" * 80)

    for dataset in DATASETS:
        df_dataset = df[df['dataset'] == dataset]
        if len(df_dataset) == 0:
            continue

        output.append(f"\n{dataset}")
        output.append("-" * 80)
        output.append(f"{'Method':<25} {'F1':<10} {'AP':<10} {'Max F1':<10}")
        output.append("-" * 80)

        for method in VPU_METHODS:
            df_method = df_dataset[df_dataset['method'] == method]
            if len(df_method) == 0:
                continue

            f1_mean = df_method['test_f1'].mean()
            ap_mean = df_method['test_ap'].mean()
            max_f1_mean = df_method['test_max_f1'].mean()

            output.append(f"{method:<25} {f1_mean:>8.4f}   {ap_mean:>8.4f}   {max_f1_mean:>8.4f}")

    return "\n".join(output)


def compare_methods_on_ap(df: pd.DataFrame) -> str:
    """Statistical comparison of VPU methods on AP."""
    output = []
    output.append("\n" + "=" * 80)
    output.append("VPU METHOD COMPARISON ON AVERAGE PRECISION")
    output.append("=" * 80)
    output.append("")

    # Compare VPU-Mean vs VPU on AP
    comparisons = [
        ('vpu_mean', 'vpu', 'With MixUp'),
        ('vpu_nomixup_mean', 'vpu_nomixup', 'Without MixUp'),
    ]

    for method1, method2, context in comparisons:
        output.append(f"{context}: {method1.upper()} vs {method2.upper()}")
        output.append("-" * 80)

        df_comparison = df[df['method'].isin([method1, method2])].copy()

        if len(df_comparison) == 0:
            output.append("No data available")
            continue

        # Paired t-test on AP
        pivot = df_comparison.pivot_table(
            index=['experiment'],
            columns='method',
            values='test_ap'
        ).dropna()

        if len(pivot) == 0 or method1 not in pivot.columns or method2 not in pivot.columns:
            output.append("Insufficient paired data for statistical test")
            continue

        ap1 = pivot[method1].values
        ap2 = pivot[method2].values

        t_stat, p_value = stats.ttest_rel(ap1, ap2)

        mean1 = df_comparison[df_comparison['method'] == method1]['test_ap'].mean()
        mean2 = df_comparison[df_comparison['method'] == method2]['test_ap'].mean()
        diff = mean1 - mean2
        pct_diff = (diff / mean2) * 100 if mean2 != 0 else 0

        output.append(f"{'Metric':<15} {method1:<12} {method2:<12} {'Diff':<12} {'p-value':<12}")
        output.append(f"{'AP':<15} {mean1:>10.4f}   {mean2:>10.4f}   {pct_diff:+10.2f}%   {p_value:>10.4f}")

        # Also compare F1 for reference
        mean1_f1 = df_comparison[df_comparison['method'] == method1]['test_f1'].mean()
        mean2_f1 = df_comparison[df_comparison['method'] == method2]['test_f1'].mean()
        diff_f1 = mean1_f1 - mean2_f1
        pct_diff_f1 = (diff_f1 / mean2_f1) * 100 if mean2_f1 != 0 else 0

        output.append(f"{'F1 (ref)':<15} {mean1_f1:>10.4f}   {mean2_f1:>10.4f}   {pct_diff_f1:+10.2f}%")

        output.append("")

    return "\n".join(output)


def main():
    """Run AP analysis."""
    print("Loading AP results...")
    df = load_ap_results()

    if len(df) == 0:
        print("\nNo data to analyze!")
        return

    # Check if any data has AP
    if df['test_ap'].isna().all():
        print("\nNo AP metrics found in results!")
        print("The updated metrics code needs to run on new experiments.")
        print("Existing results don't have AP/max_f1.")
        return

    print("\nGenerating analysis...")

    sections = [
        analyze_overall(df),
        compare_methods_on_ap(df),
        analyze_by_dataset(df),
    ]

    full_report = "\n\n".join(sections)

    output_file = RESULTS_DIR / 'AVERAGE_PRECISION_ANALYSIS.md'
    with open(output_file, 'w') as f:
        f.write(full_report)

    print(f"\n✓ Analysis complete!")
    print(f"  Report saved to: {output_file}")


if __name__ == '__main__':
    main()
