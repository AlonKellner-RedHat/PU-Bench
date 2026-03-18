#!/usr/bin/env python3
"""
Analyze VPU variants benchmark results to compare:
1. VPU vs VPU-Mean (log-of-mean vs mean variance reduction)
2. With vs without MixUp data augmentation
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd


def load_results():
    """Load all benchmark results."""
    results_dir = Path('results/seed_42')

    target_methods = {'oracle_bce', 'vpu', 'vpu_mean', 'vpu_nomixup', 'vpu_nomixup_mean'}
    target_datasets = {'MNIST', 'FashionMNIST', 'IMDB', '20News', 'Connect4', 'Spambase', 'Mushrooms'}

    data = []

    for result_file in results_dir.glob('*case-control_random*.json'):
        dataset_name = result_file.name.split('_')[0]
        if dataset_name not in target_datasets:
            continue

        with open(result_file) as f:
            result_data = json.load(f)

        # Parse experiment config from filename
        parts = result_file.stem.split('_')
        config = {'dataset': dataset_name}

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
            elif part.startswith('seed'):
                try:
                    config['seed'] = int(part[4:])
                except ValueError:
                    pass

        # Extract metrics for each method
        for method_name, method_data in result_data.get('runs', {}).items():
            if method_name not in target_methods:
                continue

            best_metrics = method_data.get('best', {}).get('metrics', {})

            row = {
                'dataset': dataset_name,
                'method': method_name,
                'c': config.get('c'),
                'prior': config.get('prior'),
                'seed': config.get('seed', 42),
                'test_f1': best_metrics.get('test_f1'),
                'test_auc': best_metrics.get('test_auc'),
                'test_accuracy': best_metrics.get('test_accuracy'),
                'test_error': best_metrics.get('test_error'),
                'test_precision': best_metrics.get('test_precision'),
                'test_recall': best_metrics.get('test_recall'),
                'duration': method_data.get('timing', {}).get('duration_seconds'),
            }

            data.append(row)

    return pd.DataFrame(data)


def analyze_vpu_vs_vpu_mean(df):
    """Compare VPU (log-of-mean) vs VPU-Mean (mean)."""
    print("=" * 80)
    print("Analysis 1: VPU vs VPU-Mean (Variance Reduction Strategy)")
    print("=" * 80)
    print()

    # Filter to only VPU variants with mixup
    vpu_df = df[df['method'].isin(['vpu', 'vpu_mean'])].copy()

    # Overall comparison
    print("Overall Performance (Test F1):")
    print("-" * 60)
    overall = vpu_df.groupby('method')['test_f1'].agg(['mean', 'std', 'count'])
    print(overall)
    print()

    # Pairwise comparison
    vpu_scores = vpu_df[vpu_df['method'] == 'vpu']['test_f1'].values
    vpu_mean_scores = vpu_df[vpu_df['method'] == 'vpu_mean']['test_f1'].values

    if len(vpu_scores) > 0 and len(vpu_mean_scores) > 0:
        vpu_wins = np.sum(vpu_scores > vpu_mean_scores)
        vpu_mean_wins = np.sum(vpu_mean_scores > vpu_scores)
        ties = len(vpu_scores) - vpu_wins - vpu_mean_wins

        print(f"Head-to-head (matched experiments):")
        print(f"  VPU wins: {vpu_wins}")
        print(f"  VPU-Mean wins: {vpu_mean_wins}")
        print(f"  Ties: {ties}")
        print()

    # By dataset
    print("Performance by Dataset (Test F1):")
    print("-" * 60)
    by_dataset = vpu_df.groupby(['dataset', 'method'])['test_f1'].mean().unstack()
    by_dataset['Winner'] = by_dataset.apply(lambda row: 'VPU' if row['vpu'] > row['vpu_mean'] else 'VPU-Mean', axis=1)
    by_dataset['Margin'] = abs(by_dataset['vpu'] - by_dataset['vpu_mean'])
    print(by_dataset.to_string())
    print()

    # By label frequency (c)
    print("Performance by Label Frequency c (Test F1):")
    print("-" * 60)
    by_c = vpu_df[vpu_df['prior'].isna()].groupby(['c', 'method'])['test_f1'].mean().unstack()
    if not by_c.empty:
        by_c['Winner'] = by_c.apply(lambda row: 'VPU' if row.get('vpu', 0) > row.get('vpu_mean', 0) else 'VPU-Mean', axis=1)
        by_c['Margin'] = abs(by_c.get('vpu', 0) - by_c.get('vpu_mean', 0))
        print(by_c.to_string())
        print()

    # By prior (class prevalence)
    print("Performance by Prior (Class Prevalence) (Test F1):")
    print("-" * 60)
    by_prior = vpu_df[vpu_df['prior'].notna()].groupby(['prior', 'method'])['test_f1'].mean().unstack()
    if not by_prior.empty:
        by_prior['Winner'] = by_prior.apply(lambda row: 'VPU' if row.get('vpu', 0) > row.get('vpu_mean', 0) else 'VPU-Mean', axis=1)
        by_prior['Margin'] = abs(by_prior.get('vpu', 0) - by_prior.get('vpu_mean', 0))
        print(by_prior.to_string())
        print()


def analyze_mixup_effect(df):
    """Compare with vs without MixUp augmentation."""
    print("=" * 80)
    print("Analysis 2: Effect of MixUp Data Augmentation")
    print("=" * 80)
    print()

    # Compare VPU vs VPU-NoMixUp
    print("VPU (with MixUp) vs VPU-NoMixUp:")
    print("-" * 60)
    vpu_mixup_df = df[df['method'].isin(['vpu', 'vpu_nomixup'])].copy()

    overall = vpu_mixup_df.groupby('method')['test_f1'].agg(['mean', 'std', 'count'])
    print(overall)
    print()

    by_dataset = vpu_mixup_df.groupby(['dataset', 'method'])['test_f1'].mean().unstack()
    if 'vpu' in by_dataset.columns and 'vpu_nomixup' in by_dataset.columns:
        by_dataset['Winner'] = by_dataset.apply(lambda row: 'With MixUp' if row['vpu'] > row['vpu_nomixup'] else 'No MixUp', axis=1)
        by_dataset['Margin'] = abs(by_dataset['vpu'] - by_dataset['vpu_nomixup'])
        print(by_dataset.to_string())
        print()

    # Compare VPU-Mean vs VPU-Mean-NoMixUp
    print("VPU-Mean (with MixUp) vs VPU-Mean-NoMixUp:")
    print("-" * 60)
    vpu_mean_mixup_df = df[df['method'].isin(['vpu_mean', 'vpu_nomixup_mean'])].copy()

    overall = vpu_mean_mixup_df.groupby('method')['test_f1'].agg(['mean', 'std', 'count'])
    print(overall)
    print()

    by_dataset = vpu_mean_mixup_df.groupby(['dataset', 'method'])['test_f1'].mean().unstack()
    if 'vpu_mean' in by_dataset.columns and 'vpu_nomixup_mean' in by_dataset.columns:
        by_dataset['Winner'] = by_dataset.apply(lambda row: 'With MixUp' if row['vpu_mean'] > row['vpu_nomixup_mean'] else 'No MixUp', axis=1)
        by_dataset['Margin'] = abs(by_dataset['vpu_mean'] - by_dataset['vpu_nomixup_mean'])
        print(by_dataset.to_string())
        print()


def analyze_extreme_conditions(df):
    """Analyze performance under extreme c and prior values."""
    print("=" * 80)
    print("Analysis 3: Performance Under Extreme Conditions")
    print("=" * 80)
    print()

    # Extreme label frequencies (c <= 0.05 or c >= 0.7)
    print("Extreme Label Frequencies (c <= 0.05 or c >= 0.7):")
    print("-" * 60)
    extreme_c = df[(df['prior'].isna()) & ((df['c'] <= 0.05) | (df['c'] >= 0.7))].copy()

    if not extreme_c.empty:
        by_method = extreme_c[extreme_c['method'].isin(['vpu', 'vpu_mean'])].groupby('method')['test_f1'].agg(['mean', 'std', 'count'])
        print(by_method)
        print()

    # Extreme priors (prior <= 0.1 or prior >= 0.7)
    print("Extreme Priors (prior <= 0.1 or prior >= 0.7):")
    print("-" * 60)
    extreme_prior = df[(df['prior'].notna()) & ((df['prior'] <= 0.1) | (df['prior'] >= 0.7))].copy()

    if not extreme_prior.empty:
        by_method = extreme_prior[extreme_prior['method'].isin(['vpu', 'vpu_mean'])].groupby('method')['test_f1'].agg(['mean', 'std', 'count'])
        print(by_method)
        print()

        by_prior = extreme_prior[extreme_prior['method'].isin(['vpu', 'vpu_mean'])].groupby(['prior', 'method'])['test_f1'].mean().unstack()
        print("By specific prior value:")
        print(by_prior.to_string())
        print()


def analyze_vs_oracle(df):
    """Compare VPU variants against Oracle BCE."""
    print("=" * 80)
    print("Analysis 4: Performance vs Oracle BCE (Upper Bound)")
    print("=" * 80)
    print()

    oracle_df = df[df['method'] == 'oracle_bce'].copy()
    vpu_variants = df[df['method'].isin(['vpu', 'vpu_mean', 'vpu_nomixup', 'vpu_nomixup_mean'])].copy()

    # Merge on dataset/c/prior to get matched comparisons
    oracle_df = oracle_df.rename(columns={'test_f1': 'oracle_f1'})
    merged = vpu_variants.merge(
        oracle_df[['dataset', 'c', 'prior', 'oracle_f1']],
        on=['dataset', 'c', 'prior'],
        how='left'
    )

    merged['gap_from_oracle'] = merged['oracle_f1'] - merged['test_f1']

    print("Average Gap from Oracle (lower is better):")
    print("-" * 60)
    gap_summary = merged.groupby('method')['gap_from_oracle'].agg(['mean', 'std', 'count'])
    print(gap_summary.sort_values('mean'))
    print()

    print("Gap from Oracle by Dataset:")
    print("-" * 60)
    gap_by_dataset = merged.groupby(['dataset', 'method'])['gap_from_oracle'].mean().unstack()
    print(gap_by_dataset.to_string())
    print()


def generate_summary_table(df):
    """Generate a comprehensive summary table."""
    print("=" * 80)
    print("Summary: Best Method Recommendations")
    print("=" * 80)
    print()

    vpu_variants = df[df['method'].isin(['vpu', 'vpu_mean', 'vpu_nomixup', 'vpu_nomixup_mean'])].copy()

    # Overall winner
    overall = vpu_variants.groupby('method')['test_f1'].mean().sort_values(ascending=False)
    print(f"Overall Best Method: {overall.index[0]} (F1: {overall.iloc[0]:.4f})")
    print()

    # Winner by dataset
    print("Best Method by Dataset:")
    print("-" * 60)
    by_dataset = vpu_variants.groupby(['dataset', 'method'])['test_f1'].mean().unstack()
    winners = by_dataset.idxmax(axis=1)
    scores = by_dataset.max(axis=1)

    for dataset in winners.index:
        print(f"  {dataset:15s}: {winners[dataset]:20s} (F1: {scores[dataset]:.4f})")
    print()

    # Winner by label frequency
    print("Best Method by Label Frequency (c):")
    print("-" * 60)
    by_c = vpu_variants[vpu_variants['prior'].isna()].groupby(['c', 'method'])['test_f1'].mean().unstack()
    if not by_c.empty:
        winners = by_c.idxmax(axis=1)
        scores = by_c.max(axis=1)

        for c_val in winners.index:
            print(f"  c={c_val:4.2f}: {winners[c_val]:20s} (F1: {scores[c_val]:.4f})")
        print()


def main():
    print("Loading benchmark results...")
    df = load_results()
    print(f"Loaded {len(df)} results from {df['dataset'].nunique()} datasets")
    print(f"Methods: {sorted(df['method'].unique())}")
    print()

    analyze_vpu_vs_vpu_mean(df)
    analyze_mixup_effect(df)
    analyze_extreme_conditions(df)
    analyze_vs_oracle(df)
    generate_summary_table(df)

    # Save detailed results to CSV
    output_file = 'results/vpu_variants_analysis.csv'
    df.to_csv(output_file, index=False)
    print(f"Detailed results saved to: {output_file}")


if __name__ == '__main__':
    main()
