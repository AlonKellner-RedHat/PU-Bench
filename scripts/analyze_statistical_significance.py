#!/usr/bin/env python3
"""
Analyze statistical significance and noise in VPU variants benchmark results.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


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
                'val_f1': best_metrics.get('val_f1'),
            }

            data.append(row)

    return pd.DataFrame(data)


def analyze_consistency(df):
    """Analyze consistency of results across different conditions."""
    print("=" * 80)
    print("Statistical Robustness Analysis")
    print("=" * 80)
    print()

    # 1. Consistency across datasets
    print("1. CONSISTENCY ACROSS DATASETS")
    print("-" * 60)

    vpu_df = df[df['method'].isin(['vpu', 'vpu_mean'])].copy()

    # For each dataset, calculate VPU vs VPU-Mean difference
    dataset_diffs = []
    for dataset in vpu_df['dataset'].unique():
        ds_data = vpu_df[vpu_df['dataset'] == dataset]
        vpu_scores = ds_data[ds_data['method'] == 'vpu']['test_f1'].values
        vpu_mean_scores = ds_data[ds_data['method'] == 'vpu_mean']['test_f1'].values

        if len(vpu_scores) > 0 and len(vpu_mean_scores) > 0:
            # Average difference for this dataset
            diff = np.mean(vpu_mean_scores) - np.mean(vpu_scores)
            dataset_diffs.append({
                'dataset': dataset,
                'diff': diff,
                'vpu_mean_better': diff > 0,
                'n_comparisons': min(len(vpu_scores), len(vpu_mean_scores))
            })

    diff_df = pd.DataFrame(dataset_diffs)
    print(f"Datasets where VPU-Mean is better: {diff_df['vpu_mean_better'].sum()}/{len(diff_df)}")
    print(f"Mean difference (VPU-Mean - VPU): {diff_df['diff'].mean():.4f}")
    print(f"Std of differences across datasets: {diff_df['diff'].std():.4f}")
    print()
    print("Per-dataset differences:")
    for _, row in diff_df.iterrows():
        sign = "+" if row['diff'] > 0 else ""
        winner = "VPU-Mean" if row['vpu_mean_better'] else "VPU"
        print(f"  {row['dataset']:15s}: {sign}{row['diff']:+7.4f} ({winner} wins, n={row['n_comparisons']})")
    print()

    # 2. Consistency across label frequencies
    print("2. CONSISTENCY ACROSS LABEL FREQUENCIES (c)")
    print("-" * 60)

    c_diffs = []
    for c_val in sorted(vpu_df[vpu_df['prior'].isna()]['c'].unique()):
        c_data = vpu_df[(vpu_df['prior'].isna()) & (vpu_df['c'] == c_val)]
        vpu_scores = c_data[c_data['method'] == 'vpu']['test_f1'].values
        vpu_mean_scores = c_data[c_data['method'] == 'vpu_mean']['test_f1'].values

        if len(vpu_scores) > 0 and len(vpu_mean_scores) > 0:
            diff = np.mean(vpu_mean_scores) - np.mean(vpu_scores)
            c_diffs.append({
                'c': c_val,
                'diff': diff,
                'vpu_mean_better': diff > 0,
                'n_comparisons': min(len(vpu_scores), len(vpu_mean_scores))
            })

    c_df = pd.DataFrame(c_diffs)
    print(f"c-values where VPU-Mean is better: {c_df['vpu_mean_better'].sum()}/{len(c_df)}")
    print(f"Mean difference across c-values: {c_df['diff'].mean():.4f}")
    print(f"Std of differences: {c_df['diff'].std():.4f}")
    print()
    print("Per-c differences:")
    for _, row in c_df.iterrows():
        sign = "+" if row['diff'] > 0 else ""
        winner = "VPU-Mean" if row['vpu_mean_better'] else "VPU"
        print(f"  c={row['c']:4.2f}: {sign}{row['diff']:+7.4f} ({winner} wins, n={row['n_comparisons']})")
    print()

    # 3. Consistency across priors
    print("3. CONSISTENCY ACROSS PRIORS (Class Prevalence)")
    print("-" * 60)

    prior_diffs = []
    for prior_val in sorted(vpu_df[vpu_df['prior'].notna()]['prior'].unique()):
        prior_data = vpu_df[vpu_df['prior'] == prior_val]
        vpu_scores = prior_data[prior_data['method'] == 'vpu']['test_f1'].values
        vpu_mean_scores = prior_data[prior_data['method'] == 'vpu_mean']['test_f1'].values

        if len(vpu_scores) > 0 and len(vpu_mean_scores) > 0:
            diff = np.mean(vpu_mean_scores) - np.mean(vpu_scores)
            prior_diffs.append({
                'prior': prior_val,
                'diff': diff,
                'vpu_mean_better': diff > 0,
                'n_comparisons': min(len(vpu_scores), len(vpu_mean_scores))
            })

    prior_df = pd.DataFrame(prior_diffs)
    print(f"Priors where VPU-Mean is better: {prior_df['vpu_mean_better'].sum()}/{len(prior_df)}")
    print(f"Mean difference across priors: {prior_df['diff'].mean():.4f}")
    print(f"Std of differences: {prior_df['diff'].std():.4f}")
    print()
    print("Per-prior differences:")
    for _, row in prior_df.iterrows():
        sign = "+" if row['diff'] > 0 else ""
        winner = "VPU-Mean" if row['vpu_mean_better'] else "VPU"
        print(f"  prior={row['prior']:3.1f}: {sign}{row['diff']:+7.4f} ({winner} wins, n={row['n_comparisons']})")
    print()


def analyze_effect_sizes(df):
    """Analyze effect sizes to determine if differences are meaningful."""
    print("=" * 80)
    print("Effect Size Analysis")
    print("=" * 80)
    print()

    vpu_df = df[df['method'].isin(['vpu', 'vpu_mean'])].copy()

    # Reshape data for paired comparison
    vpu_scores = vpu_df[vpu_df['method'] == 'vpu'].sort_values(['dataset', 'c', 'prior'])['test_f1'].values
    vpu_mean_scores = vpu_df[vpu_df['method'] == 'vpu_mean'].sort_values(['dataset', 'c', 'prior'])['test_f1'].values

    # Only compare where we have matched pairs
    min_len = min(len(vpu_scores), len(vpu_mean_scores))
    vpu_scores = vpu_scores[:min_len]
    vpu_mean_scores = vpu_mean_scores[:min_len]

    # Calculate effect size (Cohen's d)
    diff = vpu_mean_scores - vpu_scores
    pooled_std = np.sqrt((np.var(vpu_scores) + np.var(vpu_mean_scores)) / 2)
    cohens_d = np.mean(diff) / pooled_std if pooled_std > 0 else 0

    print(f"Cohen's d (VPU-Mean vs VPU): {cohens_d:.4f}")
    print()
    print("Effect size interpretation:")
    print("  |d| < 0.2  : Negligible")
    print("  |d| < 0.5  : Small")
    print("  |d| < 0.8  : Medium")
    print("  |d| >= 0.8 : Large")
    print()

    if abs(cohens_d) < 0.2:
        print(f"→ NEGLIGIBLE effect size: differences are not practically significant")
    elif abs(cohens_d) < 0.5:
        print(f"→ SMALL effect size: differences are modest")
    elif abs(cohens_d) < 0.8:
        print(f"→ MEDIUM effect size: differences are meaningful")
    else:
        print(f"→ LARGE effect size: differences are substantial")
    print()

    # Absolute differences
    print("Magnitude of differences:")
    print(f"  Mean absolute difference: {np.mean(np.abs(diff)):.4f}")
    print(f"  Median absolute difference: {np.median(np.abs(diff)):.4f}")
    print(f"  Max absolute difference: {np.max(np.abs(diff)):.4f}")
    print(f"  Min absolute difference: {np.min(np.abs(diff)):.4f}")
    print()


def analyze_noise_sources(df):
    """Identify potential sources of noise in the results."""
    print("=" * 80)
    print("Noise Source Analysis")
    print("=" * 80)
    print()

    print("1. SINGLE SEED LIMITATION")
    print("-" * 60)
    print("⚠ WARNING: All experiments use a single seed (42)")
    print("  - Cannot estimate variance due to random initialization")
    print("  - Cannot perform proper statistical significance tests")
    print("  - Results may be sensitive to this particular seed")
    print("  - Recommendation: Run with multiple seeds (3-10) for robustness")
    print()

    print("2. WITHIN-DATASET VARIANCE")
    print("-" * 60)

    # For datasets with multiple configs, check variance
    vpu_df = df[df['method'].isin(['vpu', 'vpu_mean'])].copy()

    for dataset in vpu_df['dataset'].unique():
        ds_data = vpu_df[vpu_df['dataset'] == dataset]

        for method in ['vpu', 'vpu_mean']:
            method_data = ds_data[ds_data['method'] == method]['test_f1']
            if len(method_data) > 1:
                print(f"{dataset:15s} - {method:10s}: mean={method_data.mean():.4f}, std={method_data.std():.4f}, cv={method_data.std()/method_data.mean():.3f}")
    print()

    print("3. VALIDATION-TEST CONSISTENCY")
    print("-" * 60)
    print("Checking if val_f1 and test_f1 are correlated (good generalization):")

    for method in ['vpu', 'vpu_mean']:
        method_data = df[df['method'] == method].dropna(subset=['val_f1', 'test_f1'])
        if len(method_data) > 0:
            corr = np.corrcoef(method_data['val_f1'], method_data['test_f1'])[0, 1]
            print(f"  {method:10s}: val-test correlation = {corr:.4f}")
    print()


def generate_confidence_assessment(df):
    """Provide overall confidence assessment."""
    print("=" * 80)
    print("OVERALL CONFIDENCE ASSESSMENT")
    print("=" * 80)
    print()

    vpu_df = df[df['method'].isin(['vpu', 'vpu_mean'])].copy()

    # Reshape for paired comparison
    vpu_scores = vpu_df[vpu_df['method'] == 'vpu'].sort_values(['dataset', 'c', 'prior'])['test_f1'].values
    vpu_mean_scores = vpu_df[vpu_df['method'] == 'vpu_mean'].sort_values(['dataset', 'c', 'prior'])['test_f1'].values

    min_len = min(len(vpu_scores), len(vpu_mean_scores))
    vpu_scores = vpu_scores[:min_len]
    vpu_mean_scores = vpu_mean_scores[:min_len]

    diff = vpu_mean_scores - vpu_scores

    # Calculate metrics
    vpu_mean_win_rate = np.mean(vpu_mean_scores > vpu_scores)
    mean_diff = np.mean(diff)
    median_diff = np.median(diff)
    pooled_std = np.sqrt((np.var(vpu_scores) + np.var(vpu_mean_scores)) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

    print(f"VPU-Mean win rate: {vpu_mean_win_rate:.1%} ({int(vpu_mean_win_rate * len(diff))}/{len(diff)} experiments)")
    print(f"Mean advantage: {mean_diff:+.4f} F1 points")
    print(f"Median advantage: {median_diff:+.4f} F1 points")
    print(f"Effect size (Cohen's d): {cohens_d:.4f}")
    print()

    print("Confidence levels:")
    print("-" * 60)

    confidence_scores = []

    # 1. Win rate consistency
    if vpu_mean_win_rate >= 0.7:
        print("✓ HIGH confidence: VPU-Mean wins >70% of experiments")
        confidence_scores.append(3)
    elif vpu_mean_win_rate >= 0.6:
        print("✓ MEDIUM confidence: VPU-Mean wins >60% of experiments")
        confidence_scores.append(2)
    else:
        print("✗ LOW confidence: Win rate is marginal (<60%)")
        confidence_scores.append(1)

    # 2. Effect size
    if abs(cohens_d) >= 0.5:
        print("✓ HIGH confidence: Effect size is meaningful (|d| >= 0.5)")
        confidence_scores.append(3)
    elif abs(cohens_d) >= 0.2:
        print("~ MEDIUM confidence: Effect size is small (0.2 <= |d| < 0.5)")
        confidence_scores.append(2)
    else:
        print("✗ LOW confidence: Effect size is negligible (|d| < 0.2)")
        confidence_scores.append(1)

    # 3. Consistency across datasets
    dataset_consistency = []
    for dataset in vpu_df['dataset'].unique():
        ds_data = vpu_df[vpu_df['dataset'] == dataset]
        vpu_ds = ds_data[ds_data['method'] == 'vpu']['test_f1'].mean()
        vpu_mean_ds = ds_data[ds_data['method'] == 'vpu_mean']['test_f1'].mean()
        dataset_consistency.append(vpu_mean_ds > vpu_ds)

    consistency_rate = np.mean(dataset_consistency)
    if consistency_rate >= 0.7:
        print(f"✓ HIGH confidence: Consistent across {consistency_rate:.0%} of datasets")
        confidence_scores.append(3)
    elif consistency_rate >= 0.5:
        print(f"~ MEDIUM confidence: Consistent across {consistency_rate:.0%} of datasets")
        confidence_scores.append(2)
    else:
        print(f"✗ LOW confidence: Inconsistent across datasets ({consistency_rate:.0%})")
        confidence_scores.append(1)

    # 4. Single seed limitation
    print("✗ CRITICAL LIMITATION: Only one random seed tested")
    print("  → Cannot assess variance due to initialization")
    print("  → Results may not generalize to other seeds")
    confidence_scores.append(1)

    print()
    avg_confidence = np.mean(confidence_scores)

    print("=" * 60)
    print(f"OVERALL CONFIDENCE: ", end="")
    if avg_confidence >= 2.5:
        print("HIGH ✓")
        print("The preference for VPU-Mean appears robust across multiple dimensions,")
        print("but single-seed limitation prevents strong statistical claims.")
    elif avg_confidence >= 1.8:
        print("MEDIUM ~")
        print("VPU-Mean shows consistent advantage, but limited by single seed and")
        print("small effect sizes in some conditions.")
    else:
        print("LOW ✗")
        print("Results are noisy and inconclusive. Multiple seeds and more experiments")
        print("are needed for confident recommendations.")
    print()

    print("RECOMMENDATIONS:")
    print("1. ⚠ Run experiments with 3-10 different random seeds")
    print("2. Consider statistical significance tests (paired t-test, Wilcoxon)")
    print("3. Report confidence intervals for mean differences")
    print("4. For critical applications, use cross-validation or bootstrap")


def main():
    print("Loading benchmark results...")
    df = load_results()
    print(f"Loaded {len(df)} results")
    print()

    analyze_consistency(df)
    analyze_effect_sizes(df)
    analyze_noise_sources(df)
    generate_confidence_assessment(df)


if __name__ == '__main__':
    main()
