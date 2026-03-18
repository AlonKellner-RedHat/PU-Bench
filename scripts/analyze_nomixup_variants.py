#!/usr/bin/env python3
"""
Analysis of VPU vs VPU-Mean WITHOUT MixUp augmentation.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


def load_nomixup_results():
    """Load results for no-mixup variants."""
    seeds = [42, 123, 456, 789, 2024]
    target_methods = {'oracle_bce', 'vpu_nomixup', 'vpu_nomixup_mean'}
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

            # Parse config
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
                    'test_f1': best_metrics.get('test_f1'),
                    'test_auc': best_metrics.get('test_auc'),
                    'test_accuracy': best_metrics.get('test_accuracy'),
                    'test_precision': best_metrics.get('test_precision'),
                    'test_recall': best_metrics.get('test_recall'),
                    'test_anice': best_metrics.get('test_anice'),
                    'test_snice': best_metrics.get('test_snice'),
                    'test_ece': best_metrics.get('test_ece'),
                    'test_mce': best_metrics.get('test_mce'),
                    'test_brier': best_metrics.get('test_brier'),
                }

                data.append(row)

    return pd.DataFrame(data)


def comprehensive_nomixup_summary(df):
    """Generate comprehensive summary for no-mixup variants."""
    print("=" * 80)
    print("COMPREHENSIVE ANALYSIS: VPU-NoMixUp vs VPU-NoMixUp-Mean")
    print("=" * 80)
    print()

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

    vpu_df = df[df['method'].isin(['vpu_nomixup', 'vpu_nomixup_mean'])].copy()

    for category, metric_dict in metrics.items():
        print(f"\n{category}:")
        print("-" * 80)
        print(f"{'Metric':<20s} {'NoMixUp':>12s} {'NoMixUp-Mean':>12s} {'Advantage':>12s} {'p-value':>10s} {'Winner':>15s}")
        print("-" * 80)

        for metric_col, (metric_name, lower_is_better) in metric_dict.items():
            vpu_vals = vpu_df[vpu_df['method'] == 'vpu_nomixup'][metric_col].dropna().values
            vpu_mean_vals = vpu_df[vpu_df['method'] == 'vpu_nomixup_mean'][metric_col].dropna().values

            if len(vpu_vals) > 5 and len(vpu_mean_vals) > 5:
                t_stat, p_value = stats.ttest_ind(vpu_mean_vals, vpu_vals, equal_var=False)

                if lower_is_better:
                    winner = "NoMixUp-Mean" if np.mean(vpu_mean_vals) < np.mean(vpu_vals) else "NoMixUp"
                    advantage = np.mean(vpu_vals) - np.mean(vpu_mean_vals)
                else:
                    winner = "NoMixUp-Mean" if np.mean(vpu_mean_vals) > np.mean(vpu_vals) else "NoMixUp"
                    advantage = np.mean(vpu_mean_vals) - np.mean(vpu_vals)

                sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""

                print(f"{metric_name:<20s} {np.mean(vpu_vals):>12.4f} {np.mean(vpu_mean_vals):>12.4f} "
                      f"{advantage:>+12.4f} {p_value:>10.4f}{sig:3s} {winner:>15s}")


def analyze_by_dataset_nomixup(df):
    """Analyze performance by dataset for no-mixup variants."""
    print("\n" + "=" * 80)
    print("PERFORMANCE BY DATASET (No MixUp)")
    print("=" * 80)
    print()

    vpu_df = df[df['method'].isin(['vpu_nomixup', 'vpu_nomixup_mean'])].copy()

    for dataset in sorted(vpu_df['dataset'].unique()):
        ds_data = vpu_df[vpu_df['dataset'] == dataset]

        vpu_scores = ds_data[ds_data['method'] == 'vpu_nomixup']['test_f1'].dropna().values
        vpu_mean_scores = ds_data[ds_data['method'] == 'vpu_nomixup_mean']['test_f1'].dropna().values

        if len(vpu_scores) > 1 and len(vpu_mean_scores) > 1:
            t_stat, p_value = stats.ttest_ind(vpu_mean_scores, vpu_scores, equal_var=False)

            pooled_std = np.sqrt((np.var(vpu_scores, ddof=1) + np.var(vpu_mean_scores, ddof=1)) / 2)
            cohens_d = (np.mean(vpu_mean_scores) - np.mean(vpu_scores)) / pooled_std if pooled_std > 0 else 0

            winner = "NoMixUp-Mean" if np.mean(vpu_mean_scores) > np.mean(vpu_scores) else "NoMixUp"
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

            print(f"{dataset:15s}:")
            print(f"  NoMixUp:      {np.mean(vpu_scores):.4f} ± {np.std(vpu_scores, ddof=1):.4f} (n={len(vpu_scores)})")
            print(f"  NoMixUp-Mean: {np.mean(vpu_mean_scores):.4f} ± {np.std(vpu_mean_scores, ddof=1):.4f} (n={len(vpu_mean_scores)})")
            print(f"  Difference: {np.mean(vpu_mean_scores) - np.mean(vpu_scores):+.4f}")
            print(f"  Cohen's d: {cohens_d:+.3f}")
            print(f"  p-value: {p_value:.4f} {sig}")
            print(f"  → {winner} wins")
            print()


def analyze_by_label_frequency_nomixup(df):
    """Analyze by label frequency for no-mixup variants."""
    print("=" * 80)
    print("PERFORMANCE BY LABEL FREQUENCY (c) - No MixUp")
    print("=" * 80)
    print()

    vpu_df = df[df['method'].isin(['vpu_nomixup', 'vpu_nomixup_mean'])].copy()

    # Filter to vary_c experiments
    vary_c = vpu_df[vpu_df['prior'].isna()].copy()

    print(f"{'c value':<10s} {'NoMixUp':>12s} {'NoMixUp-Mean':>12s} {'Diff':>10s} {'p-value':>10s} {'Winner':>15s}")
    print("-" * 80)

    for c_val in sorted(vary_c['c'].unique()):
        c_data = vary_c[vary_c['c'] == c_val]

        vpu_scores = c_data[c_data['method'] == 'vpu_nomixup']['test_f1'].dropna().values
        vpu_mean_scores = c_data[c_data['method'] == 'vpu_nomixup_mean']['test_f1'].dropna().values

        if len(vpu_scores) > 2 and len(vpu_mean_scores) > 2:
            t_stat, p_value = stats.ttest_ind(vpu_mean_scores, vpu_scores, equal_var=False)

            winner = "NoMixUp-Mean" if np.mean(vpu_mean_scores) > np.mean(vpu_scores) else "NoMixUp"
            diff = np.mean(vpu_mean_scores) - np.mean(vpu_scores)
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""

            print(f"c={c_val:<7.2f} {np.mean(vpu_scores):>12.4f} {np.mean(vpu_mean_scores):>12.4f} "
                  f"{diff:>+10.4f} {p_value:>10.4f}{sig:3s} {winner:>15s}")

    print()


def analyze_calibration_nomixup(df):
    """Analyze calibration for no-mixup variants."""
    print("=" * 80)
    print("CALIBRATION ANALYSIS (No MixUp)")
    print("=" * 80)
    print()

    vpu_df = df[df['method'].isin(['vpu_nomixup', 'vpu_nomixup_mean'])].copy()

    calibration_metrics = {
        'test_anice': 'A-NICE',
        'test_snice': 'S-NICE',
        'test_ece': 'ECE',
        'test_mce': 'MCE',
        'test_brier': 'Brier Score',
    }

    for metric_col, metric_name in calibration_metrics.items():
        vpu_scores = vpu_df[vpu_df['method'] == 'vpu_nomixup'][metric_col].dropna().values
        vpu_mean_scores = vpu_df[vpu_df['method'] == 'vpu_nomixup_mean'][metric_col].dropna().values

        if len(vpu_scores) > 5 and len(vpu_mean_scores) > 5:
            t_stat, p_value = stats.ttest_ind(vpu_mean_scores, vpu_scores, equal_var=False)

            winner = "NoMixUp" if np.mean(vpu_scores) < np.mean(vpu_mean_scores) else "NoMixUp-Mean"
            improvement = (np.mean(vpu_scores) - np.mean(vpu_mean_scores)) / np.mean(vpu_scores) * 100
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

            print(f"{metric_name} (lower is better):")
            print(f"  NoMixUp:      {np.mean(vpu_scores):.4f} ± {np.std(vpu_scores, ddof=1):.4f}")
            print(f"  NoMixUp-Mean: {np.mean(vpu_mean_scores):.4f} ± {np.std(vpu_mean_scores, ddof=1):.4f}")
            print(f"  Improvement: {improvement:+.2f}% ({winner} is better)")
            print(f"  p-value: {p_value:.4f} {sig}")
            print()


def compare_with_mixup_results(df_nomixup, df_all):
    """Compare no-mixup results with mixup results."""
    print("=" * 80)
    print("COMPARISON: Impact of Removing MixUp")
    print("=" * 80)
    print()

    # Get with-mixup results
    mixup_df = df_all[df_all['method'].isin(['vpu', 'vpu_mean'])].copy()

    print("Performance Degradation from Removing MixUp:")
    print("-" * 80)
    print(f"{'Method':<20s} {'With MixUp':>12s} {'No MixUp':>12s} {'Loss':>10s} {'Loss %':>10s}")
    print("-" * 80)

    # VPU variants
    vpu_mixup = mixup_df[mixup_df['method'] == 'vpu']['test_f1'].mean()
    vpu_nomixup = df_nomixup[df_nomixup['method'] == 'vpu_nomixup']['test_f1'].mean()
    loss = vpu_nomixup - vpu_mixup
    loss_pct = (loss / vpu_mixup) * 100

    print(f"{'VPU (log-of-mean)':<20s} {vpu_mixup:>12.4f} {vpu_nomixup:>12.4f} {loss:>+10.4f} {loss_pct:>+9.2f}%")

    # VPU-Mean variants
    vpu_mean_mixup = mixup_df[mixup_df['method'] == 'vpu_mean']['test_f1'].mean()
    vpu_mean_nomixup = df_nomixup[df_nomixup['method'] == 'vpu_nomixup_mean']['test_f1'].mean()
    loss = vpu_mean_nomixup - vpu_mean_mixup
    loss_pct = (loss / vpu_mean_mixup) * 100

    print(f"{'VPU-Mean (mean)':<20s} {vpu_mean_mixup:>12.4f} {vpu_mean_nomixup:>12.4f} {loss:>+10.4f} {loss_pct:>+9.2f}%")
    print()

    print("Key Finding:")
    if abs(vpu_nomixup - vpu_mixup) > abs(vpu_mean_nomixup - vpu_mean_mixup):
        print("  → VPU (log-of-mean) is MORE SENSITIVE to MixUp removal")
        print(f"    VPU loses {abs((vpu_nomixup - vpu_mixup) / vpu_mixup * 100):.2f}% vs VPU-Mean loses {abs((vpu_mean_nomixup - vpu_mean_mixup) / vpu_mean_mixup * 100):.2f}%")
    else:
        print("  → VPU-Mean is MORE SENSITIVE to MixUp removal")
        print(f"    VPU-Mean loses {abs((vpu_mean_nomixup - vpu_mean_mixup) / vpu_mean_mixup * 100):.2f}% vs VPU loses {abs((vpu_nomixup - vpu_mixup) / vpu_mixup * 100):.2f}%")
    print()


def final_nomixup_recommendations(df):
    """Generate final recommendations for no-mixup scenario."""
    print("=" * 80)
    print("FINAL RECOMMENDATIONS (Without MixUp)")
    print("=" * 80)
    print()

    vpu_df = df[df['method'].isin(['vpu_nomixup', 'vpu_nomixup_mean'])].copy()

    # Overall comparison
    vpu_all = vpu_df[vpu_df['method'] == 'vpu_nomixup']['test_f1'].dropna().values
    vpu_mean_all = vpu_df[vpu_df['method'] == 'vpu_nomixup_mean']['test_f1'].dropna().values

    t_stat, p_value = stats.ttest_ind(vpu_mean_all, vpu_all, equal_var=False)
    mean_diff = np.mean(vpu_mean_all) - np.mean(vpu_all)
    pooled_std = np.sqrt((np.var(vpu_all, ddof=1) + np.var(vpu_mean_all, ddof=1)) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

    print(f"Overall Performance (F1):")
    print(f"  NoMixUp:      {np.mean(vpu_all):.4f} ± {np.std(vpu_all, ddof=1):.4f}")
    print(f"  NoMixUp-Mean: {np.mean(vpu_mean_all):.4f} ± {np.std(vpu_mean_all, ddof=1):.4f}")
    print(f"  Difference: {mean_diff:+.4f}")
    print(f"  Cohen's d: {cohens_d:+.3f}")
    print(f"  p-value: {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
    print()

    if p_value < 0.05:
        winner = "NoMixUp-Mean" if mean_diff > 0 else "NoMixUp"
        print(f"✓ {winner} is SIGNIFICANTLY better overall (p={p_value:.4f})")
    else:
        print(f"→ No significant overall difference (p={p_value:.4f})")
        print(f"  Methods are statistically comparable without MixUp")

    print()
    print("Recommended Choice:")
    print("-" * 80)

    # Check calibration
    cal_metrics = ['test_anice', 'test_ece', 'test_mce']
    cal_wins = 0

    for metric in cal_metrics:
        vpu_cal = vpu_df[vpu_df['method'] == 'vpu_nomixup'][metric].dropna().mean()
        vpu_mean_cal = vpu_df[vpu_df['method'] == 'vpu_nomixup_mean'][metric].dropna().mean()
        if vpu_cal < vpu_mean_cal:  # Lower is better for calibration
            cal_wins += 1

    if mean_diff > 0.01 and p_value < 0.1:
        print("✓ Use VPU-NoMixUp-Mean:")
        print(f"  - Better F1 performance ({mean_diff:+.4f}, p={p_value:.4f})")
        print(f"  - {np.mean(vpu_mean_all):.4f} vs {np.mean(vpu_all):.4f}")
    elif cal_wins >= 2:
        print("✓ Use VPU-NoMixUp:")
        print(f"  - Better calibration ({cal_wins}/3 metrics)")
        print(f"  - F1 performance is comparable ({mean_diff:+.4f}, p={p_value:.4f})")
    else:
        print("✓ Either method is acceptable:")
        print(f"  - Performance difference is negligible ({mean_diff:+.4f}, p={p_value:.4f})")
        print(f"  - Choose based on specific application needs")

    print()


def main():
    print("Loading results for NO-MIXUP variants...")
    df_nomixup = load_nomixup_results()

    seeds = df_nomixup['seed'].unique()
    print(f"Loaded {len(df_nomixup)} results from {len(seeds)} seeds: {sorted(seeds)}")
    print(f"Datasets: {sorted(df_nomixup['dataset'].unique())}")
    print()

    comprehensive_nomixup_summary(df_nomixup)
    analyze_by_dataset_nomixup(df_nomixup)
    analyze_by_label_frequency_nomixup(df_nomixup)
    analyze_calibration_nomixup(df_nomixup)

    # Load all results for comparison
    print("\n" + "=" * 80)
    print("Loading WITH-MIXUP results for comparison...")

    # Reuse the loading function from the other script
    from analyze_multiseed_all_metrics import load_multiseed_results_all_metrics
    df_all = load_multiseed_results_all_metrics()

    compare_with_mixup_results(df_nomixup, df_all)
    final_nomixup_recommendations(df_nomixup)


if __name__ == '__main__':
    main()
