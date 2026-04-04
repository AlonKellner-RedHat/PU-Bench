#!/usr/bin/env python3
"""Compare 8 VPU Variants: 4 No-Mixup + 4 Mixup

No-Mixup variants:
1. vpu_nomixup - Baseline (no mean, no prior)
2. vpu_nomixup_mean_prior (1.0) - With mean (prior=1.0)
3. vpu_nomixup_mean_prior (auto) - With mean and true prior
4. vpu_nomixup_mean_prior (0.5) - With mean and fixed prior=0.5

Mixup variants:
5. vpu - Classic VPU with mixup
6. vpu_mean - With mean + mixup (prior=1.0)
7. vpu_mean_prior (auto) - With mean + mixup + true prior
8. vpu_mean_prior (0.5) - With mean + mixup + prior=0.5
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
            method_prior = hyperparams.get('method_prior')

            if all([dataset, seed, c, true_prior_actual]):
                results.append({
                    'method': method_name,
                    'method_prior': method_prior if method_prior is not None else 'auto',
                    'dataset': dataset,
                    'seed': seed,
                    'c': c,
                    'true_prior': true_prior_actual,
                    'test_ap': best_metrics.get('test_ap'),
                    'test_f1': best_metrics.get('test_f1'),
                    'test_auc': best_metrics.get('test_auc'),
                    'test_ece': best_metrics.get('test_ece'),
                    'test_oracle_ce': best_metrics.get('test_oracle_ce'),
                    'convergence_epoch': method_data.get('best', {}).get('epoch'),
                })

        except Exception as e:
            continue

    return pd.DataFrame(results)


def load_all_methods(results_dir="results_cartesian"):
    """Load all 8 method variants"""

    print("Loading method results...")

    # NO-MIXUP VARIANTS
    # 1. vpu_nomixup (baseline)
    df_nomixup_baseline = load_method_results(results_dir, 'vpu_nomixup')
    df_nomixup_baseline['method_label'] = 'no-mixup: baseline'
    df_nomixup_baseline['group'] = 'no-mixup'
    print(f"  vpu_nomixup (baseline): {len(df_nomixup_baseline)} experiments")

    # 2. vpu_nomixup_mean_prior with method_prior=1.0
    df_nomixup_mean = load_method_results(results_dir, 'vpu_nomixup_mean_prior', method_prior_filter=1.0)
    df_nomixup_mean['method_label'] = 'no-mixup: prior=1.0'
    df_nomixup_mean['group'] = 'no-mixup'
    print(f"  vpu_nomixup_mean_prior (1.0): {len(df_nomixup_mean)} experiments")

    # 3. vpu_nomixup_mean_prior with auto (true prior)
    df_nomixup_auto = load_method_results(results_dir, 'vpu_nomixup_mean_prior', method_prior_filter="auto")
    df_nomixup_auto['method_label'] = 'no-mixup: prior=auto'
    df_nomixup_auto['group'] = 'no-mixup'
    print(f"  vpu_nomixup_mean_prior (auto): {len(df_nomixup_auto)} experiments")

    # 4. vpu_nomixup_mean_prior with method_prior=0.5
    df_nomixup_05 = load_method_results(results_dir, 'vpu_nomixup_mean_prior', method_prior_filter=0.5)
    df_nomixup_05['method_label'] = 'no-mixup: prior=0.5'
    df_nomixup_05['group'] = 'no-mixup'
    print(f"  vpu_nomixup_mean_prior (0.5): {len(df_nomixup_05)} experiments")

    # MIXUP VARIANTS
    # 5. vpu (classic with mixup)
    df_vpu = load_method_results(results_dir, 'vpu')
    df_vpu['method_label'] = 'mixup: classic VPU'
    df_vpu['group'] = 'mixup'
    print(f"  vpu (classic with mixup): {len(df_vpu)} experiments")

    # 6. vpu_mean with mixup (implicit prior=1.0)
    df_vpu_mean = load_method_results(results_dir, 'vpu_mean')
    df_vpu_mean['method_label'] = 'mixup: prior=1.0'
    df_vpu_mean['group'] = 'mixup'
    print(f"  vpu_mean (mixup, prior=1.0): {len(df_vpu_mean)} experiments")

    # 7. vpu_mean_prior with auto
    df_vpu_auto = load_method_results(results_dir, 'vpu_mean_prior', method_prior_filter="auto")
    df_vpu_auto['method_label'] = 'mixup: prior=auto'
    df_vpu_auto['group'] = 'mixup'
    print(f"  vpu_mean_prior (auto, mixup): {len(df_vpu_auto)} experiments")

    # 8. vpu_mean_prior with 0.5
    df_vpu_05 = load_method_results(results_dir, 'vpu_mean_prior', method_prior_filter=0.5)
    df_vpu_05['method_label'] = 'mixup: prior=0.5'
    df_vpu_05['group'] = 'mixup'
    print(f"  vpu_mean_prior (0.5, mixup): {len(df_vpu_05)} experiments")

    # Combine all
    df_all = pd.concat([
        df_nomixup_baseline, df_nomixup_mean, df_nomixup_auto, df_nomixup_05,
        df_vpu, df_vpu_mean, df_vpu_auto, df_vpu_05
    ], ignore_index=True)

    print(f"\nTotal: {len(df_all)} experiments across 8 methods")

    return df_all


def create_performance_comparison_table(df, output_dir='results_cartesian'):
    """Create summary table comparing all 8 methods"""

    output_dir = Path(output_dir)

    print("\n" + "="*80)
    print("Performance Comparison (All 8 Methods)")
    print("="*80)

    # Order methods logically
    method_order = [
        'no-mixup: baseline',
        'no-mixup: prior=1.0',
        'no-mixup: prior=auto',
        'no-mixup: prior=0.5',
        'mixup: classic VPU',
        'mixup: prior=1.0',
        'mixup: prior=auto',
        'mixup: prior=0.5',
    ]

    summary_data = []
    for method in method_order:
        subset = df[df['method_label'] == method]
        if len(subset) > 0:
            summary_data.append({
                'Method': method,
                'Count': len(subset),
                'AP_mean': subset['test_ap'].mean(),
                'AP_std': subset['test_ap'].std(),
                'F1_mean': subset['test_f1'].mean(),
                'ECE_mean': subset['test_ece'].mean(),
                'Oracle_CE_mean': subset['test_oracle_ce'].mean(),
                'Convergence_mean': subset['convergence_epoch'].mean(),
            })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print()

    # Find winners
    best_ap_idx = summary_df['AP_mean'].idxmax()
    best_ece_idx = summary_df['ECE_mean'].idxmin()
    best_ce_idx = summary_df['Oracle_CE_mean'].idxmin()

    print("WINNERS:")
    print(f"  Best AP: {summary_df.loc[best_ap_idx, 'Method']} (AP={summary_df.loc[best_ap_idx, 'AP_mean']:.4f})")
    print(f"  Best ECE: {summary_df.loc[best_ece_idx, 'Method']} (ECE={summary_df.loc[best_ece_idx, 'ECE_mean']:.4f})")
    print(f"  Best Oracle CE: {summary_df.loc[best_ce_idx, 'Method']} (CE={summary_df.loc[best_ce_idx, 'Oracle_CE_mean']:.4f})")
    print()

    # Compare mixup vs no-mixup
    print("MIXUP vs NO-MIXUP:")
    for prior_setting in ['baseline', 'prior=1.0', 'prior=auto', 'prior=0.5']:
        if 'baseline' in prior_setting:
            nomixup_label = 'no-mixup: baseline'
            mixup_label = 'mixup: classic VPU'
        else:
            nomixup_label = f'no-mixup: {prior_setting}'
            mixup_label = f'mixup: {prior_setting}'

        nomixup_ap = df[df['method_label'] == nomixup_label]['test_ap'].mean()
        mixup_ap = df[df['method_label'] == mixup_label]['test_ap'].mean()
        improvement = mixup_ap - nomixup_ap

        print(f"  {prior_setting}:")
        print(f"    No-mixup: AP={nomixup_ap:.4f}")
        print(f"    Mixup:    AP={mixup_ap:.4f} ({improvement:+.4f})")

    # Save table
    summary_df.to_csv(output_dir / "eight_method_comparison_summary.csv", index=False)
    print(f"\n✓ Saved summary table to {output_dir / 'eight_method_comparison_summary.csv'}")


def plot_eight_method_boxplot(df, output_dir='results_cartesian'):
    """Create boxplot comparing all 8 methods"""

    output_dir = Path(output_dir)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics = [
        ('test_ap', 'Test AP', axes[0]),
        ('test_f1', 'Test F1', axes[1]),
        ('test_ece', 'Test ECE (lower is better)', axes[2]),
    ]

    method_order = [
        'no-mixup: baseline',
        'no-mixup: prior=1.0',
        'no-mixup: prior=auto',
        'no-mixup: prior=0.5',
        'mixup: classic VPU',
        'mixup: prior=1.0',
        'mixup: prior=auto',
        'mixup: prior=0.5',
    ]

    # Colors: blue for no-mixup, orange for mixup
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12',
              '#5dade2', '#ec7063', '#58d68d', '#f8c471']

    for metric, ylabel, ax in metrics:
        data_to_plot = [df[df['method_label'] == m][metric].dropna() for m in method_order]

        bp = ax.boxplot(data_to_plot, labels=[m.split(': ')[1] if ': ' in m else m for m in method_order],
                        patch_artist=True, widths=0.6)

        # Color boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticklabels([m.split(': ')[1] if ': ' in m else m for m in method_order],
                           rotation=25, ha='right', fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)

        # Add mean values
        for i, (data, method) in enumerate(zip(data_to_plot, method_order), 1):
            if len(data) > 0:
                mean_val = data.mean()
                ax.text(i, mean_val, f'{mean_val:.3f}', ha='center', va='bottom',
                       fontweight='bold', fontsize=8)

        # Add vertical separator between no-mixup and mixup
        ax.axvline(x=4.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.text(2.5, ax.get_ylim()[1] * 0.98, 'No-Mixup', ha='center', fontsize=10, fontweight='bold')
        ax.text(6.5, ax.get_ylim()[1] * 0.98, 'Mixup', ha='center', fontsize=10, fontweight='bold')

    plt.suptitle('8-Way Method Comparison: No-Mixup vs Mixup', fontsize=14, y=1.00)
    plt.tight_layout()
    output_path = output_dir / "eight_method_comparison_boxplot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved boxplot to {output_path}")
    plt.close()


def compare_classic_vpu_vs_prior05_mixup(df, output_dir='results_cartesian'):
    """Specific comparison: Classic VPU (mixup) vs prior=0.5 (mixup)"""

    output_dir = Path(output_dir)

    print("\n" + "="*80)
    print("KEY COMPARISON: Classic VPU (mixup) vs prior=0.5 (mixup)")
    print("="*80)

    classic = df[df['method_label'] == 'mixup: classic VPU']
    prior05 = df[df['method_label'] == 'mixup: prior=0.5']

    print(f"\nClassic VPU (mixup):")
    print(f"  Count: {len(classic)}")
    print(f"  AP: {classic['test_ap'].mean():.4f} ± {classic['test_ap'].std():.4f}")
    print(f"  F1: {classic['test_f1'].mean():.4f} ± {classic['test_f1'].std():.4f}")
    print(f"  ECE: {classic['test_ece'].mean():.4f} ± {classic['test_ece'].std():.4f}")
    print(f"  Oracle CE: {classic['test_oracle_ce'].mean():.4f} ± {classic['test_oracle_ce'].std():.4f}")

    print(f"\nprior=0.5 (mixup):")
    print(f"  Count: {len(prior05)}")
    print(f"  AP: {prior05['test_ap'].mean():.4f} ± {prior05['test_ap'].std():.4f}")
    print(f"  F1: {prior05['test_f1'].mean():.4f} ± {prior05['test_f1'].std():.4f}")
    print(f"  ECE: {prior05['test_ece'].mean():.4f} ± {prior05['test_ece'].std():.4f}")
    print(f"  Oracle CE: {prior05['test_oracle_ce'].mean():.4f} ± {prior05['test_oracle_ce'].std():.4f}")

    # Statistical test
    t_stat, p_value = stats.ttest_ind(classic['test_ap'].dropna(), prior05['test_ap'].dropna())
    improvement = prior05['test_ap'].mean() - classic['test_ap'].mean()

    print(f"\nStatistical Test (AP):")
    print(f"  Difference: {improvement:+.4f}")
    print(f"  t={t_stat:.3f}, p={p_value:.4f}", end="")
    if p_value < 0.001:
        print(" *** (highly significant)")
    elif p_value < 0.01:
        print(" ** (significant)")
    elif p_value < 0.05:
        print(" * (significant)")
    else:
        print(" (not significant)")

    # Conclusion
    print("\nCONCLUSION:")
    if improvement > 0 and p_value < 0.05:
        print(f"  ✓ prior=0.5 with mixup BEATS classic VPU by {improvement:.4f} AP (p={p_value:.4f})")
    elif improvement < 0 and p_value < 0.05:
        print(f"  ✗ Classic VPU BEATS prior=0.5 with mixup by {-improvement:.4f} AP (p={p_value:.4f})")
    else:
        print(f"  ≈ No significant difference (Δ={improvement:+.4f}, p={p_value:.4f})")


def plot_mixup_effect_by_prior_setting(df, output_dir='results_cartesian'):
    """Show mixup effect for each prior setting"""

    output_dir = Path(output_dir)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    prior_settings = [
        ('baseline', 'mixup: classic VPU'),
        ('prior=1.0', 'mixup: prior=1.0'),
        ('prior=auto', 'mixup: prior=auto'),
        ('prior=0.5', 'mixup: prior=0.5'),
    ]

    for idx, (prior, mixup_label) in enumerate(prior_settings):
        ax = axes[idx]

        if 'baseline' in prior:
            nomixup_label = 'no-mixup: baseline'
        else:
            nomixup_label = f'no-mixup: {prior}'

        nomixup_data = df[df['method_label'] == nomixup_label]['test_ap']
        mixup_data = df[df['method_label'] == mixup_label]['test_ap']

        # Side-by-side boxplots
        data = [nomixup_data, mixup_data]
        bp = ax.boxplot(data, labels=['No Mixup', 'Mixup'], patch_artist=True, widths=0.5)

        bp['boxes'][0].set_facecolor('#3498db')
        bp['boxes'][1].set_facecolor('#ec7063')

        for patch in bp['boxes']:
            patch.set_alpha(0.6)

        ax.set_ylabel('Test AP', fontsize=11)
        ax.set_title(f'{prior.replace("baseline", "Baseline (no prior)")}', fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)

        # Add improvement annotation
        improvement = mixup_data.mean() - nomixup_data.mean()
        y_pos = max(nomixup_data.max(), mixup_data.max()) * 1.02
        ax.text(1.5, y_pos, f'Δ = {improvement:+.4f}', ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='yellow' if improvement > 0 else 'lightgray', alpha=0.5))

        # Add mean values
        ax.text(1, nomixup_data.mean(), f'{nomixup_data.mean():.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax.text(2, mixup_data.mean(), f'{mixup_data.mean():.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.suptitle('Mixup Effect Across Different Prior Settings', fontsize=14, y=0.995)
    plt.tight_layout()
    output_path = output_dir / "mixup_effect_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved mixup effect plot to {output_path}")
    plt.close()


def main():
    print("="*80)
    print("8-Way VPU Method Comparison: No-Mixup vs Mixup")
    print("="*80)
    print()

    # Load all methods
    df = load_all_methods()

    if len(df) == 0:
        print("No results found!")
        return

    print(f"\nDatasets: {sorted(df['dataset'].unique())}")
    print(f"Seeds: {sorted(df['seed'].unique())}")
    print(f"Label frequencies: {sorted(df['c'].unique())}")
    print()

    # Generate analyses
    create_performance_comparison_table(df)
    plot_eight_method_boxplot(df)
    compare_classic_vpu_vs_prior05_mixup(df)
    plot_mixup_effect_by_prior_setting(df)

    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print("\nGenerated files:")
    print("  - eight_method_comparison_summary.csv")
    print("  - eight_method_comparison_boxplot.png")
    print("  - mixup_effect_comparison.png")


if __name__ == "__main__":
    main()
