#!/usr/bin/env python3
"""Compare 4 VPU Variants Across Full Prior Range

Methods compared:
1. vpu_nomixup - Baseline (no mean, no prior)
2. vpu_nomixup_mean - With mean (implicit method_prior=1.0)
3. vpu_nomixup_mean_prior (auto) - With mean and true prior
4. vpu_nomixup_mean_prior (0.5) - With mean and fixed prior=0.5
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
                    'convergence_epoch': method_data.get('best', {}).get('epoch'),
                })

        except Exception as e:
            continue

    return pd.DataFrame(results)


def load_all_methods(results_dir="results_cartesian"):
    """Load all 4 method variants"""

    print("Loading method results...")

    # 1. vpu_nomixup (baseline)
    df_baseline = load_method_results(results_dir, 'vpu_nomixup')
    df_baseline['method_label'] = 'baseline (no mean)'
    print(f"  vpu_nomixup (baseline): {len(df_baseline)} experiments")

    # 2. vpu_nomixup_mean_prior with method_prior=1.0
    df_mean = load_method_results(results_dir, 'vpu_nomixup_mean_prior', method_prior_filter=1.0)
    df_mean['method_label'] = 'mean (prior=1.0)'
    print(f"  vpu_nomixup_mean (prior=1.0): {len(df_mean)} experiments")

    # 3. vpu_nomixup_mean_prior with auto (true prior)
    df_auto = load_method_results(results_dir, 'vpu_nomixup_mean_prior', method_prior_filter="auto")
    df_auto['method_label'] = 'mean_prior (auto)'
    print(f"  vpu_nomixup_mean_prior (auto): {len(df_auto)} experiments")

    # 4. vpu_nomixup_mean_prior with method_prior=0.5
    df_05 = load_method_results(results_dir, 'vpu_nomixup_mean_prior', method_prior_filter=0.5)
    df_05['method_label'] = 'mean_prior (0.5)'
    print(f"  vpu_nomixup_mean_prior (0.5): {len(df_05)} experiments")

    # Combine all
    df_all = pd.concat([df_baseline, df_mean, df_auto, df_05], ignore_index=True)
    print(f"\nTotal: {len(df_all)} experiments across 4 methods")

    return df_all


def create_performance_comparison_table(df, output_dir='results_cartesian'):
    """Create summary table comparing all 4 methods"""

    output_dir = Path(output_dir)

    print("\n" + "="*80)
    print("Performance Comparison (Across All True Priors)")
    print("="*80)

    summary = df.groupby('method_label').agg({
        'test_ap': ['mean', 'std', 'min', 'max'],
        'test_f1': ['mean', 'std'],
        'test_ece': ['mean', 'std'],
        'convergence_epoch': ['mean', 'std'],
    }).round(4)

    print(summary.to_string())
    print()

    # Statistical tests
    methods = df['method_label'].unique()
    baseline_ap = df[df['method_label'] == 'baseline (no mean)']['test_ap'].dropna()

    print("Statistical Comparison (vs baseline):")
    for method in methods:
        if method == 'baseline (no mean)':
            continue

        method_ap = df[df['method_label'] == method]['test_ap'].dropna()

        if len(method_ap) == 0:
            print(f"  {method}: No data")
            continue

        t_stat, p_value = stats.ttest_ind(baseline_ap, method_ap)
        mean_diff = method_ap.mean() - baseline_ap.mean()

        print(f"  {method}:")
        print(f"    Mean AP: {method_ap.mean():.4f} ({mean_diff:+.4f} vs baseline)")
        print(f"    t={t_stat:.3f}, p={p_value:.4f}", end="")
        if p_value < 0.001:
            print(" ***")
        elif p_value < 0.01:
            print(" **")
        elif p_value < 0.05:
            print(" *")
        else:
            print(" (ns)")

    # Save table
    summary.to_csv(output_dir / "method_comparison_summary.csv")
    print(f"\n✓ Saved summary table to {output_dir / 'method_comparison_summary.csv'}")


def plot_performance_by_true_prior(df, output_dir='results_cartesian'):
    """Plot performance curves for each method across true prior range"""

    output_dir = Path(output_dir)

    # Bin true priors into categories
    bins = [0, 0.15, 0.35, 0.55, 0.75, 1.0]
    labels = ['0.1', '0.3', '0.5', '0.7', '0.9']
    df['true_prior_bin'] = pd.cut(df['true_prior'], bins=bins, labels=labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # AP comparison
    ax = axes[0]
    for method_label in ['baseline (no mean)', 'mean (prior=1.0)', 'mean_prior (auto)', 'mean_prior (0.5)']:
        subset = df[df['method_label'] == method_label]
        grouped = subset.groupby('true_prior_bin')['test_ap'].agg(['mean', 'std']).reset_index()

        x = range(len(grouped))
        ax.plot(x, grouped['mean'], 'o-', label=method_label, linewidth=2, markersize=8)
        ax.fill_between(x, grouped['mean'] - grouped['std'],
                        grouped['mean'] + grouped['std'], alpha=0.2)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel('True Prior (π)', fontsize=12)
    ax.set_ylabel('Test AP', fontsize=12)
    ax.set_title('Performance Across True Prior Range', fontsize=14)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # ECE comparison
    ax = axes[1]
    for method_label in ['baseline (no mean)', 'mean (prior=1.0)', 'mean_prior (auto)', 'mean_prior (0.5)']:
        subset = df[df['method_label'] == method_label]
        grouped = subset.groupby('true_prior_bin')['test_ece'].agg(['mean', 'std']).reset_index()

        x = range(len(grouped))
        ax.plot(x, grouped['mean'], 'o-', label=method_label, linewidth=2, markersize=8)
        ax.fill_between(x, grouped['mean'] - grouped['std'],
                        grouped['mean'] + grouped['std'], alpha=0.2)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel('True Prior (π)', fontsize=12)
    ax.set_ylabel('Test ECE (lower is better)', fontsize=12)
    ax.set_title('Calibration Across True Prior Range', fontsize=14)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "method_comparison_by_prior.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot to {output_path}")
    plt.close()


def plot_method_comparison_boxplot(df, output_dir='results_cartesian'):
    """Create boxplot comparing all 4 methods"""

    output_dir = Path(output_dir)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metrics = [
        ('test_ap', 'Test AP', axes[0]),
        ('test_f1', 'Test F1', axes[1]),
        ('test_ece', 'Test ECE (lower is better)', axes[2]),
    ]

    method_order = ['baseline (no mean)', 'mean (prior=1.0)', 'mean_prior (auto)', 'mean_prior (0.5)']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    for metric, ylabel, ax in metrics:
        data_to_plot = [df[df['method_label'] == m][metric].dropna() for m in method_order]

        bp = ax.boxplot(data_to_plot, labels=method_order, patch_artist=True, widths=0.6)

        # Color boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticklabels(method_order, rotation=15, ha='right', fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)

        # Add mean values
        for i, (data, method) in enumerate(zip(data_to_plot, method_order), 1):
            mean_val = data.mean()
            ax.text(i, mean_val, f'{mean_val:.3f}', ha='center', va='bottom',
                   fontweight='bold', fontsize=9)

    plt.suptitle('4-Way Method Comparison (All True Priors)', fontsize=14, y=1.02)
    plt.tight_layout()
    output_path = output_dir / "method_comparison_boxplot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved boxplot to {output_path}")
    plt.close()


def create_winner_table_by_prior(df, output_dir='results_cartesian'):
    """For each true prior range, determine the best method"""

    output_dir = Path(output_dir)

    # Bin true priors
    bins = [0, 0.15, 0.35, 0.55, 0.75, 1.0]
    labels = ['~0.1', '~0.3', '~0.5', '~0.7', '~0.9']
    df['true_prior_bin'] = pd.cut(df['true_prior'], bins=bins, labels=labels)

    results = []

    for prior_bin in labels:
        subset = df[df['true_prior_bin'] == prior_bin]

        if len(subset) == 0:
            continue

        # Get mean AP for each method
        method_performance = subset.groupby('method_label')['test_ap'].mean().sort_values(ascending=False)

        winner = method_performance.index[0]
        winner_ap = method_performance.iloc[0]

        results.append({
            'true_prior_range': prior_bin,
            'best_method': winner,
            'best_ap': winner_ap,
            'baseline_ap': method_performance.get('baseline (no mean)', np.nan),
            'improvement_vs_baseline': winner_ap - method_performance.get('baseline (no mean)', winner_ap),
        })

    result_df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("Best Method by True Prior Range")
    print("="*80)
    print(result_df.to_string(index=False))
    print()

    result_df.to_csv(output_dir / "best_method_by_prior.csv", index=False)
    print(f"✓ Saved to {output_dir / 'best_method_by_prior.csv'}")


def main():
    print("="*80)
    print("4-Way VPU Method Comparison")
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
    plot_method_comparison_boxplot(df)
    plot_performance_by_true_prior(df)
    create_winner_table_by_prior(df)

    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print("\nGenerated files:")
    print("  - method_comparison_summary.csv")
    print("  - method_comparison_boxplot.png")
    print("  - method_comparison_by_prior.png")
    print("  - best_method_by_prior.csv")


if __name__ == "__main__":
    main()
