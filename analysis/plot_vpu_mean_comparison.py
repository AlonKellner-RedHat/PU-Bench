#!/usr/bin/env python3
"""
Compare VPU-mean variants to VPU variants across datasets, priors, and hyperparameters.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

def load_data():
    """Load the results CSV"""
    results_path = Path(__file__).parent.parent / "results" / "vpu_variants_analysis.csv"
    df = pd.read_csv(results_path)
    return df

def filter_vpu_variants(df):
    """Filter for VPU and VPU-mean variants only"""
    vpu_methods = ['vpu', 'vpu_mean', 'vpu_nomixup', 'vpu_nomixup_mean']
    return df[df['method'].isin(vpu_methods)].copy()

def create_comparison_plots(df, output_dir):
    """Create comprehensive comparison plots"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Filter for VPU variants
    df_vpu = filter_vpu_variants(df)

    # Remove Connect4 (all zeros)
    df_vpu = df_vpu[df_vpu['dataset'] != 'Connect4']

    # 1. Overall F1 comparison: VPU vs VPU-mean
    plot_overall_comparison(df_vpu, output_dir)

    # 2. Per-dataset comparison
    plot_per_dataset_comparison(df_vpu, output_dir)

    # 3. Prior sensitivity comparison
    plot_prior_sensitivity(df_vpu, output_dir)

    # 4. C value sensitivity comparison
    plot_c_sensitivity(df_vpu, output_dir)

    # 5. Metric comparison (F1, AUC, Accuracy)
    plot_metrics_comparison(df_vpu, output_dir)

    # 6. Performance degradation analysis
    plot_degradation_analysis(df_vpu, output_dir)

    print(f"\nAll plots saved to {output_dir}")

def plot_overall_comparison(df, output_dir):
    """Compare overall performance: VPU vs VPU-mean"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Group by method and compute mean
    metrics = ['test_f1', 'test_auc', 'test_accuracy', 'test_precision']
    metric_names = ['F1 Score', 'AUC', 'Accuracy', 'Precision']

    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]

        # Aggregate by method
        method_stats = df.groupby('method')[metric].agg(['mean', 'std']).reset_index()

        # Reorder for better comparison
        order = ['vpu', 'vpu_mean', 'vpu_nomixup', 'vpu_nomixup_mean']
        method_stats = method_stats.set_index('method').loc[order].reset_index()

        # Bar plot with error bars
        x = np.arange(len(method_stats))
        bars = ax.bar(x, method_stats['mean'], yerr=method_stats['std'],
                      capsize=5, alpha=0.7,
                      color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

        # Add value labels on bars
        for i, (bar, mean_val) in enumerate(zip(bars, method_stats['mean'])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean_val:.3f}',
                   ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(method_stats['method'], rotation=45, ha='right')
        ax.set_ylabel(name)
        ax.set_title(f'Overall {name} Comparison')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'overall_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created: overall_comparison.png")

def plot_per_dataset_comparison(df, output_dir):
    """Compare performance per dataset"""
    datasets = sorted(df['dataset'].unique())

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()

    for idx, dataset in enumerate(datasets):
        if idx >= len(axes):
            break

        ax = axes[idx]
        df_dataset = df[df['dataset'] == dataset]

        # Group by method and compute mean F1
        method_f1 = df_dataset.groupby('method')['test_f1'].mean().reset_index()
        order = ['vpu', 'vpu_mean', 'vpu_nomixup', 'vpu_nomixup_mean']
        method_f1 = method_f1.set_index('method').reindex(order).reset_index()

        # Bar plot
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars = ax.bar(range(len(method_f1)), method_f1['test_f1'],
                      color=colors, alpha=0.7)

        # Add value labels
        for bar, val in zip(bars, method_f1['test_f1']):
            if not np.isnan(val):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}',
                       ha='center', va='bottom', fontsize=8)

        ax.set_xticks(range(len(method_f1)))
        ax.set_xticklabels(method_f1['method'], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('F1 Score')
        ax.set_title(f'{dataset}', fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.05)

    # Remove extra subplots
    for idx in range(len(datasets), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(output_dir / 'per_dataset_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created: per_dataset_comparison.png")

def plot_prior_sensitivity(df, output_dir):
    """Compare performance across different prior values"""
    # Filter rows where prior is not null
    df_with_prior = df[df['prior'].notna()].copy()

    if len(df_with_prior) == 0:
        print("No prior data available, skipping prior sensitivity plot")
        return

    # Get unique datasets that have prior experiments
    datasets_with_prior = df_with_prior['dataset'].unique()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, dataset in enumerate(sorted(datasets_with_prior)):
        if idx >= len(axes):
            break

        ax = axes[idx]
        df_dataset = df_with_prior[df_with_prior['dataset'] == dataset]

        # Plot each method
        methods = ['vpu', 'vpu_mean', 'vpu_nomixup', 'vpu_nomixup_mean']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        markers = ['o', 's', '^', 'D']

        for method, color, marker in zip(methods, colors, markers):
            df_method = df_dataset[df_dataset['method'] == method]
            if len(df_method) > 0:
                # Group by prior and compute mean
                prior_f1 = df_method.groupby('prior')['test_f1'].mean().reset_index()
                prior_f1 = prior_f1.sort_values('prior')

                ax.plot(prior_f1['prior'], prior_f1['test_f1'],
                       marker=marker, label=method, color=color,
                       linewidth=2, markersize=6, alpha=0.7)

        ax.set_xlabel('Prior (π)')
        ax.set_ylabel('F1 Score')
        ax.set_title(f'{dataset}', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)

    # Remove extra subplots
    for idx in range(len(datasets_with_prior), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(output_dir / 'prior_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created: prior_sensitivity.png")

def plot_c_sensitivity(df, output_dir):
    """Compare performance across different c values"""
    datasets = sorted(df['dataset'].unique())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, dataset in enumerate(datasets):
        if idx >= len(axes):
            break

        ax = axes[idx]
        df_dataset = df[df['dataset'] == dataset]

        # Plot each method
        methods = ['vpu', 'vpu_mean', 'vpu_nomixup', 'vpu_nomixup_mean']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        markers = ['o', 's', '^', 'D']

        for method, color, marker in zip(methods, colors, markers):
            df_method = df_dataset[df_dataset['method'] == method]
            if len(df_method) > 0:
                # Group by c and compute mean
                c_f1 = df_method.groupby('c')['test_f1'].mean().reset_index()
                c_f1 = c_f1.sort_values('c')

                if len(c_f1) > 1:  # Only plot if there are multiple c values
                    ax.plot(c_f1['c'], c_f1['test_f1'],
                           marker=marker, label=method, color=color,
                           linewidth=2, markersize=6, alpha=0.7)

        ax.set_xlabel('Label Frequency (c)')
        ax.set_ylabel('F1 Score')
        ax.set_title(f'{dataset}', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)
        ax.set_xscale('log')

    # Remove extra subplots
    for idx in range(len(datasets), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(output_dir / 'c_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created: c_sensitivity.png")

def plot_metrics_comparison(df, output_dir):
    """Compare different metrics across methods"""
    metrics = ['test_f1', 'test_auc', 'test_accuracy', 'test_precision', 'test_recall']
    metric_names = ['F1 Score', 'AUC', 'Accuracy', 'Precision', 'Recall']

    # Compute mean performance per method per metric
    data = []
    methods = ['vpu', 'vpu_mean', 'vpu_nomixup', 'vpu_nomixup_mean']

    for method in methods:
        df_method = df[df['method'] == method]
        for metric, name in zip(metrics, metric_names):
            mean_val = df_method[metric].mean()
            data.append({
                'Method': method,
                'Metric': name,
                'Value': mean_val
            })

    df_metrics = pd.DataFrame(data)

    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=(14, 6))

    # Pivot for grouped bars
    pivot_df = df_metrics.pivot(index='Metric', columns='Method', values='Value')
    pivot_df = pivot_df[methods]  # Reorder columns

    pivot_df.plot(kind='bar', ax=ax, width=0.8,
                  color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                  alpha=0.7)

    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics Comparison Across VPU Variants', fontsize=12, fontweight='bold')
    ax.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created: metrics_comparison.png")

def plot_degradation_analysis(df, output_dir):
    """Analyze performance degradation: VPU vs VPU-mean"""

    # Create pairs: vpu vs vpu_mean, vpu_nomixup vs vpu_nomixup_mean
    pairs = [
        ('vpu', 'vpu_mean'),
        ('vpu_nomixup', 'vpu_nomixup_mean')
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, (method1, method2) in enumerate(pairs):
        ax = axes[idx]

        # Get all unique configurations (dataset, c, prior)
        df1 = df[df['method'] == method1].copy()
        df2 = df[df['method'] == method2].copy()

        # Merge to compare
        df1['key'] = df1['dataset'] + '_' + df1['c'].astype(str) + '_' + df1['prior'].astype(str)
        df2['key'] = df2['dataset'] + '_' + df2['c'].astype(str) + '_' + df2['prior'].astype(str)

        merged = pd.merge(df1[['key', 'test_f1', 'dataset']],
                         df2[['key', 'test_f1']],
                         on='key',
                         suffixes=('_base', '_mean'))

        # Compute degradation
        merged['degradation'] = merged['test_f1_base'] - merged['test_f1_mean']
        merged['degradation_pct'] = (merged['degradation'] / merged['test_f1_base']) * 100

        # Plot per dataset
        datasets = sorted(merged['dataset'].unique())
        colors = plt.cm.Set3(np.linspace(0, 1, len(datasets)))

        for dataset, color in zip(datasets, colors):
            df_dataset = merged[merged['dataset'] == dataset]
            ax.scatter(df_dataset['test_f1_base'], df_dataset['degradation_pct'],
                      label=dataset, alpha=0.6, s=80, color=color)

        # Add reference line at y=0
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='No degradation')

        ax.set_xlabel(f'{method1} F1 Score')
        ax.set_ylabel('Performance Degradation (%)')
        ax.set_title(f'{method1} vs {method2}', fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(alpha=0.3)

        # Add mean degradation text
        mean_deg = merged['degradation_pct'].mean()
        ax.text(0.05, 0.95, f'Mean degradation: {mean_deg:.2f}%',
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'degradation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created: degradation_analysis.png")

def print_summary_statistics(df):
    """Print summary statistics comparing VPU and VPU-mean"""
    print("\n" + "="*80)
    print("VPU vs VPU-MEAN COMPARISON SUMMARY")
    print("="*80)

    # Overall statistics
    print("\n1. Overall Performance (Mean ± Std):")
    print("-" * 80)
    methods = ['vpu', 'vpu_mean', 'vpu_nomixup', 'vpu_nomixup_mean']
    metrics = ['test_f1', 'test_auc', 'test_accuracy']

    for metric in metrics:
        print(f"\n{metric.upper()}:")
        for method in methods:
            df_method = df[df['method'] == method]
            mean_val = df_method[metric].mean()
            std_val = df_method[metric].std()
            print(f"  {method:20s}: {mean_val:.4f} ± {std_val:.4f}")

    # Degradation analysis
    print("\n\n2. Performance Degradation Analysis:")
    print("-" * 80)

    pairs = [('vpu', 'vpu_mean'), ('vpu_nomixup', 'vpu_nomixup_mean')]

    for method1, method2 in pairs:
        df1 = df[df['method'] == method1].copy()
        df2 = df[df['method'] == method2].copy()

        df1['key'] = df1['dataset'] + '_' + df1['c'].astype(str) + '_' + df1['prior'].astype(str)
        df2['key'] = df2['dataset'] + '_' + df2['c'].astype(str) + '_' + df2['prior'].astype(str)

        merged = pd.merge(df1[['key', 'test_f1']],
                         df2[['key', 'test_f1']],
                         on='key',
                         suffixes=('_base', '_mean'))

        merged['degradation'] = merged['test_f1_base'] - merged['test_f1_mean']
        merged['degradation_pct'] = (merged['degradation'] / merged['test_f1_base']) * 100

        print(f"\n{method1} → {method2}:")
        print(f"  Mean degradation: {merged['degradation_pct'].mean():.2f}%")
        print(f"  Std degradation:  {merged['degradation_pct'].std():.2f}%")
        print(f"  Min degradation:  {merged['degradation_pct'].min():.2f}%")
        print(f"  Max degradation:  {merged['degradation_pct'].max():.2f}%")

        # Count wins/losses
        wins = (merged['degradation'] < 0).sum()
        losses = (merged['degradation'] > 0).sum()
        ties = (merged['degradation'] == 0).sum()
        total = len(merged)

        print(f"  {method2} wins:    {wins}/{total} ({100*wins/total:.1f}%)")
        print(f"  {method1} wins:    {losses}/{total} ({100*losses/total:.1f}%)")
        print(f"  Ties:             {ties}/{total} ({100*ties/total:.1f}%)")

def main():
    """Main function"""
    print("Loading data...")
    df = load_data()

    print(f"Loaded {len(df)} rows")
    print(f"Methods: {sorted(df['method'].unique())}")
    print(f"Datasets: {sorted(df['dataset'].unique())}")

    # Filter VPU variants
    df_vpu = filter_vpu_variants(df)
    print(f"\nFiltered to {len(df_vpu)} rows with VPU variants")

    # Print summary statistics
    print_summary_statistics(df_vpu)

    # Create plots
    print("\n\nCreating comparison plots...")
    output_dir = Path(__file__).parent.parent / "results" / "vpu_mean_comparison_plots"
    create_comparison_plots(df, output_dir)

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    main()
