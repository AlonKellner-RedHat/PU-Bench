#!/usr/bin/env python3
"""
Comprehensive comparison of VPU-mean variants to VPU variants.
Includes ALL metrics: standard, calibration, and threshold-independent.
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
    """Load the comprehensive results CSV"""
    results_path = Path(__file__).parent.parent / "results" / "comprehensive_metrics.csv"
    df = pd.read_csv(results_path)
    return df

def filter_vpu_variants(df, exclude_batch_size=True):
    """Filter for VPU and VPU-mean variants only"""
    vpu_methods = ['vpu', 'vpu_mean', 'vpu_nomixup', 'vpu_nomixup_mean']

    if exclude_batch_size:
        # Only include default batch size (exclude batch size experiments)
        df_filtered = df[df['method'].isin(vpu_methods)].copy()
    else:
        # Include all VPU variants including batch size experiments
        vpu_pattern = df['method'].str.contains('vpu')
        df_filtered = df[vpu_pattern].copy()

    return df_filtered

def create_comparison_plots(df, output_dir):
    """Create comprehensive comparison plots"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Filter for VPU variants (default batch size only)
    df_vpu = filter_vpu_variants(df, exclude_batch_size=True)

    # Remove Connect4 and profile (all zeros)
    df_vpu = df_vpu[~df_vpu['dataset'].isin(['Connect4', 'profile'])]

    print(f"Filtered to {len(df_vpu)} rows for VPU variants (default batch size)")
    print(f"Datasets: {sorted(df_vpu['dataset'].unique())}")
    print(f"Methods: {sorted(df_vpu['method'].unique())}")

    # 1. Overall comprehensive metrics comparison
    plot_overall_comprehensive(df_vpu, output_dir)

    # 2. Per-dataset comparison with multiple metrics
    plot_per_dataset_multi_metric(df_vpu, output_dir)

    # 3. Calibration metrics comparison
    plot_calibration_metrics(df_vpu, output_dir)

    # 4. Threshold-independent metrics (if available)
    plot_threshold_independent(df_vpu, output_dir)

    # 5. Prior sensitivity comparison
    plot_prior_sensitivity(df_vpu, output_dir)

    # 6. C value sensitivity comparison
    plot_c_sensitivity(df_vpu, output_dir)

    # 7. Comprehensive metric heatmap
    plot_metrics_heatmap(df_vpu, output_dir)

    # 8. Performance degradation analysis
    plot_degradation_analysis(df_vpu, output_dir)

    # 9. Calibration vs Performance trade-off
    plot_calibration_vs_performance(df_vpu, output_dir)

    print(f"\nAll plots saved to {output_dir}")

def plot_overall_comprehensive(df, output_dir):
    """Compare overall performance across ALL metrics"""
    # Define all metrics to plot
    all_metrics = [
        ('test_f1', 'F1 Score', False),
        ('test_auc', 'AUC', False),
        ('test_accuracy', 'Accuracy', False),
        ('test_precision', 'Precision', False),
        ('test_recall', 'Recall', False),
        ('test_max_f1', 'Max F1', False),
        ('test_ap', 'Average Precision', False),
        ('test_anice', 'A-NICE', True),  # Lower is better
        ('test_snice', 'S-NICE', True),  # Lower is better
        ('test_ece', 'ECE', True),        # Lower is better
        ('test_mce', 'MCE', True),        # Lower is better
        ('test_brier', 'Brier Score', True),  # Lower is better
    ]

    # Filter to only available metrics
    available_metrics = [(m, n, inv) for m, n, inv in all_metrics if m in df.columns and df[m].notna().sum() > 10]

    n_metrics = len(available_metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    order = ['vpu', 'vpu_mean', 'vpu_nomixup', 'vpu_nomixup_mean']

    for idx, (metric, name, invert) in enumerate(available_metrics):
        ax = axes[idx]

        # Aggregate by method
        method_stats = df.groupby('method')[metric].agg(['mean', 'std']).reset_index()
        method_stats = method_stats.set_index('method').reindex(order).reset_index().dropna()

        # Bar plot with error bars
        x = np.arange(len(method_stats))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(method_stats)]
        bars = ax.bar(x, method_stats['mean'], yerr=method_stats['std'],
                      capsize=5, alpha=0.7, color=colors)

        # Add value labels on bars
        for i, (bar, mean_val) in enumerate(zip(bars, method_stats['mean'])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean_val:.3f}',
                   ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(method_stats['method'], rotation=45, ha='right')
        ax.set_ylabel(name)

        # Highlight best performer
        if invert:
            best_idx = method_stats['mean'].idxmin()
            title_suffix = " (lower is better)"
        else:
            best_idx = method_stats['mean'].idxmax()
            title_suffix = " (higher is better)"

        ax.set_title(f'{name}{title_suffix}', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    # Remove extra subplots
    for idx in range(n_metrics, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(output_dir / 'overall_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created: overall_comprehensive.png")

def plot_calibration_metrics(df, output_dir):
    """Dedicated plot for calibration metrics"""
    calibration_metrics = {
        'test_anice': 'A-NICE (lower is better)',
        'test_snice': 'S-NICE (lower is better)',
        'test_ece': 'ECE (lower is better)',
        'test_mce': 'MCE (lower is better)',
        'test_brier': 'Brier Score (lower is better)'
    }

    available = {k: v for k, v in calibration_metrics.items() if k in df.columns and df[k].notna().sum() > 10}

    if not available:
        print("No calibration metrics available, skipping plot")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    order = ['vpu', 'vpu_mean', 'vpu_nomixup', 'vpu_nomixup_mean']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, (metric, name) in enumerate(available.items()):
        if idx >= len(axes):
            break

        ax = axes[idx]
        method_stats = df.groupby('method')[metric].agg(['mean', 'std']).reset_index()
        method_stats = method_stats.set_index('method').reindex(order).reset_index().dropna()

        x = np.arange(len(method_stats))
        bars = ax.bar(x, method_stats['mean'], yerr=method_stats['std'],
                      capsize=5, alpha=0.7, color=colors[:len(method_stats)])

        # Add value labels and highlight best (lowest)
        for i, (bar, mean_val) in enumerate(zip(bars, method_stats['mean'])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean_val:.3f}',
                   ha='center', va='bottom', fontsize=9)

        best_idx = method_stats['mean'].idxmin()
        best_method = method_stats.loc[best_idx, 'method']
        bars[best_idx].set_edgecolor('green')
        bars[best_idx].set_linewidth(3)

        ax.set_xticks(x)
        ax.set_xticklabels(method_stats['method'], rotation=45, ha='right')
        ax.set_ylabel(name.split('(')[0].strip())
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    # Remove extra subplots
    for idx in range(len(available), len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle('Calibration Metrics Comparison (Lower is Better)', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'calibration_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created: calibration_metrics.png")

def plot_threshold_independent(df, output_dir):
    """Plot threshold-independent metrics if available"""
    ti_metrics = {
        'test_max_f1': 'Max F1 (threshold-independent)',
        'test_ap': 'Average Precision (AP)'
    }

    available = {k: v for k, v in ti_metrics.items() if k in df.columns and df[k].notna().sum() > 10}

    if not available:
        print("Threshold-independent metrics not widely available, skipping plot")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(7*len(available), 6))
    if len(available) == 1:
        axes = [axes]

    order = ['vpu', 'vpu_mean', 'vpu_nomixup', 'vpu_nomixup_mean']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, (metric, name) in enumerate(available.items()):
        ax = axes[idx]
        method_stats = df.groupby('method')[metric].agg(['mean', 'std']).reset_index()
        method_stats = method_stats.set_index('method').reindex(order).reset_index().dropna()

        x = np.arange(len(method_stats))
        bars = ax.bar(x, method_stats['mean'], yerr=method_stats['std'],
                      capsize=5, alpha=0.7, color=colors[:len(method_stats)])

        for i, (bar, mean_val) in enumerate(zip(bars, method_stats['mean'])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean_val:.3f}',
                   ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(method_stats['method'], rotation=45, ha='right')
        ax.set_ylabel(name.split('(')[0].strip())
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_dir / 'threshold_independent.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created: threshold_independent.png")

def plot_per_dataset_multi_metric(df, output_dir):
    """Per-dataset comparison with F1, AUC, and calibration"""
    datasets = sorted([d for d in df['dataset'].unique() if d not in ['Connect4', 'profile']])

    # Select key metrics
    metrics_to_plot = [
        ('test_f1', 'F1 Score'),
        ('test_auc', 'AUC'),
        ('test_anice', 'A-NICE (↓)'),
    ]

    available_metrics = [(m, n) for m, n in metrics_to_plot
                         if m in df.columns and df[m].notna().sum() > 10]

    fig, axes = plt.subplots(len(available_metrics), 3, figsize=(18, 5*len(available_metrics)))
    if len(available_metrics) == 1:
        axes = axes.reshape(1, -1)

    order = ['vpu', 'vpu_mean', 'vpu_nomixup', 'vpu_nomixup_mean']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for metric_idx, (metric, metric_name) in enumerate(available_metrics):
        for dataset_idx, dataset in enumerate(datasets[:3]):  # Show top 3 datasets
            if dataset_idx >= 3:
                break

            ax = axes[metric_idx, dataset_idx]
            df_dataset = df[df['dataset'] == dataset]

            method_stats = df_dataset.groupby('method')[metric].mean().reset_index()
            method_stats = method_stats.set_index('method').reindex(order).reset_index().dropna()

            bars = ax.bar(range(len(method_stats)), method_stats[metric],
                          color=colors[:len(method_stats)], alpha=0.7)

            for bar, val in zip(bars, method_stats[metric]):
                if not np.isnan(val):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.3f}',
                           ha='center', va='bottom', fontsize=8)

            ax.set_xticks(range(len(method_stats)))
            ax.set_xticklabels(method_stats['method'], rotation=45, ha='right', fontsize=8)

            if dataset_idx == 0:
                ax.set_ylabel(metric_name)
            if metric_idx == 0:
                ax.set_title(dataset, fontsize=11, fontweight='bold')

            ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'per_dataset_multi_metric.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created: per_dataset_multi_metric.png")

def plot_metrics_heatmap(df, output_dir):
    """Heatmap showing all metrics for all methods"""
    metrics = [
        'test_f1', 'test_auc', 'test_accuracy', 'test_precision', 'test_recall',
        'test_max_f1', 'test_ap',
        'test_anice', 'test_snice', 'test_ece', 'test_mce', 'test_brier'
    ]

    available_metrics = [m for m in metrics if m in df.columns and df[m].notna().sum() > 10]
    order = ['vpu', 'vpu_mean', 'vpu_nomixup', 'vpu_nomixup_mean']

    # Create matrix
    data_matrix = []
    for method in order:
        df_method = df[df['method'] == method]
        row = [df_method[m].mean() for m in available_metrics]
        data_matrix.append(row)

    data_matrix = np.array(data_matrix)

    # Normalize each column (metric) to 0-1 for better visualization
    # For inverted metrics (calibration), we'll handle separately
    inverted_metrics = ['test_anice', 'test_snice', 'test_ece', 'test_mce', 'test_brier']

    fig, ax = plt.subplots(figsize=(12, 5))

    # Create normalized version for heatmap
    data_normalized = data_matrix.copy()
    for i, metric in enumerate(available_metrics):
        col = data_matrix[:, i]
        if metric in inverted_metrics:
            # For inverted metrics, lower is better, so invert the normalization
            data_normalized[:, i] = 1 - (col - np.nanmin(col)) / (np.nanmax(col) - np.nanmin(col) + 1e-10)
        else:
            data_normalized[:, i] = (col - np.nanmin(col)) / (np.nanmax(col) - np.nanmin(col) + 1e-10)

    # Plot heatmap
    im = ax.imshow(data_normalized, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(np.arange(len(available_metrics)))
    ax.set_yticks(np.arange(len(order)))
    ax.set_xticklabels([m.replace('test_', '') for m in available_metrics], rotation=45, ha='right')
    ax.set_yticklabels(order)

    # Add text annotations with actual values
    for i in range(len(order)):
        for j in range(len(available_metrics)):
            value = data_matrix[i, j]
            if not np.isnan(value):
                text = ax.text(j, i, f'{value:.3f}',
                              ha="center", va="center", color="black", fontsize=8)

    ax.set_title('Performance Heatmap: All Metrics Across VPU Variants\n(Greener = Better)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Normalized Score (0=Worst, 1=Best)')
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created: metrics_heatmap.png")

def plot_calibration_vs_performance(df, output_dir):
    """Scatter plot: F1 vs calibration metrics"""
    if 'test_anice' not in df.columns or df['test_anice'].notna().sum() < 10:
        print("Calibration metrics not available, skipping calibration vs performance plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    methods = ['vpu', 'vpu_mean', 'vpu_nomixup', 'vpu_nomixup_mean']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    # Plot 1: F1 vs A-NICE
    ax = axes[0]
    for method, color, marker in zip(methods, colors, markers):
        df_method = df[(df['method'] == method) & df['test_f1'].notna() & df['test_anice'].notna()]
        if len(df_method) > 0:
            ax.scatter(df_method['test_f1'], df_method['test_anice'],
                      label=method, color=color, marker=marker, s=50, alpha=0.6)

    ax.set_xlabel('F1 Score (higher is better)')
    ax.set_ylabel('A-NICE (lower is better)')
    ax.set_title('Performance vs Calibration: F1 vs A-NICE', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: F1 vs ECE
    ax = axes[1]
    if 'test_ece' in df.columns:
        for method, color, marker in zip(methods, colors, markers):
            df_method = df[(df['method'] == method) & df['test_f1'].notna() & df['test_ece'].notna()]
            if len(df_method) > 0:
                ax.scatter(df_method['test_f1'], df_method['test_ece'],
                          label=method, color=color, marker=marker, s=50, alpha=0.6)

        ax.set_xlabel('F1 Score (higher is better)')
        ax.set_ylabel('ECE (lower is better)')
        ax.set_title('Performance vs Calibration: F1 vs ECE', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'calibration_vs_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created: calibration_vs_performance.png")

def plot_prior_sensitivity(df, output_dir):
    """Compare performance across different prior values"""
    df_with_prior = df[df['prior'].notna()].copy()

    if len(df_with_prior) == 0:
        print("No prior data available, skipping prior sensitivity plot")
        return

    datasets_with_prior = df_with_prior['dataset'].unique()

    # Select key metrics
    metrics = [('test_f1', 'F1'), ('test_anice', 'A-NICE (↓)')]
    available_metrics = [(m, n) for m, n in metrics if m in df.columns and df[m].notna().sum() > 10]

    for metric, metric_name in available_metrics:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, dataset in enumerate(sorted(datasets_with_prior)[:6]):
            if idx >= len(axes):
                break

            ax = axes[idx]
            df_dataset = df_with_prior[df_with_prior['dataset'] == dataset]

            methods = ['vpu', 'vpu_mean', 'vpu_nomixup', 'vpu_nomixup_mean']
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            markers = ['o', 's', '^', 'D']

            for method, color, marker in zip(methods, colors, markers):
                df_method = df_dataset[df_dataset['method'] == method]
                if len(df_method) > 0:
                    prior_stats = df_method.groupby('prior')[metric].mean().reset_index()
                    prior_stats = prior_stats.sort_values('prior')

                    ax.plot(prior_stats['prior'], prior_stats[metric],
                           marker=marker, label=method, color=color,
                           linewidth=2, markersize=6, alpha=0.7)

            ax.set_xlabel('Prior (π)')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{dataset}', fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        # Remove extra subplots
        for idx in range(len(datasets_with_prior), len(axes)):
            fig.delaxes(axes[idx])

        plt.suptitle(f'Prior Sensitivity: {metric_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'prior_sensitivity_{metric}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Created: prior_sensitivity_{metric}.png")

def plot_c_sensitivity(df, output_dir):
    """Compare performance across different c values"""
    datasets = sorted([d for d in df['dataset'].unique() if d not in ['Connect4', 'profile']])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, dataset in enumerate(datasets[:6]):
        if idx >= len(axes):
            break

        ax = axes[idx]
        df_dataset = df[df['dataset'] == dataset]

        methods = ['vpu', 'vpu_mean', 'vpu_nomixup', 'vpu_nomixup_mean']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        markers = ['o', 's', '^', 'D']

        for method, color, marker in zip(methods, colors, markers):
            df_method = df_dataset[df_dataset['method'] == method]
            if len(df_method) > 0:
                c_f1 = df_method.groupby('c')['test_f1'].mean().reset_index()
                c_f1 = c_f1.sort_values('c')

                if len(c_f1) > 1:
                    ax.plot(c_f1['c'], c_f1['test_f1'],
                           marker=marker, label=method, color=color,
                           linewidth=2, markersize=6, alpha=0.7)

        ax.set_xlabel('Label Frequency (c)')
        ax.set_ylabel('F1 Score')
        ax.set_title(f'{dataset}', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xscale('log')
        ax.set_ylim(0, 1.05)

    # Remove extra subplots
    for idx in range(len(datasets[:6]), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(output_dir / 'c_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created: c_sensitivity.png")

def plot_degradation_analysis(df, output_dir):
    """Analyze performance degradation: VPU vs VPU-mean"""
    pairs = [
        ('vpu', 'vpu_mean'),
        ('vpu_nomixup', 'vpu_nomixup_mean')
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, (method1, method2) in enumerate(pairs):
        ax = axes[idx]

        df1 = df[df['method'] == method1].copy()
        df2 = df[df['method'] == method2].copy()

        df1['key'] = df1['dataset'] + '_' + df1['c'].astype(str) + '_' + df1['prior'].astype(str) + '_' + df1['seed'].astype(str)
        df2['key'] = df2['dataset'] + '_' + df2['c'].astype(str) + '_' + df2['prior'].astype(str) + '_' + df2['seed'].astype(str)

        merged = pd.merge(df1[['key', 'test_f1', 'dataset']],
                         df2[['key', 'test_f1']],
                         on='key',
                         suffixes=('_base', '_mean'))

        merged['degradation'] = merged['test_f1_base'] - merged['test_f1_mean']
        merged['degradation_pct'] = (merged['degradation'] / (merged['test_f1_base'] + 1e-10)) * 100

        # Filter out inf/nan
        merged = merged[np.isfinite(merged['degradation_pct'])]

        datasets = sorted(merged['dataset'].unique())
        colors = plt.cm.Set3(np.linspace(0, 1, len(datasets)))

        for dataset, color in zip(datasets, colors):
            df_dataset = merged[merged['dataset'] == dataset]
            ax.scatter(df_dataset['test_f1_base'], df_dataset['degradation_pct'],
                      label=dataset, alpha=0.6, s=80, color=color)

        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='No degradation')

        ax.set_xlabel(f'{method1} F1 Score')
        ax.set_ylabel('Performance Degradation (%)')
        ax.set_title(f'{method1} vs {method2}', fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(alpha=0.3)

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
    """Print comprehensive summary statistics"""
    print("\n" + "="*80)
    print("COMPREHENSIVE VPU vs VPU-MEAN COMPARISON SUMMARY")
    print("="*80)

    methods = ['vpu', 'vpu_mean', 'vpu_nomixup', 'vpu_nomixup_mean']

    # All available metrics
    all_metrics = [
        ('test_f1', 'F1 Score', False),
        ('test_auc', 'AUC', False),
        ('test_accuracy', 'Accuracy', False),
        ('test_precision', 'Precision', False),
        ('test_recall', 'Recall', False),
        ('test_max_f1', 'Max F1', False),
        ('test_ap', 'Average Precision (AP)', False),
        ('test_anice', 'A-NICE', True),
        ('test_snice', 'S-NICE', True),
        ('test_ece', 'ECE', True),
        ('test_mce', 'MCE', True),
        ('test_brier', 'Brier Score', True),
    ]

    print("\n1. Overall Performance (Mean ± Std):")
    print("-" * 80)

    for metric, name, invert in all_metrics:
        if metric not in df.columns or df[metric].notna().sum() < 10:
            continue

        print(f"\n{name}:")
        best_val = None
        best_method = None

        for method in methods:
            df_method = df[df['method'] == method]
            mean_val = df_method[metric].mean()
            std_val = df_method[metric].std()
            count = df_method[metric].notna().sum()

            if best_val is None or (invert and mean_val < best_val) or (not invert and mean_val > best_val):
                best_val = mean_val
                best_method = method

            marker = " ★" if method == best_method else ""
            print(f"  {method:20s}: {mean_val:.4f} ± {std_val:.4f} (n={count}){marker}")

def main():
    """Main function"""
    print("Loading comprehensive metrics...")
    df = load_data()

    print(f"Loaded {len(df)} rows")
    print(f"Datasets: {sorted(df['dataset'].unique())}")
    print(f"All methods: {len(df['method'].unique())} methods")

    # Filter VPU variants
    df_vpu = filter_vpu_variants(df, exclude_batch_size=True)
    print(f"\nFiltered to {len(df_vpu)} rows with VPU variants (default batch size)")
    print(f"VPU methods: {sorted(df_vpu['method'].unique())}")

    # Print summary statistics
    print_summary_statistics(df_vpu)

    # Create plots
    print("\n\nCreating comprehensive comparison plots...")
    output_dir = Path(__file__).parent.parent / "results" / "vpu_mean_comprehensive_plots"
    create_comparison_plots(df, output_dir)

    print("\n" + "="*80)
    print("Comprehensive analysis complete!")
    print("="*80)

if __name__ == "__main__":
    main()
