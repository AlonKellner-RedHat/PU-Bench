#!/usr/bin/env python3
"""
Comprehensive VPU comparison including prior-based variants and multi-seed analysis.
Includes box plots to show variance across seeds.
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

def filter_vpu_all_variants(df):
    """Filter for ALL VPU variants including prior-based"""
    vpu_methods = [
        'vpu', 'vpu_mean', 'vpu_nomixup', 'vpu_nomixup_mean',
        'vpu_mean_prior', 'vpu_nomixup_mean_prior'
    ]
    df_filtered = df[df['method'].isin(vpu_methods)].copy()
    return df_filtered

def create_comprehensive_plots(df, output_dir):
    """Create comprehensive comparison plots with box plots for variance"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Filter for all VPU variants
    df_vpu = filter_vpu_all_variants(df)
    df_vpu = df_vpu[~df_vpu['dataset'].isin(['Connect4', 'profile'])]

    print(f"Filtered to {len(df_vpu)} rows for all VPU variants")
    print(f"Datasets: {sorted(df_vpu['dataset'].unique())}")
    print(f"Methods: {sorted(df_vpu['method'].unique())}")

    # 1. Box plots for all metrics (multi-seed variance)
    plot_boxplots_all_metrics(df_vpu, output_dir)

    # 2. Overall bar charts with error bars
    plot_overall_with_errorbar(df_vpu, output_dir)

    # 3. Calibration metrics with box plots
    plot_calibration_boxplots(df_vpu, output_dir)

    # 4. Per-dataset box plots
    plot_per_dataset_boxplots(df_vpu, output_dir)

    # 5. Prior-based vs non-prior comparison
    plot_prior_variant_comparison(df_vpu, output_dir)

    # 6. Multi-seed heatmap (variance analysis)
    plot_seed_variance_heatmap(df_vpu, output_dir)

    # 7. Performance vs calibration scatter with all variants
    plot_performance_calibration_all(df_vpu, output_dir)

    # 8. Prior sensitivity with all variants
    plot_prior_sensitivity_all_variants(df_vpu, output_dir)

    # 9. C sensitivity with all variants
    plot_c_sensitivity_all_variants(df_vpu, output_dir)

    print(f"\nAll plots saved to {output_dir}")

def plot_boxplots_all_metrics(df, output_dir):
    """Box plots for all metrics showing multi-seed variance"""
    metrics = [
        ('test_f1', 'F1 Score', False),
        ('test_auc', 'AUC', False),
        ('test_accuracy', 'Accuracy', False),
        ('test_precision', 'Precision', False),
        ('test_recall', 'Recall', False),
        ('test_anice', 'A-NICE', True),
        ('test_snice', 'S-NICE', True),
        ('test_ece', 'ECE', True),
        ('test_mce', 'MCE', True),
        ('test_brier', 'Brier Score', True),
    ]

    available_metrics = [(m, n, inv) for m, n, inv in metrics
                         if m in df.columns and df[m].notna().sum() > 10]

    n_metrics = len(available_metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    method_order = ['vpu', 'vpu_mean', 'vpu_mean_prior',
                    'vpu_nomixup', 'vpu_nomixup_mean', 'vpu_nomixup_mean_prior']
    colors = ['#1f77b4', '#ff7f0e', '#ff9933', '#2ca02c', '#d62728', '#e377c2']

    for idx, (metric, name, invert) in enumerate(available_metrics):
        ax = axes[idx]

        # Prepare data for box plot
        data_to_plot = []
        labels = []
        for method in method_order:
            df_method = df[df['method'] == method]
            if len(df_method) > 0:
                values = df_method[metric].dropna().values
                if len(values) > 0:
                    data_to_plot.append(values)
                    labels.append(method)

        # Create box plot
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                        showmeans=True, meanline=True,
                        boxprops=dict(alpha=0.7),
                        medianprops=dict(color='red', linewidth=2),
                        meanprops=dict(color='blue', linewidth=2, linestyle='--'))

        # Color boxes
        for patch, method in zip(bp['boxes'], labels):
            color_idx = method_order.index(method)
            patch.set_facecolor(colors[color_idx])

        # Add sample size annotations
        for i, (method, data) in enumerate(zip(labels, data_to_plot)):
            n = len(data)
            ax.text(i+1, ax.get_ylim()[0], f'n={n}',
                   ha='center', va='top', fontsize=7)

        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(name)

        title_suffix = " (lower is better)" if invert else " (higher is better)"
        ax.set_title(f'{name}{title_suffix}', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add legend for median and mean
        if idx == 0:
            ax.legend([bp['medians'][0], bp['means'][0]],
                     ['Median', 'Mean'], loc='upper right', fontsize=8)

    # Remove extra subplots
    for idx in range(n_metrics, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle('VPU Variants: Box Plots Across All Seeds\n(Shows variance across multiple random seeds)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'boxplots_all_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created: boxplots_all_metrics.png")

def plot_overall_with_errorbar(df, output_dir):
    """Overall comparison with error bars (std across seeds)"""
    metrics = [
        ('test_f1', 'F1 Score', False),
        ('test_auc', 'AUC', False),
        ('test_accuracy', 'Accuracy', False),
        ('test_anice', 'A-NICE', True),
        ('test_ece', 'ECE', True),
        ('test_brier', 'Brier Score', True),
    ]

    available_metrics = [(m, n, inv) for m, n, inv in metrics
                         if m in df.columns and df[m].notna().sum() > 10]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    method_order = ['vpu', 'vpu_mean', 'vpu_mean_prior',
                    'vpu_nomixup', 'vpu_nomixup_mean', 'vpu_nomixup_mean_prior']
    colors = ['#1f77b4', '#ff7f0e', '#ff9933', '#2ca02c', '#d62728', '#e377c2']

    for idx, (metric, name, invert) in enumerate(available_metrics):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Aggregate by method
        method_stats = df.groupby('method')[metric].agg(['mean', 'std', 'count']).reset_index()
        method_stats = method_stats.set_index('method').reindex(method_order).reset_index().dropna()

        # Bar plot with error bars
        x = np.arange(len(method_stats))
        bars = ax.bar(x, method_stats['mean'], yerr=method_stats['std'],
                      capsize=5, alpha=0.7,
                      color=[colors[method_order.index(m)] for m in method_stats['method']])

        # Add value labels on bars
        for i, (bar, mean_val, std_val, count) in enumerate(zip(bars, method_stats['mean'],
                                                                  method_stats['std'], method_stats['count'])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean_val:.3f}\n±{std_val:.3f}\n(n={int(count)})',
                   ha='center', va='bottom', fontsize=8)

        # Highlight best
        if invert:
            best_idx = method_stats['mean'].idxmin()
        else:
            best_idx = method_stats['mean'].idxmax()
        bars[best_idx].set_edgecolor('green')
        bars[best_idx].set_linewidth(3)

        ax.set_xticks(x)
        ax.set_xticklabels(method_stats['method'], rotation=45, ha='right')
        ax.set_ylabel(name)

        title_suffix = " (lower is better)" if invert else " (higher is better)"
        ax.set_title(f'{name}{title_suffix}', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    # Remove extra subplots
    for idx in range(len(available_metrics), len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle('VPU Variants: Overall Performance (Mean ± Std across seeds)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_with_errorbar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created: overall_with_errorbar.png")

def plot_calibration_boxplots(df, output_dir):
    """Box plots for calibration metrics"""
    calibration_metrics = [
        ('test_anice', 'A-NICE'),
        ('test_snice', 'S-NICE'),
        ('test_ece', 'ECE'),
        ('test_mce', 'MCE'),
        ('test_brier', 'Brier Score'),
    ]

    available = [(m, n) for m, n in calibration_metrics
                 if m in df.columns and df[m].notna().sum() > 10]

    if not available:
        return

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    method_order = ['vpu', 'vpu_mean', 'vpu_mean_prior',
                    'vpu_nomixup', 'vpu_nomixup_mean', 'vpu_nomixup_mean_prior']
    colors = ['#1f77b4', '#ff7f0e', '#ff9933', '#2ca02c', '#d62728', '#e377c2']

    for idx, (metric, name) in enumerate(available):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Prepare data for box plot
        data_to_plot = []
        labels = []
        for method in method_order:
            df_method = df[df['method'] == method]
            if len(df_method) > 0:
                values = df_method[metric].dropna().values
                if len(values) > 0:
                    data_to_plot.append(values)
                    labels.append(method)

        # Create box plot
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                        showmeans=True, meanline=True,
                        boxprops=dict(alpha=0.7),
                        medianprops=dict(color='red', linewidth=2),
                        meanprops=dict(color='blue', linewidth=2, linestyle='--'))

        # Color boxes
        for patch, method in zip(bp['boxes'], labels):
            color_idx = method_order.index(method)
            patch.set_facecolor(colors[color_idx])

        # Highlight best (lowest)
        means = [np.mean(d) for d in data_to_plot]
        best_idx = np.argmin(means)
        bp['boxes'][best_idx].set_edgecolor('green')
        bp['boxes'][best_idx].set_linewidth(3)

        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(name)
        ax.set_title(f'{name} (lower is better)', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    # Remove extra subplots
    for idx in range(len(available), len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle('Calibration Metrics: Box Plots Across All Seeds',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'calibration_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created: calibration_boxplots.png")

def plot_per_dataset_boxplots(df, output_dir):
    """Box plots per dataset for F1 score"""
    datasets = sorted([d for d in df['dataset'].unique() if d not in ['Connect4', 'profile']])

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    method_order = ['vpu', 'vpu_mean', 'vpu_mean_prior',
                    'vpu_nomixup', 'vpu_nomixup_mean', 'vpu_nomixup_mean_prior']
    colors = ['#1f77b4', '#ff7f0e', '#ff9933', '#2ca02c', '#d62728', '#e377c2']

    for idx, dataset in enumerate(datasets[:8]):
        if idx >= len(axes):
            break

        ax = axes[idx]
        df_dataset = df[df['dataset'] == dataset]

        # Prepare data
        data_to_plot = []
        labels = []
        for method in method_order:
            df_method = df_dataset[df_dataset['method'] == method]
            if len(df_method) > 0:
                values = df_method['test_f1'].dropna().values
                if len(values) > 0:
                    data_to_plot.append(values)
                    labels.append(method.replace('vpu_', '').replace('_', '\n'))

        if not data_to_plot:
            continue

        # Create box plot
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                        showmeans=True, meanline=True,
                        boxprops=dict(alpha=0.7),
                        medianprops=dict(color='red', linewidth=2),
                        meanprops=dict(color='blue', linewidth=2, linestyle='--'))

        # Color boxes
        for patch, method in zip(bp['boxes'], method_order[:len(data_to_plot)]):
            color_idx = method_order.index(method)
            patch.set_facecolor(colors[color_idx])

        ax.set_ylabel('F1 Score')
        ax.set_title(dataset, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.05)

    # Remove extra subplots
    for idx in range(len(datasets[:8]), len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle('F1 Score by Dataset: Box Plots Across All Seeds',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'per_dataset_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created: per_dataset_boxplots.png")

def plot_prior_variant_comparison(df, output_dir):
    """Compare prior-based variants to non-prior variants"""
    pairs = [
        ('vpu_mean', 'vpu_mean_prior', 'VPU Mean vs VPU Mean Prior'),
        ('vpu_nomixup_mean', 'vpu_nomixup_mean_prior', 'VPU Nomixup Mean vs VPU Nomixup Mean Prior'),
    ]

    metrics = ['test_f1', 'test_auc', 'test_anice', 'test_ece']
    metric_names = ['F1 Score', 'AUC', 'A-NICE (↓)', 'ECE (↓)']

    for method1, method2, title in pairs:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            if metric not in df.columns or df[metric].notna().sum() < 10:
                continue

            ax = axes[idx]

            # Get data for both methods
            df1 = df[df['method'] == method1]
            df2 = df[df['method'] == method2]

            data_to_plot = [
                df1[metric].dropna().values,
                df2[metric].dropna().values
            ]
            labels = [method1.replace('vpu_', ''), method2.replace('vpu_', '')]

            # Create box plot
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                            showmeans=True, meanline=True,
                            boxprops=dict(alpha=0.7),
                            medianprops=dict(color='red', linewidth=2),
                            meanprops=dict(color='blue', linewidth=2, linestyle='--'))

            # Color boxes
            bp['boxes'][0].set_facecolor('#ff7f0e')
            bp['boxes'][1].set_facecolor('#ff9933')

            # Add stats
            for i, data in enumerate(data_to_plot):
                mean_val = np.mean(data)
                std_val = np.std(data)
                ax.text(i+1, ax.get_ylim()[1]*0.95,
                       f'μ={mean_val:.3f}\nσ={std_val:.3f}\nn={len(data)}',
                       ha='center', va='top', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.set_ylabel(metric_name)
            ax.set_title(metric_name, fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        filename = f"prior_comparison_{method1}_vs_{method2}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Created: {filename}")

def plot_seed_variance_heatmap(df, output_dir):
    """Heatmap showing coefficient of variation (std/mean) across metrics"""
    metrics = ['test_f1', 'test_auc', 'test_accuracy', 'test_anice', 'test_ece', 'test_brier']
    available_metrics = [m for m in metrics if m in df.columns and df[m].notna().sum() > 10]

    method_order = ['vpu', 'vpu_mean', 'vpu_mean_prior',
                    'vpu_nomixup', 'vpu_nomixup_mean', 'vpu_nomixup_mean_prior']

    # Create matrix: coefficient of variation (CV = std/mean)
    cv_matrix = []
    for method in method_order:
        df_method = df[df['method'] == method]
        row = []
        for metric in available_metrics:
            values = df_method[metric].dropna()
            if len(values) > 1:
                cv = values.std() / (values.mean() + 1e-10)
                row.append(cv)
            else:
                row.append(np.nan)
        cv_matrix.append(row)

    cv_matrix = np.array(cv_matrix)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot heatmap
    im = ax.imshow(cv_matrix, cmap='YlOrRd', aspect='auto', vmin=0)

    # Set ticks
    ax.set_xticks(np.arange(len(available_metrics)))
    ax.set_yticks(np.arange(len(method_order)))
    ax.set_xticklabels([m.replace('test_', '') for m in available_metrics], rotation=45, ha='right')
    ax.set_yticklabels(method_order)

    # Add text annotations
    for i in range(len(method_order)):
        for j in range(len(available_metrics)):
            value = cv_matrix[i, j]
            if not np.isnan(value):
                text = ax.text(j, i, f'{value:.3f}',
                              ha="center", va="center", color="black", fontsize=9)

    ax.set_title('Coefficient of Variation (Std/Mean) Across Seeds\n(Lower = More Stable)',
                 fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='CV (Std/Mean)')
    plt.tight_layout()
    plt.savefig(output_dir / 'seed_variance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created: seed_variance_heatmap.png")

def plot_performance_calibration_all(df, output_dir):
    """Scatter plot: F1 vs calibration for all variants"""
    if 'test_anice' not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    method_order = ['vpu', 'vpu_mean', 'vpu_mean_prior',
                    'vpu_nomixup', 'vpu_nomixup_mean', 'vpu_nomixup_mean_prior']
    colors = ['#1f77b4', '#ff7f0e', '#ff9933', '#2ca02c', '#d62728', '#e377c2']
    markers = ['o', 's', 'D', '^', 'v', 'p']

    # Plot 1: F1 vs A-NICE
    ax = axes[0]
    for method, color, marker in zip(method_order, colors, markers):
        df_method = df[(df['method'] == method) & df['test_f1'].notna() & df['test_anice'].notna()]
        if len(df_method) > 0:
            ax.scatter(df_method['test_f1'], df_method['test_anice'],
                      label=method, color=color, marker=marker, s=50, alpha=0.6)

    ax.set_xlabel('F1 Score (higher is better)')
    ax.set_ylabel('A-NICE (lower is better)')
    ax.set_title('Performance vs Calibration: F1 vs A-NICE', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Plot 2: F1 vs ECE
    ax = axes[1]
    if 'test_ece' in df.columns:
        for method, color, marker in zip(method_order, colors, markers):
            df_method = df[(df['method'] == method) & df['test_f1'].notna() & df['test_ece'].notna()]
            if len(df_method) > 0:
                ax.scatter(df_method['test_f1'], df_method['test_ece'],
                          label=method, color=color, marker=marker, s=50, alpha=0.6)

        ax.set_xlabel('F1 Score (higher is better)')
        ax.set_ylabel('ECE (lower is better)')
        ax.set_title('Performance vs Calibration: F1 vs ECE', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_calibration_all.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created: performance_calibration_all.png")

def plot_prior_sensitivity_all_variants(df, output_dir):
    """Prior sensitivity for all variants"""
    df_with_prior = df[df['prior'].notna()].copy()
    if len(df_with_prior) == 0:
        return

    datasets = sorted(df_with_prior['dataset'].unique())[:6]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    method_order = ['vpu', 'vpu_mean', 'vpu_mean_prior',
                    'vpu_nomixup', 'vpu_nomixup_mean', 'vpu_nomixup_mean_prior']
    colors = ['#1f77b4', '#ff7f0e', '#ff9933', '#2ca02c', '#d62728', '#e377c2']
    markers = ['o', 's', 'D', '^', 'v', 'p']

    for idx, dataset in enumerate(datasets):
        if idx >= len(axes):
            break

        ax = axes[idx]
        df_dataset = df_with_prior[df_with_prior['dataset'] == dataset]

        for method, color, marker in zip(method_order, colors, markers):
            df_method = df_dataset[df_dataset['method'] == method]
            if len(df_method) > 0:
                prior_stats = df_method.groupby('prior')['test_f1'].mean().reset_index()
                prior_stats = prior_stats.sort_values('prior')

                ax.plot(prior_stats['prior'], prior_stats['test_f1'],
                       marker=marker, label=method, color=color,
                       linewidth=2, markersize=6, alpha=0.7)

        ax.set_xlabel('Prior (π)')
        ax.set_ylabel('F1 Score')
        ax.set_title(dataset, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)

    for idx in range(len(datasets), len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle('Prior Sensitivity: All VPU Variants', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'prior_sensitivity_all_variants.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created: prior_sensitivity_all_variants.png")

def plot_c_sensitivity_all_variants(df, output_dir):
    """C sensitivity for all variants"""
    datasets = sorted([d for d in df['dataset'].unique() if d not in ['Connect4', 'profile']])[:6]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    method_order = ['vpu', 'vpu_mean', 'vpu_mean_prior',
                    'vpu_nomixup', 'vpu_nomixup_mean', 'vpu_nomixup_mean_prior']
    colors = ['#1f77b4', '#ff7f0e', '#ff9933', '#2ca02c', '#d62728', '#e377c2']
    markers = ['o', 's', 'D', '^', 'v', 'p']

    for idx, dataset in enumerate(datasets):
        if idx >= len(axes):
            break

        ax = axes[idx]
        df_dataset = df[df['dataset'] == dataset]

        for method, color, marker in zip(method_order, colors, markers):
            df_method = df_dataset[df_dataset['method'] == method]
            if len(df_method) > 0:
                c_stats = df_method.groupby('c')['test_f1'].mean().reset_index()
                c_stats = c_stats.sort_values('c')

                if len(c_stats) > 1:
                    ax.plot(c_stats['c'], c_stats['test_f1'],
                           marker=marker, label=method, color=color,
                           linewidth=2, markersize=6, alpha=0.7)

        ax.set_xlabel('Label Frequency (c)')
        ax.set_ylabel('F1 Score')
        ax.set_title(dataset, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_xscale('log')
        ax.set_ylim(0, 1.05)

    for idx in range(len(datasets), len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle('Label Frequency Sensitivity: All VPU Variants', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'c_sensitivity_all_variants.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created: c_sensitivity_all_variants.png")

def print_summary_statistics(df):
    """Print summary statistics for all variants"""
    print("\n" + "="*80)
    print("COMPREHENSIVE SUMMARY: ALL VPU VARIANTS")
    print("="*80)

    methods = ['vpu', 'vpu_mean', 'vpu_mean_prior',
               'vpu_nomixup', 'vpu_nomixup_mean', 'vpu_nomixup_mean_prior']

    metrics = [
        ('test_f1', 'F1 Score', False),
        ('test_auc', 'AUC', False),
        ('test_accuracy', 'Accuracy', False),
        ('test_anice', 'A-NICE', True),
        ('test_ece', 'ECE', True),
        ('test_brier', 'Brier Score', True),
    ]

    for metric, name, invert in metrics:
        if metric not in df.columns or df[metric].notna().sum() < 10:
            continue

        print(f"\n{name}:")
        best_val = None
        best_method = None

        for method in methods:
            df_method = df[df['method'] == method]
            values = df_method[metric].dropna()
            if len(values) == 0:
                continue

            mean_val = values.mean()
            std_val = values.std()
            count = len(values)

            if best_val is None or (invert and mean_val < best_val) or (not invert and mean_val > best_val):
                best_val = mean_val
                best_method = method

            marker = " ★" if method == best_method else ""
            print(f"  {method:30s}: {mean_val:.4f} ± {std_val:.4f} (n={count}){marker}")

def main():
    """Main function"""
    print("Loading comprehensive metrics...")
    df = load_data()

    print(f"Loaded {len(df)} rows")

    # Filter all VPU variants
    df_vpu = filter_vpu_all_variants(df)
    print(f"\nFiltered to {len(df_vpu)} rows for all VPU variants")
    print(f"Methods: {sorted(df_vpu['method'].unique())}")

    # Print summary statistics
    print_summary_statistics(df_vpu)

    # Create plots
    print("\n\nCreating comprehensive plots with all variants...")
    output_dir = Path(__file__).parent.parent / "results" / "vpu_comprehensive_with_priors"
    create_comprehensive_plots(df, output_dir)

    print("\n" + "="*80)
    print("Comprehensive analysis with priors complete!")
    print("="*80)

if __name__ == "__main__":
    main()
