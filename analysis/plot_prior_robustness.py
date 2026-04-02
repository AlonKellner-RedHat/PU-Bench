#!/usr/bin/env python3
"""Create prior robustness visualizations

Generates publication-ready plots:
1. Robustness curves: performance vs prior error
2. Degradation heatmap: dataset × prior_error
3. Label frequency interaction plot
4. Method comparison
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


def load_results(results_dir="results_robustness"):
    """Load processed results"""
    full_results = pd.read_csv(f"{results_dir}/robustness_full_results.csv")
    degradation = pd.read_csv(f"{results_dir}/robustness_degradation.csv")
    return full_results, degradation


def plot_robustness_curves(df, output_dir):
    """Plot performance vs prior error for all methods and datasets"""

    datasets = sorted(df['dataset'].unique())
    methods = ['vpu_nomixup', 'vpu_nomixup_mean', 'vpu_nomixup_mean_prior']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    method_colors = {
        'vpu_nomixup': '#1f77b4',
        'vpu_nomixup_mean': '#ff7f0e',
        'vpu_nomixup_mean_prior': '#2ca02c'
    }

    method_labels = {
        'vpu_nomixup': 'VPU-NoMixup',
        'vpu_nomixup_mean': 'VPU-NoMixup-Mean',
        'vpu_nomixup_mean_prior': 'VPU-NoMixup-Mean-Prior'
    }

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        data_ds = df[df['dataset'] == dataset]

        for method in methods:
            method_data = data_ds[data_ds['method'] == method]

            if method_data.empty:
                continue

            # For methods without prior, prior_error is always 0
            # For vpu_nomixup_mean_prior, we have different prior_error values

            # Group by prior_error and calculate mean ± std across seeds
            grouped = method_data.groupby('prior_error')['test_f1'].agg(['mean', 'std', 'count'])

            if len(grouped) == 1:
                # Methods without priors (single point at error=0)
                ax.scatter([0], [grouped['mean'].values[0]],
                          color=method_colors[method],
                          label=method_labels[method],
                          marker='o', s=100, zorder=10)
            else:
                # Method with priors (vpu_nomixup_mean_prior)
                errors = grouped.index.values
                means = grouped['mean'].values
                stds = grouped['std'].values

                ax.errorbar(errors, means, yerr=stds,
                           label=method_labels[method],
                           color=method_colors[method],
                           marker='o', markersize=6,
                           linewidth=2, capsize=4)

        ax.set_xlabel('Prior Error |π_method - π_true|')
        ax.set_ylabel('Test F1 Score')
        ax.set_title(dataset)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, max(df['prior_error']) + 0.05)

    plt.tight_layout()
    output_path = Path(output_dir) / "robustness_curves_all_datasets.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved robustness curves to {output_path}")
    plt.close()


def plot_degradation_heatmap(df_deg, output_dir):
    """Heatmap: dataset × prior_error showing relative F1 drop"""

    # Bin prior errors for cleaner visualization
    df_deg_copy = df_deg.copy()
    df_deg_copy['error_bin'] = pd.cut(df_deg_copy['prior_error'],
                                       bins=[0, 0.15, 0.25, 0.35, 0.6, 1.0],
                                       labels=['0-0.15', '0.15-0.25', '0.25-0.35', '0.35-0.6', '0.6+'])

    # Pivot table: dataset × error_bin
    pivot = df_deg_copy.groupby(['dataset', 'error_bin'])['f1_drop_rel_pct'].mean().unstack()

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                vmin=0, vmax=25,
                cbar_kws={'label': 'Relative F1 Drop (%)'},
                linewidths=0.5, linecolor='gray')
    plt.xlabel('Prior Error Range')
    plt.ylabel('Dataset')
    plt.title('vpu_nomixup_mean_prior: Performance Degradation vs Prior Error')
    plt.tight_layout()

    output_path = Path(output_dir) / "degradation_heatmap.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved degradation heatmap to {output_path}")
    plt.close()


def plot_c_interaction(df, output_dir):
    """Does label frequency affect prior sensitivity?"""

    # Filter for vpu_nomixup_mean_prior only
    df_prior = df[df['method'] == 'vpu_nomixup_mean_prior']

    fig, ax = plt.subplots(figsize=(10, 6))

    c_values = sorted(df_prior['c'].unique())
    colors = ['#e74c3c', '#3498db', '#2ecc71']  # Red, Blue, Green

    for c, color in zip(c_values, colors):
        data_c = df_prior[df_prior['c'] == c]

        # Group by prior_error
        grouped = data_c.groupby('prior_error')['test_f1'].agg(['mean', 'std'])

        if not grouped.empty:
            ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                       label=f'c={c}', color=color, marker='o', markersize=6,
                       linewidth=2, capsize=4)

    ax.set_xlabel('Prior Error |π_method - π_true|')
    ax.set_ylabel('Test F1 Score')
    ax.set_title('Effect of Label Frequency on Prior Sensitivity')
    ax.legend(title='Label Frequency', loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = Path(output_dir) / "label_frequency_interaction.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved label frequency interaction plot to {output_path}")
    plt.close()


def plot_degradation_by_error_bins(df_deg, output_dir):
    """Bar plot showing degradation at different error levels"""

    # Create error bins
    bins = [0, 0.1, 0.2, 0.3, 0.5, 1.0]
    labels = ['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.5', '0.5+']
    df_deg_copy = df_deg.copy()
    df_deg_copy['error_bin'] = pd.cut(df_deg_copy['prior_error'], bins=bins, labels=labels)

    # Calculate mean and std for each bin
    summary = df_deg_copy.groupby('error_bin')['f1_drop_rel_pct'].agg(['mean', 'std', 'count'])

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(summary))
    ax.bar(x, summary['mean'], yerr=summary['std'], capsize=5,
           color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Prior Error Range')
    ax.set_ylabel('Mean F1 Drop (%)')
    ax.set_title('Performance Degradation by Prior Error Level')
    ax.set_xticks(x)
    ax.set_xticklabels(summary.index)
    ax.grid(True, axis='y', alpha=0.3)

    # Add count labels on bars
    for i, (_, row) in enumerate(summary.iterrows()):
        ax.text(i, row['mean'] + row['std'] + 0.5, f"n={int(row['count'])}",
               ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_path = Path(output_dir) / "degradation_by_error_bins.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved degradation by error bins to {output_path}")
    plt.close()


def plot_method_comparison_at_errors(df, output_dir):
    """Compare all 3 methods at different prior error levels"""

    methods = ['vpu_nomixup', 'vpu_nomixup_mean', 'vpu_nomixup_mean_prior']
    method_labels = {
        'vpu_nomixup': 'VPU-NoMixup',
        'vpu_nomixup_mean': 'VPU-NoMixup-Mean',
        'vpu_nomixup_mean_prior': 'VPU-NoMixup-Mean-Prior'
    }

    # Select specific prior error levels to compare
    error_levels = [0.0, 0.1, 0.2, 0.3, 0.5]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(error_levels))
    width = 0.25

    for idx, method in enumerate(methods):
        method_data = df[df['method'] == method]
        means = []
        stds = []

        for error in error_levels:
            # Get data close to this error level (within ±0.05)
            subset = method_data[np.abs(method_data['prior_error'] - error) < 0.05]

            if not subset.empty:
                means.append(subset['test_f1'].mean())
                stds.append(subset['test_f1'].std())
            else:
                means.append(0)
                stds.append(0)

        ax.bar(x + idx * width, means, width, yerr=stds,
               label=method_labels[method], capsize=3)

    ax.set_xlabel('Prior Error Level')
    ax.set_ylabel('Test F1 Score')
    ax.set_title('Method Comparison at Different Prior Error Levels')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'{e:.1f}' for e in error_levels])
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    output_path = Path(output_dir) / "method_comparison_by_error.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved method comparison plot to {output_path}")
    plt.close()


def create_robustness_table(df_deg, output_dir):
    """Create publication-ready summary table"""

    # Group by error bins
    bins = [0, 0.1, 0.2, 0.3, 0.5, 1.0]
    labels = ['≤0.1', '≤0.2', '≤0.3', '≤0.5', '≤1.0']
    df_deg_copy = df_deg.copy()

    table_data = []
    for i, (bin_min, bin_max, label) in enumerate(zip([0, 0, 0, 0, 0], bins[1:], labels)):
        subset = df_deg_copy[df_deg_copy['prior_error'] <= bin_max]

        if not subset.empty:
            table_data.append({
                'Prior Error': label,
                'Mean F1 Drop (%)': f"{subset['f1_drop_rel_pct'].mean():.2f}",
                'Std F1 Drop (%)': f"{subset['f1_drop_rel_pct'].std():.2f}",
                'Max F1 Drop (%)': f"{subset['f1_drop_rel_pct'].max():.2f}",
                'N Experiments': len(subset)
            })

    table_df = pd.DataFrame(table_data)

    # Save as CSV
    output_path = Path(output_dir) / "robustness_summary_table.csv"
    table_df.to_csv(output_path, index=False)
    print(f"✓ Saved summary table to {output_path}")

    # Print LaTeX format
    print("\nLaTeX Table:")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Performance degradation of vpu\\_nomixup\\_mean\\_prior with prior misspecification}")
    print("\\begin{tabular}{lrrrr}")
    print("\\hline")
    print("Prior Error & Mean F1 Drop (\\%) & Std F1 Drop (\\%) & Max F1 Drop (\\%) & N \\\\")
    print("\\hline")
    for _, row in table_df.iterrows():
        print(f"{row['Prior Error']} & {row['Mean F1 Drop (%)']} & {row['Std F1 Drop (%)']} & {row['Max F1 Drop (%)']} & {row['N Experiments']} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")


def main():
    results_dir = "results_robustness"

    # Check if analysis results exist
    full_results_path = Path(results_dir) / "robustness_full_results.csv"
    if not full_results_path.exists():
        print(f"Results not found at {full_results_path}")
        print("Run analysis first: python analysis/analyze_prior_robustness.py")
        return

    # Load results
    print("Loading results...")
    df_full, df_deg = load_results(results_dir)

    # Create output directory for plots
    output_dir = Path(results_dir) / "plots"
    output_dir.mkdir(exist_ok=True)

    print(f"\nGenerating plots to {output_dir}/...")

    # Generate all plots
    plot_robustness_curves(df_full, output_dir)
    plot_degradation_heatmap(df_deg, output_dir)
    plot_c_interaction(df_full, output_dir)
    plot_degradation_by_error_bins(df_deg, output_dir)
    plot_method_comparison_at_errors(df_full, output_dir)

    # Create summary table
    create_robustness_table(df_deg, output_dir)

    print("\n" + "="*80)
    print("All plots generated successfully!")
    print(f"Output directory: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
