#!/usr/bin/env python3
"""Analyze Cartesian Product Prior Experiments

This script analyzes experiments where:
- True prior was simulated via training set resampling
- Method prior was varied as loss parameter
- Full cartesian product tested: true_prior × method_prior
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_cartesian_results(results_dir="results_cartesian"):
    """Load results from cartesian experiments"""
    results = []

    results_path = Path(results_dir)
    json_files = list(results_path.glob("seed_*/*.json"))

    print(f"Loading {len(json_files)} result files...")

    for json_file in json_files:
        try:
            with open(json_file) as f:
                data = json.load(f)

            for method_name, method_data in data.get('runs', {}).items():
                if method_name == 'vpu_nomixup_mean_prior':
                    hyperparams = method_data.get('hyperparameters', {})
                    dataset_info = method_data.get('dataset', {})
                    best_metrics = method_data.get('best', {}).get('metrics', {})

                    dataset = hyperparams.get('dataset_class')
                    seed = hyperparams.get('seed')
                    c = hyperparams.get('labeled_ratio')

                    # True prior from config (target)
                    true_prior_target = hyperparams.get('target_prevalence_train')
                    # True prior from data (actual measured)
                    true_prior_actual = dataset_info.get('train', {}).get('prior')

                    # Method prior from config
                    method_prior = hyperparams.get('method_prior')

                    if all([dataset, seed, c, true_prior_actual]):
                        if method_prior is None:
                            method_prior_value = 'auto'
                        else:
                            method_prior_value = float(method_prior)

                        results.append({
                            'dataset': dataset,
                            'seed': seed,
                            'c': c,
                            'true_prior_target': true_prior_target,
                            'true_prior_actual': true_prior_actual,
                            'method_prior': method_prior_value,
                            'prior_error': abs(method_prior_value - true_prior_actual) if method_prior_value != 'auto' else 0,
                            'test_ap': best_metrics.get('test_ap'),
                            'test_f1': best_metrics.get('test_f1'),
                            'test_auc': best_metrics.get('test_auc'),
                            'test_ece': best_metrics.get('test_ece'),
                            'convergence_epoch': method_data.get('best', {}).get('epoch'),
                        })
                    break
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue

    df = pd.DataFrame(results)
    print(f"✓ Loaded {len(df)} experiments")
    return df


def create_heatmap_matrix(df, metric='test_ap', output_dir='results_cartesian'):
    """Create heatmap: true_prior × method_prior → performance"""

    output_dir = Path(output_dir)

    # Aggregate over datasets and seeds
    pivot = df.pivot_table(
        values=metric,
        index='true_prior_actual',
        columns='method_prior',
        aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=(12, 9))

    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=pivot.min().min(), vmax=pivot.max().max(),
                linewidths=0.5, linecolor='gray', ax=ax,
                cbar_kws={'label': metric.upper()})

    ax.set_xlabel('Method Prior (π_method)', fontsize=12)
    ax.set_ylabel('True Prior (π_true, actual)', fontsize=12)
    ax.set_title(f'Performance ({metric.upper()}) by Prior Combination\n(Averaged across all datasets and seeds)', fontsize=14)

    plt.tight_layout()
    output_path = output_dir / f"heatmap_{metric}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved heatmap to {output_path}")
    plt.close()


def find_optimal_method_prior_by_true_prior(df, metric='test_ap'):
    """For each true_prior, find the best method_prior"""

    results = []

    true_priors = sorted(df['true_prior_actual'].unique())

    for true_prior in true_priors:
        subset = df[df['true_prior_actual'].round(1) == round(true_prior, 1)]

        if len(subset) == 0:
            continue

        # Group by method_prior, get mean performance
        perf_by_method = subset.groupby('method_prior')[metric].agg(['mean', 'std', 'count'])

        best_method_prior = perf_by_method['mean'].idxmax()
        best_value = perf_by_method.loc[best_method_prior, 'mean']

        # Get performance when π_method = π_true (diagonal)
        match_subset = subset[subset['method_prior'].astype(float).round(1) == round(true_prior, 1)]
        if len(match_subset) > 0:
            match_value = match_subset[metric].mean()
        else:
            match_value = np.nan

        results.append({
            'true_prior': true_prior,
            'best_method_prior': best_method_prior,
            f'best_{metric}': best_value,
            f'match_{metric}': match_value,
            'improvement': best_value - match_value if not np.isnan(match_value) else np.nan,
        })

    return pd.DataFrame(results)


def plot_robustness_curves(df, metric='test_ap', output_dir='results_cartesian'):
    """For each true_prior, plot performance vs method_prior"""

    output_dir = Path(output_dir)

    true_priors = sorted(df['true_prior_actual'].unique())

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, true_prior in enumerate(true_priors[:6]):
        ax = axes[idx]
        subset = df[df['true_prior_actual'].round(1) == round(true_prior, 1)]

        # Group by method_prior
        grouped = subset.groupby('method_prior')[metric].agg(['mean', 'std'])

        method_priors = grouped.index.tolist()
        means = grouped['mean'].values
        stds = grouped['std'].values

        ax.plot(method_priors, means, 'o-', linewidth=2, markersize=8)
        ax.fill_between(method_priors, means - stds, means + stds, alpha=0.3)

        # Mark the diagonal (where method_prior = true_prior)
        if true_prior in method_priors:
            idx_match = method_priors.index(true_prior)
            ax.plot(true_prior, means[idx_match], 'r*', markersize=15,
                   label=f'π_method = π_true = {true_prior:.1f}')

        ax.set_xlabel('Method Prior (π_method)')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'True Prior π_true ≈ {true_prior:.2f}')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    output_path = output_dir / f"robustness_curves_{metric}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved robustness curves to {output_path}")
    plt.close()


def analyze_diagonal_performance(df, metric='test_ap'):
    """Analyze performance when π_method = π_true (diagonal)"""

    # Filter to approximately matching priors (within 0.1)
    diagonal_rows = []
    for _, row in df.iterrows():
        if row['method_prior'] == 'auto':
            continue
        mp = float(row['method_prior'])
        tp = float(row['true_prior_actual'])
        if abs(mp - tp) < 0.15:  # Allow 0.15 tolerance due to resampling approximation
            diagonal_rows.append(row)

    diagonal_df = pd.DataFrame(diagonal_rows)

    print("\n" + "="*80)
    print("Diagonal Performance (π_method ≈ π_true)")
    print("="*80)
    print(f"Number of diagonal experiments: {len(diagonal_df)}")
    print(f"Mean {metric}: {diagonal_df[metric].mean():.4f} ± {diagonal_df[metric].std():.4f}")
    print()

    # Compare to off-diagonal
    off_diagonal_df = df[~df.index.isin(diagonal_df.index)]
    off_diagonal_df = off_diagonal_df[off_diagonal_df['method_prior'] != 'auto']

    print(f"Off-diagonal mean {metric}: {off_diagonal_df[metric].mean():.4f} ± {off_diagonal_df[metric].std():.4f}")

    # Statistical test
    t_stat, p_value = stats.ttest_ind(diagonal_df[metric].dropna(), off_diagonal_df[metric].dropna())
    print(f"T-test: t={t_stat:.3f}, p={p_value:.4f}")
    if p_value < 0.05:
        print("✓ Significant difference (p<0.05)")
    else:
        print("✗ No significant difference (p>=0.05)")


def generate_summary_table(df, output_dir='results_cartesian'):
    """Generate summary statistics table"""

    output_dir = Path(output_dir)

    # Overall statistics
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)

    print("\nBy True Prior:")
    summary_by_true = df.groupby('true_prior_actual').agg({
        'test_ap': ['mean', 'std', 'min', 'max'],
        'test_f1': ['mean', 'std'],
        'test_ece': ['mean', 'std'],
    }).round(4)
    print(summary_by_true.to_string())

    print("\nBy Method Prior:")
    summary_by_method = df.groupby('method_prior').agg({
        'test_ap': ['mean', 'std', 'min', 'max'],
        'test_f1': ['mean', 'std'],
        'test_ece': ['mean', 'std'],
    }).round(4)
    print(summary_by_method.to_string())

    # Save to CSV
    summary_by_true.to_csv(output_dir / "summary_by_true_prior.csv")
    summary_by_method.to_csv(output_dir / "summary_by_method_prior.csv")
    print(f"\n✓ Saved summary tables to {output_dir}")


def main():
    print("=" * 80)
    print("Cartesian Product Prior Experiments - Analysis")
    print("=" * 80)
    print()

    # Load data
    df = load_cartesian_results()

    if len(df) == 0:
        print("No results found!")
        return

    print(f"\nDatasets: {df['dataset'].unique().tolist()}")
    print(f"Seeds: {sorted(df['seed'].unique().tolist())}")
    print(f"Label frequencies (c): {sorted(df['c'].unique().tolist())}")
    print(f"True priors (actual): {sorted(df['true_prior_actual'].unique().tolist())}")
    print(f"Method priors: {sorted(df['method_prior'].unique().tolist())}")
    print()

    # Generate analyses
    print("Generating visualizations...")
    create_heatmap_matrix(df, metric='test_ap')
    create_heatmap_matrix(df, metric='test_f1')
    create_heatmap_matrix(df, metric='test_ece')

    plot_robustness_curves(df, metric='test_ap')
    plot_robustness_curves(df, metric='test_f1')

    # Optimal priors
    print("\n" + "="*80)
    print("Optimal Method Prior for Each True Prior")
    print("="*80)
    optimal_df = find_optimal_method_prior_by_true_prior(df, metric='test_ap')
    print(optimal_df.to_string(index=False))
    optimal_df.to_csv("results_cartesian/optimal_method_prior_by_true_prior.csv", index=False)

    # Diagonal analysis
    analyze_diagonal_performance(df, metric='test_ap')

    # Summary tables
    generate_summary_table(df)

    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print("\nGenerated files:")
    print("  - heatmap_test_ap.png")
    print("  - heatmap_test_f1.png")
    print("  - heatmap_test_ece.png")
    print("  - robustness_curves_test_ap.png")
    print("  - robustness_curves_test_f1.png")
    print("  - optimal_method_prior_by_true_prior.csv")
    print("  - summary_by_true_prior.csv")
    print("  - summary_by_method_prior.csv")


if __name__ == "__main__":
    main()
