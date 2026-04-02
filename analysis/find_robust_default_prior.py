#!/usr/bin/env python3
"""Find the most robust default prior value when true prior is unknown

Instead of asking "what's optimal for a given true prior?", we ask:
"Which method_prior value works best ON AVERAGE across different true priors?"

This is the value you'd use when you have no information about the prior.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_all_robustness_data():
    """Load all robustness experiments

    Includes both vpu_nomixup_mean_prior and vpu_nomixup_mean (equivalent to prior=1.0)
    """
    results = []

    results_dir = Path("results_robustness")
    json_files = list(results_dir.glob("seed_*/*.json"))

    print(f"Loading {len(json_files)} result files...")

    for json_file in json_files:
        try:
            with open(json_file) as f:
                data = json.load(f)

            for method_name, method_data in data.get('runs', {}).items():
                if method_name not in ['vpu_nomixup_mean_prior', 'vpu_nomixup_mean']:
                    continue

                hyperparams = method_data.get('hyperparameters', {})
                dataset_info = method_data.get('dataset', {})
                best_metrics = method_data.get('best', {}).get('metrics', {})

                dataset = hyperparams.get('dataset_class')
                seed = hyperparams.get('seed')
                c = hyperparams.get('labeled_ratio')
                true_prior = dataset_info.get('train', {}).get('prior')

                # Handle method_prior: vpu_nomixup_mean is equivalent to prior=1.0
                if method_name == 'vpu_nomixup_mean':
                    method_prior = 1.0
                else:
                    method_prior = hyperparams.get('method_prior')

                # Get metrics
                test_ap = best_metrics.get('test_ap')
                test_auc = best_metrics.get('test_auc')
                test_max_f1 = best_metrics.get('test_max_f1')
                test_f1 = best_metrics.get('test_f1')

                if all(v is not None for v in [dataset, seed, c, true_prior, test_ap]):
                    # Normalize method_prior representation
                    if method_prior is None:
                        method_prior_value = 'auto'
                        prior_error = 0.0
                    else:
                        method_prior_value = float(method_prior)
                        prior_error = abs(method_prior_value - true_prior)

                    results.append({
                        'dataset': dataset,
                        'seed': seed,
                        'c': c,
                        'method_prior': method_prior_value,
                        'true_prior': true_prior,
                        'prior_error': prior_error,
                        'test_ap': test_ap,
                        'test_auc': test_auc,
                        'test_max_f1': test_max_f1,
                        'test_f1': test_f1,
                    })

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

    return pd.DataFrame(results)


def analyze_method_prior_robustness(df, metric='test_ap'):
    """For each method_prior value, compute average performance across all true priors"""

    results = []

    # For each method_prior value and c value
    for c in sorted(df['c'].unique()):
        df_c = df[df['c'] == c]

        # Sort method_prior values (auto first, then numeric)
        method_priors = df['method_prior'].unique()
        method_priors_sorted = ['auto'] + sorted([p for p in method_priors if p != 'auto'],
                                                  key=lambda x: float(x))

        for method_prior in method_priors_sorted:
            subset = df_c[df_c['method_prior'] == method_prior]

            if subset.empty:
                continue

            # Statistics across all (dataset, seed, true_prior) combinations
            mean_perf = subset[metric].mean()
            std_perf = subset[metric].std()
            min_perf = subset[metric].min()
            max_perf = subset[metric].max()
            count = len(subset)

            # Average prior error
            mean_error = subset['prior_error'].mean()

            results.append({
                'c': c,
                'method_prior': method_prior,
                'mean_performance': mean_perf,
                'std_performance': std_perf,
                'min_performance': min_perf,
                'max_performance': max_perf,
                'mean_prior_error': mean_error,
                'count': count,
            })

    return pd.DataFrame(results)


def find_best_default_per_c(df_robust, metric_name='mean_performance'):
    """Find the best default method_prior for each c value"""

    results = []

    for c in sorted(df_robust['c'].unique()):
        subset = df_robust[df_robust['c'] == c]

        # Find the method_prior with highest mean performance
        best_idx = subset[metric_name].idxmax()
        best_row = subset.loc[best_idx]

        # Also get auto performance for comparison
        auto_row = subset[subset['method_prior'] == 'auto']
        auto_perf = auto_row[metric_name].iloc[0] if not auto_row.empty else np.nan

        results.append({
            'c': c,
            'best_default': best_row['method_prior'],
            'best_mean_performance': best_row['mean_performance'],
            'best_std_performance': best_row['std_performance'],
            'auto_mean_performance': auto_perf,
            'improvement_over_auto': best_row['mean_performance'] - auto_perf if not np.isnan(auto_perf) else np.nan,
        })

    return pd.DataFrame(results)


def plot_robustness_by_method_prior(df_robust, metric='mean_performance', output_dir='results_robustness'):
    """Plot performance of each method_prior value across c values"""

    output_dir = Path(output_dir)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    c_values = sorted(df_robust['c'].unique())

    for ax, c in zip(axes, c_values):
        subset = df_robust[df_robust['c'] == c].copy()

        # Sort by method_prior value (handle auto separately)
        subset['method_prior_for_sort'] = subset['method_prior'].apply(
            lambda x: -0.05 if x == 'auto' else float(x)
        )
        subset = subset.sort_values('method_prior_for_sort')

        # Convert auto to numeric for plotting
        subset['method_prior_numeric'] = subset['method_prior'].apply(
            lambda x: -0.05 if x == 'auto' else float(x)
        )

        # Plot mean with error bars
        ax.errorbar(subset['method_prior_numeric'], subset['mean_performance'],
                   yerr=subset['std_performance'], marker='o', markersize=8,
                   linewidth=2, capsize=5, color='#2ca02c')

        # Highlight auto
        auto_row = subset[subset['method_prior'] == 'auto']
        if not auto_row.empty:
            ax.scatter([-0.05], [auto_row['mean_performance'].iloc[0]],
                      color='red', s=150, marker='*', zorder=10,
                      label='auto (true prior)')

        # Find best
        best_idx = subset['mean_performance'].idxmax()
        best_row = subset.loc[best_idx]
        ax.scatter([best_row['method_prior_numeric']], [best_row['mean_performance']],
                  color='blue', s=150, marker='D', zorder=10,
                  label=f'best: {best_row["method_prior"]}')

        ax.set_xlabel('Method Prior Value')
        ax.set_ylabel('Mean Average Precision (AP)')
        ax.set_title(f'c = {c}')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Set x-axis ticks
        ticks = [-0.05] + list(range(0, 11, 1))  # Include 10 for 1.0
        labels = ['auto'] + [f'{x/10:.1f}' for x in range(0, 11, 1)]
        ax.set_xticks([t/10 if t >= 0 else t for t in ticks])
        ax.set_xticklabels(labels, rotation=45)

    plt.tight_layout()
    output_path = output_dir / "default_prior_robustness.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")
    plt.close()


def plot_performance_heatmap(df_robust, output_dir='results_robustness'):
    """Heatmap of method_prior performance across c values"""

    output_dir = Path(output_dir)

    # Pivot table: c × method_prior
    pivot = df_robust.pivot_table(
        values='mean_performance',
        index='c',
        columns='method_prior',
        aggfunc='mean'
    )

    # Reorder columns: auto first, then numeric
    cols = ['auto'] + sorted([c for c in pivot.columns if c != 'auto'],
                            key=lambda x: float(x))
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(12, 5))

    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn',
                vmin=pivot.min().min(), vmax=pivot.max().max(),
                cbar_kws={'label': 'Mean Average Precision (AP)'},
                linewidths=0.5, linecolor='gray', ax=ax)

    ax.set_xlabel('Method Prior Value')
    ax.set_ylabel('Label Frequency (c)')
    ax.set_title('Average Performance of Each Method Prior Value')

    plt.tight_layout()
    output_path = output_dir / "default_prior_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved heatmap to {output_path}")
    plt.close()


def analyze_prior_range_robustness(df):
    """Group method priors into ranges and find most robust range"""

    def categorize_prior(p):
        if p == 'auto':
            return 'auto'
        p = float(p)
        if p < 0.25:
            return 'low (0.1-0.2)'
        elif p < 0.4:
            return 'low-mid (0.3)'
        elif p < 0.6:
            return 'mid (0.5)'
        elif p < 0.8:
            return 'mid-high (0.7)'
        elif p < 0.95:
            return 'high (0.9)'
        else:
            return 'very high (1.0)'

    df['prior_range'] = df['method_prior'].apply(categorize_prior)

    results = []

    for c in sorted(df['c'].unique()):
        df_c = df[df['c'] == c]

        for range_name in ['auto', 'low (0.1-0.2)', 'low-mid (0.3)', 'mid (0.5)',
                          'mid-high (0.7)', 'high (0.9)', 'very high (1.0)']:
            subset = df_c[df_c['prior_range'] == range_name]

            if subset.empty:
                continue

            results.append({
                'c': c,
                'prior_range': range_name,
                'mean_ap': subset['test_ap'].mean(),
                'std_ap': subset['test_ap'].std(),
                'count': len(subset),
            })

    return pd.DataFrame(results)


def main():
    print("=" * 80)
    print("Finding Robust Default Prior (When True Prior Unknown)")
    print("=" * 80)
    print()

    # Load data
    print("Loading all robustness data...")
    df = load_all_robustness_data()
    print(f"✓ Loaded {len(df)} experiments")
    print()

    # Analyze robustness of each method_prior value
    print("Analyzing average performance of each method_prior value...")
    df_robust = analyze_method_prior_robustness(df, metric='test_ap')
    print()

    # Find best default for each c
    print("=" * 80)
    print("Best Default Method Prior for Each Label Frequency")
    print("=" * 80)
    print()

    df_best = find_best_default_per_c(df_robust)
    print(df_best.to_string(index=False))
    print()

    # Save results
    output_dir = Path("results_robustness")
    df_robust.to_csv(output_dir / "method_prior_robustness.csv", index=False)
    df_best.to_csv(output_dir / "best_default_prior.csv", index=False)
    print(f"✓ Saved to {output_dir / 'method_prior_robustness.csv'}")
    print(f"✓ Saved to {output_dir / 'best_default_prior.csv'}")
    print()

    # Detailed table for each c
    print("=" * 80)
    print("Detailed Performance by Method Prior Value")
    print("=" * 80)
    print()

    for c in sorted(df_robust['c'].unique()):
        print(f"\n### c = {c} ###")
        subset = df_robust[df_robust['c'] == c].copy()
        subset = subset.sort_values('mean_performance', ascending=False)
        print(subset[['method_prior', 'mean_performance', 'std_performance',
                     'mean_prior_error', 'count']].to_string(index=False))

    print()

    # Create visualizations
    print("Creating visualizations...")
    plot_robustness_by_method_prior(df_robust, output_dir=output_dir)
    plot_performance_heatmap(df_robust, output_dir=output_dir)
    print()

    # Analyze by range
    print("=" * 80)
    print("Performance by Prior Range")
    print("=" * 80)
    print()

    df_range = analyze_prior_range_robustness(df)
    for c in sorted(df_range['c'].unique()):
        print(f"\n### c = {c} ###")
        subset = df_range[df_range['c'] == c].sort_values('mean_ap', ascending=False)
        print(subset.to_string(index=False))

    print()

    # Overall recommendation
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print()

    # Find the method_prior that works best on average across all c values
    overall_best = df_robust.groupby('method_prior')['mean_performance'].mean().idxmax()
    overall_best_perf = df_robust.groupby('method_prior')['mean_performance'].mean().max()
    overall_auto_perf = df_robust[df_robust['method_prior'] == 'auto']['mean_performance'].mean()

    print(f"Best universal default: {overall_best}")
    print(f"  Mean AP across all c: {overall_best_perf:.4f}")
    print(f"  Auto (true prior) AP: {overall_auto_perf:.4f}")
    print(f"  Difference: {overall_best_perf - overall_auto_perf:+.4f}")
    print()

    # Per-c recommendations
    print("Per-c recommendations:")
    for _, row in df_best.iterrows():
        improvement = row['improvement_over_auto']
        if pd.notna(improvement):
            print(f"  c={row['c']:.1f}: Use {row['best_default']} "
                  f"(AP={row['best_mean_performance']:.4f}, "
                  f"{improvement:+.4f} vs auto)")
    print()

    print("Practical advice:")
    print("  - If you have NO information about true prior:")
    print(f"    → Use method_prior = {overall_best}")
    print("  - If you know label frequency (c):")
    print("    → Use per-c recommendation above")
    print("  - If you have ANY labeled data:")
    print("    → Compute true prior and use auto (always competitive)")


if __name__ == "__main__":
    main()
