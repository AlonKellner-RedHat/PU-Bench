#!/usr/bin/env python3
"""Analyze the limitation of true_prior range in robustness experiments"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_true_prior_distribution():
    """Load all experiments and extract true_prior values"""

    results = []
    results_dir = Path("results_robustness")
    json_files = list(results_dir.glob("seed_*/*.json"))

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
                    method_prior = hyperparams.get('method_prior')
                    true_prior = dataset_info.get('train', {}).get('prior')

                    if all([dataset, seed, c, true_prior]):
                        if method_prior is None:
                            method_prior_value = 'auto'
                        else:
                            method_prior_value = float(method_prior)

                        results.append({
                            'dataset': dataset,
                            'seed': seed,
                            'c': c,
                            'true_prior': true_prior,
                            'method_prior': method_prior_value,
                            'test_ap': best_metrics.get('test_ap'),
                        })
                    break
        except Exception as e:
            continue

    return pd.DataFrame(results)


def plot_true_prior_distribution(df, output_dir='results_robustness'):
    """Plot distribution of true_prior values"""

    output_dir = Path(output_dir)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Histogram of true priors
    ax = axes[0, 0]
    unique_priors = df.groupby(['dataset', 'c'])['true_prior'].first()

    ax.hist(unique_priors, bins=20, edgecolor='black', alpha=0.7, color='#3498db')
    ax.axvline(unique_priors.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {unique_priors.mean():.3f}')
    ax.axvline(0.5, color='green', linestyle=':', linewidth=2, label='0.5 (balanced)')

    ax.set_xlabel('True Prior (π)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of True Prior Values in Experiments')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. True prior by dataset
    ax = axes[0, 1]

    dataset_priors = df.groupby(['dataset', 'c'])['true_prior'].first().reset_index()
    datasets = sorted(dataset_priors['dataset'].unique())

    for dataset in datasets:
        subset = dataset_priors[dataset_priors['dataset'] == dataset]
        ax.scatter(subset['c'], subset['true_prior'], label=dataset, s=100, alpha=0.7)

    ax.set_xlabel('Label Frequency (c)')
    ax.set_ylabel('True Prior (π)')
    ax.set_title('True Prior by Dataset and Label Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.1, 1.1)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)

    # 3. Coverage plot: method_prior vs true_prior
    ax = axes[1, 0]

    # Get unique combinations
    unique_combos = df.groupby(['dataset', 'c']).agg({
        'true_prior': 'first',
        'method_prior': lambda x: list(x.unique())
    }).reset_index()

    # Plot grid
    true_priors = unique_combos['true_prior'].unique()
    method_prior_values = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]

    # Shade the tested region
    ax.axvspan(true_priors.min(), true_priors.max(), alpha=0.2, color='green', label='Tested true prior range')

    # Plot all method_prior values tested
    for tp in true_priors:
        ax.scatter([tp]*len(method_prior_values), method_prior_values,
                  alpha=0.5, color='blue', s=30)

    # Diagonal line (where method_prior = true_prior)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='method_prior = true_prior')

    # Highlight untested regions
    ax.axvspan(0, true_priors.min(), alpha=0.2, color='red', label='Untested true prior region')
    ax.axvspan(true_priors.max(), 1.0, alpha=0.2, color='red')

    ax.set_xlabel('True Prior (π)')
    ax.set_ylabel('Method Prior (tested values)')
    ax.set_title('Coverage: Method Prior vs True Prior')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)

    # 4. Performance when method_prior matches vs mismatches true_prior range
    ax = axes[1, 1]

    # Define in-range vs out-of-range
    true_prior_min = true_priors.min()
    true_prior_max = true_priors.max()

    df_numeric = df[df['method_prior'] != 'auto'].copy()
    df_numeric['method_prior_float'] = df_numeric['method_prior'].astype(float)

    df_numeric['in_true_range'] = df_numeric['method_prior_float'].apply(
        lambda x: true_prior_min <= x <= true_prior_max
    )

    # Box plot
    data_in = df_numeric[df_numeric['in_true_range']]['test_ap'].dropna()
    data_out = df_numeric[~df_numeric['in_true_range']]['test_ap'].dropna()
    data_auto = df[df['method_prior'] == 'auto']['test_ap'].dropna()

    positions = [1, 2, 3]
    data_to_plot = [data_in, data_out, data_auto]
    labels = ['In-range\n[0.42-0.71]', 'Out-of-range\n[0.1-0.3, 0.9-1.0]', 'Auto\n(true prior)']

    bp = ax.boxplot(data_to_plot, positions=positions, labels=labels,
                    patch_artist=True, widths=0.6)

    # Color boxes
    colors = ['green', 'red', 'blue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax.set_ylabel('Test AP')
    ax.set_title('Performance: Method Prior In-Range vs Out-of-Range')
    ax.grid(True, axis='y', alpha=0.3)

    # Add statistics
    for i, (data, label) in enumerate(zip(data_to_plot, labels), 1):
        mean_val = data.mean()
        ax.text(i, mean_val, f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    output_path = output_dir / "true_prior_range_limitation.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization to {output_path}")
    plt.close()


def analyze_performance_by_range_match(df):
    """Analyze if performance depends on whether method_prior is in the true_prior range"""

    # Get true prior range
    true_priors = df.groupby(['dataset', 'c'])['true_prior'].first()
    true_prior_min = true_priors.min()
    true_prior_max = true_priors.max()

    print(f"True prior range in data: [{true_prior_min:.3f}, {true_prior_max:.3f}]")
    print()

    # Categorize method_priors
    df_numeric = df[df['method_prior'] != 'auto'].copy()
    df_numeric['method_prior_float'] = df_numeric['method_prior'].astype(float)

    df_numeric['range_category'] = df_numeric['method_prior_float'].apply(
        lambda x: 'in-range' if true_prior_min <= x <= true_prior_max else 'out-of-range'
    )

    # Compare performance
    print("Performance by range match:")
    print()

    summary = df_numeric.groupby('range_category')['test_ap'].agg(['mean', 'std', 'count'])
    print(summary)
    print()

    # Statistical test
    from scipy import stats
    in_range = df_numeric[df_numeric['range_category'] == 'in-range']['test_ap'].dropna()
    out_range = df_numeric[df_numeric['range_category'] == 'out-of-range']['test_ap'].dropna()

    t_stat, p_value = stats.ttest_ind(in_range, out_range)
    print(f"T-test: t={t_stat:.3f}, p={p_value:.4f}")

    if p_value < 0.05:
        print("✓ Significant difference (p<0.05)")
    else:
        print("✗ No significant difference (p>=0.05)")

    print()

    # Compare with auto
    auto_ap = df[df['method_prior'] == 'auto']['test_ap'].mean()
    print(f"Auto (true prior) mean AP: {auto_ap:.4f}")
    print(f"In-range mean AP: {in_range.mean():.4f} ({in_range.mean()-auto_ap:+.4f} vs auto)")
    print(f"Out-of-range mean AP: {out_range.mean():.4f} ({out_range.mean()-auto_ap:+.4f} vs auto)")


def main():
    print("=" * 80)
    print("Analysis: True Prior Range Limitation")
    print("=" * 80)
    print()

    # Load data
    df = load_true_prior_distribution()
    print(f"✓ Loaded {len(df)} experiments")
    print()

    # Analyze distribution
    unique_priors = df.groupby(['dataset', 'c'])['true_prior'].first()

    print("True Prior Distribution:")
    print(f"  Min: {unique_priors.min():.3f}")
    print(f"  Max: {unique_priors.max():.3f}")
    print(f"  Mean: {unique_priors.mean():.3f}")
    print(f"  Median: {unique_priors.median():.3f}")
    print(f"  Std: {unique_priors.std():.3f}")
    print()

    print("Unique true prior values:")
    for val in sorted(unique_priors.unique()):
        print(f"  {val:.3f}")
    print()

    # Show by dataset
    print("By dataset:")
    by_dataset = df.groupby('dataset')['true_prior'].agg(['min', 'max', 'mean'])
    print(by_dataset.to_string())
    print()

    # Key limitation
    print("=" * 80)
    print("KEY LIMITATION")
    print("=" * 80)
    print()
    print("The robustness experiments test:")
    print(f"  ✓ Method priors: 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0 (wide range)")
    print(f"  ✗ True priors: [{unique_priors.min():.3f}, {unique_priors.max():.3f}] (NARROW range)")
    print()
    print("Implications:")
    print(f"  1. We don't know how methods perform when true_prior < {unique_priors.min():.3f}")
    print(f"  2. We don't know how methods perform when true_prior > {unique_priors.max():.3f}")
    print("  3. 'Best default' = 0.5 may just mean 'close to typical true_prior'")
    print("  4. Recommendations may not generalize to imbalanced datasets")
    print()

    # Analyze performance
    print("=" * 80)
    print("Performance Analysis")
    print("=" * 80)
    print()

    analyze_performance_by_range_match(df)

    # Create visualization
    print()
    print("Creating visualization...")
    plot_true_prior_distribution(df)

    print()
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()
    print("The 'robust default prior' analysis is LIMITED to datasets with")
    print(f"true priors in the range [{unique_priors.min():.3f}, {unique_priors.max():.3f}].")
    print()
    print("For datasets outside this range (highly imbalanced), the recommendations")
    print("may not hold. Further experiments needed with imbalanced datasets.")


if __name__ == "__main__":
    main()
