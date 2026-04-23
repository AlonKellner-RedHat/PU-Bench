#!/usr/bin/env python3
"""
Phase 3 Analysis (Seed 42): Method Prior Comparison

Analyzes Phase 3 results focusing on:
1. method_prior comparison: 0.69 vs 0.5 vs auto
2. Performance across label frequency (c) and true prior (π) grid
3. General insights on VPU variants and baselines
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Results directory
RESULTS_DIR = Path("results_phase3/seed_42")
PLOTS_DIR = Path("analysis/plots/phase3_seed42")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
DATASETS = ["MNIST", "FashionMNIST", "IMDB", "20News", "Mushrooms", "Spambase", "Connect4"]
C_VALUES = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
PI_VALUES = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]

METHOD_LABELS = {
    "vpu": "VPU",
    "vpu_nomixup": "VPU-nomix",
    "vpu_mean_prior": "VPU-MP",
    "vpu_nomixup_mean_prior": "VPU-nomix-MP",
    "oracle_bce": "Oracle",
    "pn_naive": "PN-Naive",
}


def parse_filename(filename):
    """Extract dataset, c, π, and method_prior from filename."""
    parts = filename.stem.split("_")

    # Dataset name (first part before "case-control")
    dataset = parts[0]

    # Extract c value (look for pattern c0.XX in parts)
    c_str = [p for p in parts if p.startswith("c") and p[1:2].isdigit()][0]
    c = float(c_str[1:])

    # Extract true prior
    pi_str = [p for p in parts if p.startswith("trueprior")][0]
    pi = float(pi_str[9:])

    # Extract method_prior if present
    if "methodprior" in filename.stem:
        if "methodprior_auto" in filename.stem:
            method_prior = "auto"
        else:
            mp_str = [p for p in parts if p.startswith("methodprior")][0]
            method_prior = mp_str[11:]  # Remove "methodprior" prefix
    else:
        method_prior = None

    return dataset, c, pi, method_prior


def load_phase3_results():
    """Load all Phase 3 results into a structured DataFrame."""
    records = []

    for json_file in RESULTS_DIR.glob("*.json"):
        dataset, c, pi, method_prior = parse_filename(json_file)

        with open(json_file) as f:
            data = json.load(f)

        # Extract metrics for each method in this file
        for method_key, method_data in data.get("runs", {}).items():
            if "best" not in method_data or "metrics" not in method_data["best"]:
                continue

            metrics = method_data["best"]["metrics"]

            # Determine full method identifier
            if method_key in ["vpu_mean_prior", "vpu_nomixup_mean_prior"]:
                method_id = f"{method_key}_{method_prior}"
            else:
                method_id = method_key

            record = {
                "dataset": dataset,
                "c": c,
                "pi": pi,
                "method": method_id,
                "base_method": method_key,
                "method_prior": method_prior,
                "test_auc": metrics.get("test_auc"),
                "test_f1": metrics.get("test_f1"),
                "test_accuracy": metrics.get("test_accuracy"),
                "test_precision": metrics.get("test_precision"),
                "test_recall": metrics.get("test_recall"),
                "epochs": method_data.get("epochs"),
            }
            records.append(record)

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} method runs from Phase 3 seed 42")
    print(f"Datasets: {df['dataset'].nunique()}")
    print(f"Unique (c, π) pairs: {df.groupby(['c', 'pi']).ngroups}")
    print(f"Methods: {df['method'].nunique()}")
    print(f"\nMethod breakdown:")
    print(df['method'].value_counts().sort_index())

    return df


def create_heatmap_comparison(df, metric="test_auc"):
    """Create heatmaps comparing method_prior variants across π and c."""
    # Filter to mean-prior methods only
    mp_methods = df[df['base_method'].isin(['vpu_mean_prior', 'vpu_nomixup_mean_prior'])].copy()

    for base_method in ['vpu_mean_prior', 'vpu_nomixup_mean_prior']:
        method_data = mp_methods[mp_methods['base_method'] == base_method]

        # Average across datasets
        avg_data = method_data.groupby(['c', 'pi', 'method_prior'])[metric].mean().reset_index()

        # Create pivot tables for each prior value
        pivots = {}
        for prior in ['auto', '0.5', '0.69']:
            prior_data = avg_data[avg_data['method_prior'] == prior]
            pivot = prior_data.pivot(index='pi', columns='c', values=metric)
            pivots[prior] = pivot

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # Heatmaps for each prior
        for idx, (prior, pivot) in enumerate(pivots.items()):
            ax = axes[idx // 2, idx % 2]
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                       vmin=0.5, vmax=1.0, ax=ax, cbar_kws={'label': metric})
            ax.set_title(f'{METHOD_LABELS[base_method]}({prior})')
            ax.set_xlabel('Label Frequency (c)')
            ax.set_ylabel('True Prior (π)')

        # Difference heatmap: 0.69 - 0.5
        ax = axes[1, 1]
        diff = pivots['0.69'] - pivots['0.5']
        sns.heatmap(diff, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                   vmin=-0.1, vmax=0.1, ax=ax, cbar_kws={'label': f'Δ{metric} (0.69 - 0.5)'})
        ax.set_title(f'{METHOD_LABELS[base_method]}: 0.69 vs 0.5 Difference')
        ax.set_xlabel('Label Frequency (c)')
        ax.set_ylabel('True Prior (π)')

        plt.suptitle(f'{METHOD_LABELS[base_method]}: {metric.upper()} Across Prior Grid (Avg over Datasets)',
                    fontsize=14, y=1.00)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f'{base_method}_heatmap_comparison_{metric}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Created heatmap comparison for {base_method}")


def analyze_prior_performance(df, metric="test_auc"):
    """Analyze which method_prior performs best across different conditions."""
    # Filter to mean-prior methods
    mp_methods = df[df['base_method'].isin(['vpu_mean_prior', 'vpu_nomixup_mean_prior'])].copy()

    results = []

    for base_method in ['vpu_mean_prior', 'vpu_nomixup_mean_prior']:
        method_data = mp_methods[mp_methods['base_method'] == base_method]

        # Group by dataset, c, pi and compare priors
        for (dataset, c, pi), group in method_data.groupby(['dataset', 'c', 'pi']):
            if len(group) < 3:  # Need all 3 priors
                continue

            # Get performance for each prior
            perf = {}
            for prior in ['auto', '0.5', '0.69']:
                prior_data = group[group['method_prior'] == prior]
                if len(prior_data) > 0:
                    perf[prior] = prior_data[metric].iloc[0]

            if len(perf) == 3:
                best_prior = max(perf, key=perf.get)
                results.append({
                    'base_method': base_method,
                    'dataset': dataset,
                    'c': c,
                    'pi': pi,
                    'auto': perf['auto'],
                    '0.5': perf['0.5'],
                    '0.69': perf['0.69'],
                    'best_prior': best_prior,
                    'best_value': perf[best_prior],
                })

    results_df = pd.DataFrame(results)

    # Summary statistics
    print(f"\n{'='*80}")
    print(f"METHOD PRIOR PERFORMANCE ANALYSIS ({metric.upper()})")
    print(f"{'='*80}\n")

    for base_method in ['vpu_mean_prior', 'vpu_nomixup_mean_prior']:
        method_results = results_df[results_df['base_method'] == base_method]

        print(f"\n{METHOD_LABELS[base_method]}:")
        print(f"  Total configurations: {len(method_results)}")

        # Count wins
        wins = method_results['best_prior'].value_counts()
        print(f"\n  Wins by prior:")
        for prior in ['auto', '0.5', '0.69']:
            count = wins.get(prior, 0)
            pct = 100 * count / len(method_results)
            print(f"    {prior:>4s}: {count:3d} / {len(method_results)} ({pct:.1f}%)")

        # Average performance
        print(f"\n  Average {metric}:")
        for prior in ['auto', '0.5', '0.69']:
            avg = method_results[prior].mean()
            print(f"    {prior:>4s}: {avg:.4f}")

        # Pairwise comparisons
        print(f"\n  Pairwise improvement (mean Δ{metric}):")
        for prior_a, prior_b in [('0.69', '0.5'), ('auto', '0.5'), ('0.69', 'auto')]:
            diff = (method_results[prior_a] - method_results[prior_b]).mean()
            better = (method_results[prior_a] > method_results[prior_b]).sum()
            worse = (method_results[prior_a] < method_results[prior_b]).sum()
            equal = (method_results[prior_a] == method_results[prior_b]).sum()

            # Paired t-test
            t_stat, p_value = stats.ttest_rel(method_results[prior_a], method_results[prior_b])

            print(f"    {prior_a:>4s} vs {prior_b:>4s}: Δ={diff:+.4f}, "
                  f"{prior_a} better in {better}/{len(method_results)}, "
                  f"p={p_value:.4f}")

    return results_df


def plot_prior_comparison_by_condition(df, metric="test_auc"):
    """Plot method_prior performance conditioned on c and π separately."""
    mp_methods = df[df['base_method'].isin(['vpu_mean_prior', 'vpu_nomixup_mean_prior'])].copy()

    for base_method in ['vpu_mean_prior', 'vpu_nomixup_mean_prior']:
        method_data = mp_methods[mp_methods['base_method'] == base_method]

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Left plot: Performance vs π (averaged over c and datasets)
        ax = axes[0]
        for prior in ['auto', '0.5', '0.69']:
            prior_data = method_data[method_data['method_prior'] == prior]
            avg_by_pi = prior_data.groupby('pi')[metric].agg(['mean', 'std']).reset_index()

            ax.plot(avg_by_pi['pi'], avg_by_pi['mean'], marker='o', label=f'prior={prior}', linewidth=2)
            ax.fill_between(avg_by_pi['pi'],
                           avg_by_pi['mean'] - avg_by_pi['std'],
                           avg_by_pi['mean'] + avg_by_pi['std'],
                           alpha=0.2)

        ax.set_xlabel('True Prior (π)', fontsize=12)
        ax.set_ylabel(f'{metric.upper()}', fontsize=12)
        ax.set_title(f'{METHOD_LABELS[base_method]}: Performance vs True Prior', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Right plot: Performance vs c (averaged over π and datasets)
        ax = axes[1]
        for prior in ['auto', '0.5', '0.69']:
            prior_data = method_data[method_data['method_prior'] == prior]
            avg_by_c = prior_data.groupby('c')[metric].agg(['mean', 'std']).reset_index()

            ax.plot(avg_by_c['c'], avg_by_c['mean'], marker='o', label=f'prior={prior}', linewidth=2)
            ax.fill_between(avg_by_c['c'],
                           avg_by_c['mean'] - avg_by_c['std'],
                           avg_by_c['mean'] + avg_by_c['std'],
                           alpha=0.2)

        ax.set_xlabel('Label Frequency (c)', fontsize=12)
        ax.set_ylabel(f'{metric.upper()}', fontsize=12)
        ax.set_title(f'{METHOD_LABELS[base_method]}: Performance vs Label Frequency', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f'{base_method}_prior_comparison_curves_{metric}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Created prior comparison curves for {base_method}")


def plot_method_comparison_overall(df, metric="test_auc"):
    """Compare all methods overall (averaged across all conditions)."""
    # Average across all conditions
    avg_perf = df.groupby('method')[metric].agg(['mean', 'std', 'count']).reset_index()
    avg_perf = avg_perf.sort_values('mean', ascending=False)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color code by method type
    colors = []
    for method in avg_perf['method']:
        if 'oracle' in method:
            colors.append('green')
        elif 'pn_naive' in method:
            colors.append('red')
        elif 'mean_prior_auto' in method:
            colors.append('blue')
        elif 'mean_prior_0.5' in method:
            colors.append('orange')
        elif 'mean_prior_0.69' in method:
            colors.append('purple')
        else:
            colors.append('gray')

    bars = ax.barh(range(len(avg_perf)), avg_perf['mean'], xerr=avg_perf['std'],
                   color=colors, alpha=0.7, capsize=5)

    ax.set_yticks(range(len(avg_perf)))
    ax.set_yticklabels(avg_perf['method'])
    ax.set_xlabel(f'{metric.upper()}', fontsize=12)
    ax.set_title(f'Phase 3: Overall Method Performance (Seed 42)\nAveraged Across All Datasets and Conditions',
                fontsize=13)
    ax.grid(True, alpha=0.3, axis='x')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Oracle'),
        Patch(facecolor='gray', alpha=0.7, label='VPU Base'),
        Patch(facecolor='blue', alpha=0.7, label='VPU-MP(auto)'),
        Patch(facecolor='orange', alpha=0.7, label='VPU-MP(0.5)'),
        Patch(facecolor='purple', alpha=0.7, label='VPU-MP(0.69)'),
        Patch(facecolor='red', alpha=0.7, label='PN-Naive'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'overall_method_comparison_{metric}.png',
               dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Created overall method comparison plot")

    # Print ranking
    print(f"\n{'='*80}")
    print(f"OVERALL METHOD RANKING ({metric.upper()})")
    print(f"{'='*80}\n")
    for idx, row in avg_perf.iterrows():
        print(f"{row['method']:30s}: {row['mean']:.4f} ± {row['std']:.4f} (n={int(row['count'])})")


def analyze_mixup_effect(df, metric="test_auc"):
    """Compare VPU with vs without mixup for each method_prior variant."""
    print(f"\n{'='*80}")
    print(f"MIXUP EFFECT ANALYSIS ({metric.upper()})")
    print(f"{'='*80}\n")

    for prior in ['auto', '0.5', '0.69', None]:
        if prior is None:
            # Base methods
            with_mixup = df[df['method'] == 'vpu']
            without_mixup = df[df['method'] == 'vpu_nomixup']
            label = "Base VPU"
        else:
            with_mixup = df[df['method'] == f'vpu_mean_prior_{prior}']
            without_mixup = df[df['method'] == f'vpu_nomixup_mean_prior_{prior}']
            label = f"VPU-MP({prior})"

        if len(with_mixup) == 0 or len(without_mixup) == 0:
            continue

        # Merge on configuration
        merged = pd.merge(
            with_mixup[['dataset', 'c', 'pi', metric]],
            without_mixup[['dataset', 'c', 'pi', metric]],
            on=['dataset', 'c', 'pi'],
            suffixes=('_mixup', '_nomixup')
        )

        diff = merged[f'{metric}_mixup'] - merged[f'{metric}_nomixup']
        better_with = (diff > 0).sum()
        better_without = (diff < 0).sum()
        equal = (diff == 0).sum()

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(merged[f'{metric}_mixup'], merged[f'{metric}_nomixup'])

        print(f"{label}:")
        print(f"  With mixup:    {merged[f'{metric}_mixup'].mean():.4f} ± {merged[f'{metric}_mixup'].std():.4f}")
        print(f"  Without mixup: {merged[f'{metric}_nomixup'].mean():.4f} ± {merged[f'{metric}_nomixup'].std():.4f}")
        print(f"  Mean difference: {diff.mean():+.4f}")
        print(f"  Mixup better: {better_with}/{len(merged)}, Worse: {better_without}/{len(merged)}")
        print(f"  Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")
        print()


def main():
    """Run full Phase 3 analysis."""
    print("="*80)
    print("PHASE 3 ANALYSIS: Method Prior Comparison (Seed 42)")
    print("="*80)

    # Load data
    df = load_phase3_results()

    # Create visualizations
    print(f"\n{'='*80}")
    print("GENERATING PLOTS")
    print(f"{'='*80}\n")

    create_heatmap_comparison(df, metric="test_auc")
    plot_prior_comparison_by_condition(df, metric="test_auc")
    plot_method_comparison_overall(df, metric="test_auc")

    # Statistical analyses
    prior_results = analyze_prior_performance(df, metric="test_auc")
    analyze_mixup_effect(df, metric="test_auc")

    # Save results
    prior_results.to_csv(PLOTS_DIR / "prior_comparison_detailed.csv", index=False)
    print(f"\nDetailed results saved to: {PLOTS_DIR / 'prior_comparison_detailed.csv'}")
    print(f"Plots saved to: {PLOTS_DIR}/")

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
