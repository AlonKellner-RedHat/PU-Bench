"""Visualize auto vs 0.5 comparison results"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_phase1_results():
    """Load results from Phase 1 datasets only"""
    results = []
    results_dir = Path("results_comprehensive")
    json_files = list(results_dir.glob("seed_*/*.json"))
    phase1_datasets = ['MNIST', 'FashionMNIST', 'IMDB', '20News', 'Mushrooms', 'Spambase', 'Connect4']

    for json_file in json_files:
        try:
            data = json.load(open(json_file))
            for method_name, method_data in data.get('runs', {}).items():
                if method_name not in ['vpu_nomixup_mean_prior']:
                    continue

                hyperparams = method_data.get('hyperparameters', {})
                dataset_class = hyperparams.get('dataset_class', '')

                if dataset_class not in phase1_datasets:
                    continue

                method_prior = hyperparams.get('method_prior')
                if method_prior is None:
                    method_prior_label = 'auto'
                else:
                    method_prior_label = method_prior

                if method_prior_label not in ['auto', 0.5]:
                    continue

                dataset_info = method_data.get('dataset', {})
                metrics = method_data.get('best', {}).get('metrics', {})

                if not metrics:
                    continue

                results.append({
                    'dataset': dataset_class,
                    'seed': hyperparams.get('seed'),
                    'c': hyperparams.get('labeled_ratio'),
                    'true_prior': hyperparams.get('target_prevalence_train'),
                    'method_prior': method_prior_label,
                    'measured_prior': dataset_info.get('train', {}).get('prior'),
                    **{k: v for k, v in metrics.items() if k.startswith('test_')}
                })
        except Exception as e:
            continue

    return pd.DataFrame(results)


def plot_win_rate_by_true_prior(df, output_path):
    """Plot auto win rate by true prior"""
    metric = 'test_ap'

    # Create paired dataset
    paired = df.pivot_table(
        values=metric,
        index=['dataset', 'seed', 'c', 'true_prior'],
        columns='method_prior',
        aggfunc='first'
    ).dropna()
    paired['diff'] = paired['auto'] - paired[0.5]
    paired['winner'] = paired['diff'].apply(lambda x: 'auto' if x > 0 else '0.5')
    paired = paired.reset_index()

    # Calculate win rate and mean diff by true prior
    results = []
    for tp in sorted(paired['true_prior'].unique()):
        subset = paired[paired['true_prior'] == tp]
        win_rate = (subset['winner'] == 'auto').mean()
        mean_diff = subset['diff'].mean()
        results.append({
            'true_prior': tp,
            'win_rate': win_rate,
            'mean_diff': mean_diff,
            'n': len(subset)
        })

    results_df = pd.DataFrame(results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Win rate
    bars = ax1.bar(results_df['true_prior'], results_df['win_rate'],
                   color=['#2ecc71' if x > 0.5 else '#e74c3c' for x in results_df['win_rate']],
                   alpha=0.8, edgecolor='black', linewidth=1)
    ax1.axhline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Tie (50%)')
    ax1.set_xlabel('True Prior (π_true)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('auto Win Rate', fontsize=12, fontweight='bold')
    ax1.set_title('auto vs 0.5: Win Rate by True Prior', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Add percentage labels
    for i, (bar, row) in enumerate(zip(bars, results_df.itertuples())):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{row.win_rate*100:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Mean difference
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in results_df['mean_diff']]
    bars2 = ax2.bar(results_df['true_prior'], results_df['mean_diff'],
                    color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('True Prior (π_true)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean AP Difference (auto - 0.5)', fontsize=12, fontweight='bold')
    ax2.set_title('auto vs 0.5: Mean Performance Difference', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, row in zip(bars2, results_df.itertuples()):
        height = bar.get_height()
        va = 'bottom' if height > 0 else 'top'
        offset = 0.001 if height > 0 else -0.001
        ax2.text(bar.get_x() + bar.get_width()/2., height + offset,
                f'{row.mean_diff:+.4f}',
                ha='center', va=va, fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_win_rate_by_c(df, output_path):
    """Plot auto win rate by label frequency"""
    metric = 'test_ap'

    paired = df.pivot_table(
        values=metric,
        index=['dataset', 'seed', 'c', 'true_prior'],
        columns='method_prior',
        aggfunc='first'
    ).dropna()
    paired['diff'] = paired['auto'] - paired[0.5]
    paired['winner'] = paired['diff'].apply(lambda x: 'auto' if x > 0 else '0.5')
    paired = paired.reset_index()

    results = []
    for c in sorted(paired['c'].unique()):
        subset = paired[paired['c'] == c]
        win_rate = (subset['winner'] == 'auto').mean()
        mean_diff = subset['diff'].mean()
        results.append({
            'c': c,
            'win_rate': win_rate,
            'mean_diff': mean_diff
        })

    results_df = pd.DataFrame(results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Win rate
    bars = ax1.bar(range(len(results_df)), results_df['win_rate'],
                   color=['#2ecc71' if x > 0.5 else '#e74c3c' for x in results_df['win_rate']],
                   alpha=0.8, edgecolor='black', linewidth=1)
    ax1.axhline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xticks(range(len(results_df)))
    ax1.set_xticklabels([f'{c}' for c in results_df['c']])
    ax1.set_xlabel('Label Frequency (c)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('auto Win Rate', fontsize=12, fontweight='bold')
    ax1.set_title('auto vs 0.5: Win Rate by Label Frequency', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)

    for bar, row in zip(bars, results_df.itertuples()):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{row.win_rate*100:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Mean difference
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in results_df['mean_diff']]
    bars2 = ax2.bar(range(len(results_df)), results_df['mean_diff'],
                    color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xticks(range(len(results_df)))
    ax2.set_xticklabels([f'{c}' for c in results_df['c']])
    ax2.set_xlabel('Label Frequency (c)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean AP Difference (auto - 0.5)', fontsize=12, fontweight='bold')
    ax2.set_title('auto vs 0.5: Mean Performance Difference', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    for bar, row in zip(bars2, results_df.itertuples()):
        height = bar.get_height()
        va = 'bottom' if height > 0 else 'top'
        offset = 0.001 if height > 0 else -0.001
        ax2.text(bar.get_x() + bar.get_width()/2., height + offset,
                f'{row.mean_diff:+.4f}',
                ha='center', va=va, fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_interaction_heatmap(df, output_path):
    """Plot interaction heatmap: true_prior × c → auto win rate"""
    metric = 'test_ap'

    paired = df.pivot_table(
        values=metric,
        index=['dataset', 'seed', 'c', 'true_prior'],
        columns='method_prior',
        aggfunc='first'
    ).dropna()
    paired['diff'] = paired['auto'] - paired[0.5]
    paired['winner'] = paired['diff'].apply(lambda x: 'auto' if x > 0 else '0.5')
    paired = paired.reset_index()

    # Create pivot for heatmap
    heatmap_data = []
    for tp in sorted(paired['true_prior'].unique()):
        for c in sorted(paired['c'].unique()):
            subset = paired[(paired['true_prior'] == tp) & (paired['c'] == c)]
            if len(subset) > 0:
                win_rate = (subset['winner'] == 'auto').mean()
                mean_diff = subset['diff'].mean()
                heatmap_data.append({
                    'true_prior': tp,
                    'c': c,
                    'win_rate': win_rate,
                    'mean_diff': mean_diff,
                    'n': len(subset)
                })

    heatmap_df = pd.DataFrame(heatmap_data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Win rate heatmap
    pivot_wr = heatmap_df.pivot(index='c', columns='true_prior', values='win_rate')
    sns.heatmap(pivot_wr, annot=True, fmt='.1%', cmap='RdYlGn', center=0.5,
                vmin=0, vmax=1, cbar_kws={'label': 'auto Win Rate'},
                linewidths=1, linecolor='gray', ax=ax1)
    ax1.set_xlabel('True Prior (π_true)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Label Frequency (c)', fontsize=12, fontweight='bold')
    ax1.set_title('auto Win Rate: true_prior × c', fontsize=13, fontweight='bold')

    # Mean difference heatmap
    pivot_diff = heatmap_df.pivot(index='c', columns='true_prior', values='mean_diff')
    sns.heatmap(pivot_diff, annot=True, fmt='.4f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Mean AP Diff (auto - 0.5)'},
                linewidths=1, linecolor='gray', ax=ax2)
    ax2.set_xlabel('True Prior (π_true)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Label Frequency (c)', fontsize=12, fontweight='bold')
    ax2.set_title('Mean AP Difference: true_prior × c', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_by_dataset(df, output_path):
    """Plot comparison by dataset"""
    metric = 'test_ap'

    paired = df.pivot_table(
        values=metric,
        index=['dataset', 'seed', 'c', 'true_prior'],
        columns='method_prior',
        aggfunc='first'
    ).dropna()
    paired['diff'] = paired['auto'] - paired[0.5]
    paired['winner'] = paired['diff'].apply(lambda x: 'auto' if x > 0 else '0.5')
    paired = paired.reset_index()

    results = []
    for dataset in sorted(paired['dataset'].unique()):
        subset = paired[paired['dataset'] == dataset]
        win_rate = (subset['winner'] == 'auto').mean()
        mean_diff = subset['diff'].mean()
        results.append({
            'dataset': dataset,
            'win_rate': win_rate,
            'mean_diff': mean_diff
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('win_rate', ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Win rate
    bars = ax1.barh(range(len(results_df)), results_df['win_rate'],
                    color=['#2ecc71' if x > 0.5 else '#e74c3c' for x in results_df['win_rate']],
                    alpha=0.8, edgecolor='black', linewidth=1)
    ax1.axvline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_yticks(range(len(results_df)))
    ax1.set_yticklabels(results_df['dataset'])
    ax1.set_xlabel('auto Win Rate', fontsize=12, fontweight='bold')
    ax1.set_title('auto vs 0.5: Win Rate by Dataset', fontsize=13, fontweight='bold')
    ax1.set_xlim([0, 1])
    ax1.grid(axis='x', alpha=0.3)

    for i, (bar, row) in enumerate(zip(bars, results_df.itertuples())):
        width = bar.get_width()
        ax1.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                f'{row.win_rate*100:.1f}%',
                ha='left', va='center', fontsize=9, fontweight='bold')

    # Mean difference
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in results_df['mean_diff']]
    bars2 = ax2.barh(range(len(results_df)), results_df['mean_diff'],
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_yticks(range(len(results_df)))
    ax2.set_yticklabels(results_df['dataset'])
    ax2.set_xlabel('Mean AP Difference (auto - 0.5)', fontsize=12, fontweight='bold')
    ax2.set_title('auto vs 0.5: Mean Performance Difference', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    for bar, row in zip(bars2, results_df.itertuples()):
        width = bar.get_width()
        ha = 'left' if width > 0 else 'right'
        offset = 0.001 if width > 0 else -0.001
        ax2.text(width + offset, bar.get_y() + bar.get_height()/2.,
                f'{row.mean_diff:+.4f}',
                ha=ha, va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("Loading data...")
    df = load_phase1_results()
    print(f"Loaded {len(df)} results\n")

    output_dir = Path("analysis/figures")
    output_dir.mkdir(exist_ok=True)

    print("Creating visualizations...")
    plot_win_rate_by_true_prior(df, output_dir / "auto_vs_05_by_true_prior.png")
    plot_win_rate_by_c(df, output_dir / "auto_vs_05_by_c.png")
    plot_interaction_heatmap(df, output_dir / "auto_vs_05_interaction.png")
    plot_by_dataset(df, output_dir / "auto_vs_05_by_dataset.png")

    print("\nAll visualizations saved to analysis/figures/")
