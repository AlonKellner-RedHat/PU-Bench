#!/usr/bin/env python3
"""Find robust default prior considering multiple objectives:
- Performance: AP, AUC, Max F1
- Calibration: A-NICE, ECE (lower is better)
- Convergence: Epochs to best validation
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_all_metrics():
    """Load all robustness experiments with all metrics"""
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
                best_epoch = method_data.get('best', {}).get('epoch', None)

                dataset = hyperparams.get('dataset_class')
                seed = hyperparams.get('seed')
                c = hyperparams.get('labeled_ratio')
                true_prior = dataset_info.get('train', {}).get('prior')

                # Handle method_prior: vpu_nomixup_mean is equivalent to prior=1.0
                if method_name == 'vpu_nomixup_mean':
                    method_prior = 1.0
                else:
                    method_prior = hyperparams.get('method_prior')

                if all(v is not None for v in [dataset, seed, c, true_prior]):
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
                        # Performance metrics (higher is better)
                        'test_ap': best_metrics.get('test_ap'),
                        'test_auc': best_metrics.get('test_auc'),
                        'test_max_f1': best_metrics.get('test_max_f1'),
                        # Calibration metrics
                        'test_anice': best_metrics.get('test_anice'),  # Higher is better
                        'test_snice': best_metrics.get('test_snice'),  # Higher is better
                        'test_ece': best_metrics.get('test_ece'),      # Lower is better
                        'test_mce': best_metrics.get('test_mce'),      # Lower is better
                        'test_brier': best_metrics.get('test_brier'),  # Lower is better
                        # Convergence speed
                        'convergence_epoch': best_epoch,               # Lower is better
                    })

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

    return pd.DataFrame(results)


def normalize_metrics(df):
    """Normalize metrics to [0, 1] scale where higher is better"""

    # Make a copy for normalized values
    df_norm = df.copy()

    # Performance metrics: already higher is better, just scale to [0,1]
    for metric in ['test_ap', 'test_auc', 'test_max_f1', 'test_anice', 'test_snice']:
        if metric in df.columns:
            min_val = df[metric].min()
            max_val = df[metric].max()
            if max_val > min_val:
                df_norm[f'{metric}_norm'] = (df[metric] - min_val) / (max_val - min_val)
            else:
                df_norm[f'{metric}_norm'] = 1.0

    # Calibration error metrics: flip so higher is better
    for metric in ['test_ece', 'test_mce', 'test_brier']:
        if metric in df.columns:
            min_val = df[metric].min()
            max_val = df[metric].max()
            if max_val > min_val:
                df_norm[f'{metric}_norm'] = 1 - (df[metric] - min_val) / (max_val - min_val)
            else:
                df_norm[f'{metric}_norm'] = 1.0

    # Convergence speed: flip so higher (faster) is better
    if 'convergence_epoch' in df.columns:
        min_val = df['convergence_epoch'].min()
        max_val = df['convergence_epoch'].max()
        if max_val > min_val:
            df_norm['convergence_epoch_norm'] = 1 - (df['convergence_epoch'] - min_val) / (max_val - min_val)
        else:
            df_norm['convergence_epoch_norm'] = 1.0

    return df_norm


def compute_composite_scores(df_norm):
    """Compute composite scores for different objectives"""

    # Performance score: average of AP, AUC, Max F1
    df_norm['performance_score'] = df_norm[['test_ap_norm', 'test_auc_norm', 'test_max_f1_norm']].mean(axis=1)

    # Calibration score: average of ANICE, SNICE, ECE, MCE, Brier
    df_norm['calibration_score'] = df_norm[['test_anice_norm', 'test_snice_norm',
                                             'test_ece_norm', 'test_mce_norm',
                                             'test_brier_norm']].mean(axis=1)

    # Convergence score: already normalized
    df_norm['convergence_score'] = df_norm['convergence_epoch_norm']

    # Overall score: equal weight to all three objectives
    df_norm['overall_score'] = (df_norm['performance_score'] +
                                df_norm['calibration_score'] +
                                df_norm['convergence_score']) / 3

    return df_norm


def analyze_by_objective(df_norm, objective='overall_score'):
    """Analyze which method_prior is best for each objective"""

    results = []

    for c in sorted(df_norm['c'].unique()):
        df_c = df_norm[df_norm['c'] == c]

        # Sort method_prior values
        method_priors = df_c['method_prior'].unique()
        method_priors_sorted = ['auto'] + sorted([p for p in method_priors if p != 'auto'],
                                                  key=lambda x: float(x))

        for method_prior in method_priors_sorted:
            subset = df_c[df_c['method_prior'] == method_prior]

            if subset.empty:
                continue

            results.append({
                'c': c,
                'method_prior': method_prior,
                f'mean_{objective}': subset[objective].mean(),
                f'std_{objective}': subset[objective].std(),
                'count': len(subset),
            })

    return pd.DataFrame(results)


def find_best_by_objective(df_analysis, objective='overall_score'):
    """Find best method_prior for each c and objective"""

    results = []

    for c in sorted(df_analysis['c'].unique()):
        subset = df_analysis[df_analysis['c'] == c]

        # Find best
        best_idx = subset[f'mean_{objective}'].idxmax()
        best_row = subset.loc[best_idx]

        # Get auto for comparison
        auto_row = subset[subset['method_prior'] == 'auto']
        auto_score = auto_row[f'mean_{objective}'].iloc[0] if not auto_row.empty else np.nan

        results.append({
            'c': c,
            'objective': objective,
            'best_prior': best_row['method_prior'],
            'best_score': best_row[f'mean_{objective}'],
            'auto_score': auto_score,
            'improvement': best_row[f'mean_{objective}'] - auto_score if not np.isnan(auto_score) else np.nan,
        })

    return pd.DataFrame(results)


def plot_multiobjective_comparison(df_norm, output_dir='results_robustness'):
    """Plot comparison across objectives"""

    output_dir = Path(output_dir)

    objectives = {
        'performance_score': 'Performance (AP/AUC/MaxF1)',
        'calibration_score': 'Calibration (ANICE/ECE/etc)',
        'convergence_score': 'Convergence Speed',
        'overall_score': 'Overall (All Metrics)'
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, (score_name, title) in zip(axes, objectives.items()):
        # Aggregate by method_prior across all c values
        df_agg = df_norm.groupby('method_prior')[score_name].agg(['mean', 'std']).reset_index()

        # Sort by method_prior
        df_agg['sort_key'] = df_agg['method_prior'].apply(
            lambda x: -0.05 if x == 'auto' else float(x)
        )
        df_agg = df_agg.sort_values('sort_key')

        # Plot
        x_positions = range(len(df_agg))
        ax.bar(x_positions, df_agg['mean'], yerr=df_agg['std'],
               capsize=5, alpha=0.7, edgecolor='black', linewidth=1.5)

        # Highlight best
        best_idx = df_agg['mean'].idxmax()
        ax.bar(best_idx, df_agg.loc[best_idx, 'mean'],
               color='#2ca02c', alpha=0.9, edgecolor='black', linewidth=2)

        # Highlight auto
        auto_idx = df_agg[df_agg['method_prior'] == 'auto'].index[0]
        ax.bar(auto_idx, df_agg.loc[auto_idx, 'mean'],
               color='#d62728', alpha=0.7, edgecolor='black', linewidth=2)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(df_agg['method_prior'], rotation=45)
        ax.set_xlabel('Method Prior')
        ax.set_ylabel('Normalized Score (higher is better)')
        ax.set_title(title)
        ax.grid(True, axis='y', alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ca02c', edgecolor='black', label='Best'),
            Patch(facecolor='#d62728', edgecolor='black', label='Auto (true prior)'),
            Patch(facecolor='C0', edgecolor='black', alpha=0.7, label='Others')
        ]
        ax.legend(handles=legend_elements, loc='lower left')

    plt.tight_layout()
    output_path = output_dir / "multiobjective_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved multiobjective comparison to {output_path}")
    plt.close()


def plot_heatmap_by_objective(df_norm, objective='overall_score', output_dir='results_robustness'):
    """Heatmap of method_prior performance by c for a specific objective"""

    output_dir = Path(output_dir)

    # Aggregate
    pivot = df_norm.pivot_table(
        values=objective,
        index='c',
        columns='method_prior',
        aggfunc='mean'
    )

    # Reorder columns
    cols = ['auto'] + sorted([c for c in pivot.columns if c != 'auto'],
                            key=lambda x: float(x))
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(12, 5))

    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=pivot.min().min(), vmax=pivot.max().max(),
                cbar_kws={'label': 'Score (higher is better)'},
                linewidths=0.5, linecolor='gray', ax=ax)

    ax.set_xlabel('Method Prior Value')
    ax.set_ylabel('Label Frequency (c)')

    title_map = {
        'performance_score': 'Performance',
        'calibration_score': 'Calibration',
        'convergence_score': 'Convergence Speed',
        'overall_score': 'Overall (All Metrics)'
    }
    ax.set_title(f'{title_map.get(objective, objective)} by Method Prior')

    plt.tight_layout()
    output_path = output_dir / f"heatmap_{objective}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved heatmap to {output_path}")
    plt.close()


def analyze_raw_metrics_by_prior(df, output_dir='results_robustness'):
    """Show raw metric values for each method_prior"""

    metrics = {
        'test_ap': 'AP',
        'test_auc': 'AUC',
        'test_anice': 'A-NICE',
        'test_ece': 'ECE',
        'convergence_epoch': 'Epochs'
    }

    results = []

    for method_prior in ['auto', 0.5, 0.7, 0.9, 1.0]:
        subset = df[df['method_prior'] == method_prior]

        if subset.empty:
            continue

        row = {'method_prior': method_prior}

        for metric_col, metric_name in metrics.items():
            if metric_col in subset.columns:
                mean_val = subset[metric_col].mean()
                std_val = subset[metric_col].std()
                row[f'{metric_name}_mean'] = mean_val
                row[f'{metric_name}_std'] = std_val

        results.append(row)

    df_results = pd.DataFrame(results)

    # Save
    output_path = Path(output_dir) / "raw_metrics_by_prior.csv"
    df_results.to_csv(output_path, index=False)
    print(f"✓ Saved raw metrics to {output_path}")

    return df_results


def main():
    print("=" * 80)
    print("Multi-Objective Default Prior Analysis")
    print("Objectives: Performance, Calibration, Convergence Speed")
    print("=" * 80)
    print()

    # Load data
    print("Loading all metrics...")
    df = load_all_metrics()
    print(f"✓ Loaded {len(df)} experiments")
    print()

    # Check for missing values
    print("Checking data completeness...")
    for col in ['test_ap', 'test_anice', 'test_ece', 'convergence_epoch']:
        missing = df[col].isna().sum()
        print(f"  {col}: {missing} missing ({missing/len(df)*100:.1f}%)")
    print()

    # Normalize metrics
    print("Normalizing metrics...")
    df_norm = normalize_metrics(df)
    df_norm = compute_composite_scores(df_norm)
    print("✓ Computed composite scores")
    print()

    # Analyze each objective
    print("=" * 80)
    print("Best Method Prior by Objective and Label Frequency")
    print("=" * 80)
    print()

    objectives = ['performance_score', 'calibration_score', 'convergence_score', 'overall_score']

    for objective in objectives:
        print(f"\n### {objective.replace('_', ' ').title()} ###")
        df_best = find_best_by_objective(
            analyze_by_objective(df_norm, objective),
            objective
        )
        print(df_best.to_string(index=False))

    print()

    # Overall best across all c values
    print("=" * 80)
    print("Overall Best Default (Averaged Across All c)")
    print("=" * 80)
    print()

    for objective in objectives:
        df_obj = analyze_by_objective(df_norm, objective)

        # Average across all c values
        overall = df_obj.groupby('method_prior')[f'mean_{objective}'].mean()
        best_prior = overall.idxmax()
        best_score = overall.max()
        auto_score = overall.get('auto', np.nan)

        print(f"{objective.replace('_', ' ').title()}:")
        print(f"  Best: {best_prior} (score={best_score:.4f})")
        if not np.isnan(auto_score):
            print(f"  Auto: auto (score={auto_score:.4f}, diff={best_score-auto_score:+.4f})")
        print()

    # Raw metrics table
    print("=" * 80)
    print("Raw Metric Averages by Method Prior")
    print("=" * 80)
    print()

    df_raw = analyze_raw_metrics_by_prior(df)
    print(df_raw.to_string(index=False))
    print()

    # Create visualizations
    print("Creating visualizations...")
    output_dir = Path("results_robustness")

    plot_multiobjective_comparison(df_norm, output_dir)

    for objective in objectives:
        plot_heatmap_by_objective(df_norm, objective, output_dir)

    print()

    # Final recommendation
    print("=" * 80)
    print("FINAL RECOMMENDATION")
    print("=" * 80)
    print()

    # Find best for overall_score
    df_overall = analyze_by_objective(df_norm, 'overall_score')
    overall_avg = df_overall.groupby('method_prior')['mean_overall_score'].mean()
    best_overall = overall_avg.idxmax()

    print(f"Best default considering ALL objectives: {best_overall}")
    print()
    print("Ranking by overall score:")
    ranking = overall_avg.sort_values(ascending=False)
    for i, (prior, score) in enumerate(ranking.items(), 1):
        marker = "✓" if i <= 3 else "⚠️" if i <= 5 else "❌"
        print(f"  {i}. {marker} {prior}: {score:.4f}")
    print()

    print("Decision tree:")
    print("  1. If you have labeled positives → use auto (true prior)")
    print(f"  2. If no prior information → use {best_overall}")
    print("  3. If computational budget is tight → check convergence_score ranking")
    print("  4. If calibration is critical → check calibration_score ranking")

    # Save all results
    output_dir = Path("results_robustness")
    df_norm.to_csv(output_dir / "multiobjective_normalized_data.csv", index=False)
    print()
    print(f"✓ All results saved to {output_dir}/")


if __name__ == "__main__":
    main()
