"""Compare vpu_nomixup_mean_prior with method_prior='auto' vs method_prior=0.5

This analysis focuses on Phase 1 datasets only (excluding CIFAR10 and AlzheimerMRI).
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats


def load_phase1_results():
    """Load results from Phase 1 datasets only (exclude CIFAR10 and AlzheimerMRI)"""
    results = []

    results_dir = Path("results_comprehensive")
    json_files = list(results_dir.glob("seed_*/*.json"))

    phase1_datasets = ['MNIST', 'FashionMNIST', 'IMDB', '20News', 'Mushrooms', 'Spambase', 'Connect4']

    for json_file in json_files:
        try:
            data = json.load(open(json_file))

            # Extract method results
            for method_name, method_data in data.get('runs', {}).items():
                if method_name not in ['vpu_nomixup_mean_prior']:
                    continue

                hyperparams = method_data.get('hyperparameters', {})
                dataset_class = hyperparams.get('dataset_class', '')

                # Skip Phase 2 datasets
                if dataset_class not in phase1_datasets:
                    continue

                method_prior = hyperparams.get('method_prior')

                # Map None/null to 'auto'
                if method_prior is None:
                    method_prior_label = 'auto'
                else:
                    method_prior_label = method_prior

                # Only keep auto and 0.5
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
            print(f"Error loading {json_file}: {e}")
            continue

    return pd.DataFrame(results)


def overall_comparison(df):
    """Overall comparison across all experiments"""
    print("=" * 80)
    print("OVERALL COMPARISON: auto vs 0.5")
    print("=" * 80)

    metrics = ['test_ap', 'test_auc', 'test_f1', 'test_accuracy', 'test_oracle_ce']

    for metric in metrics:
        auto_vals = df[df['method_prior'] == 'auto'][metric].dropna()
        fixed_vals = df[df['method_prior'] == 0.5][metric].dropna()

        # For oracle_ce, lower is better; for others, higher is better
        is_lower_better = (metric == 'test_oracle_ce')

        auto_mean = auto_vals.mean()
        fixed_mean = fixed_vals.mean()

        if is_lower_better:
            winner = 'auto' if auto_mean < fixed_mean else '0.5'
            diff = fixed_mean - auto_mean  # positive means auto is better
        else:
            winner = 'auto' if auto_mean > fixed_mean else '0.5'
            diff = auto_mean - fixed_mean  # positive means auto is better

        # Paired t-test (match by dataset, seed, c, true_prior)
        merged = df.pivot_table(
            values=metric,
            index=['dataset', 'seed', 'c', 'true_prior'],
            columns='method_prior',
            aggfunc='first'
        ).dropna()

        if len(merged) > 0 and 'auto' in merged.columns and 0.5 in merged.columns:
            t_stat, p_val = stats.ttest_rel(merged['auto'], merged[0.5])
        else:
            t_stat, p_val = np.nan, np.nan

        print(f"\n{metric}:")
        print(f"  auto:  {auto_mean:.4f} (n={len(auto_vals)})")
        print(f"  0.5:   {fixed_mean:.4f} (n={len(fixed_vals)})")
        print(f"  diff:  {diff:+.4f} ({'↓ better' if is_lower_better else '↑ better'})")
        print(f"  winner: {winner}")
        print(f"  paired t-test: t={t_stat:.3f}, p={p_val:.4f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}")


def by_dataset_comparison(df):
    """Compare by dataset"""
    print("\n" + "=" * 80)
    print("BY DATASET COMPARISON")
    print("=" * 80)

    metric = 'test_ap'  # Use AP as primary metric

    for dataset in sorted(df['dataset'].unique()):
        subset = df[df['dataset'] == dataset]

        auto_vals = subset[subset['method_prior'] == 'auto'][metric].dropna()
        fixed_vals = subset[subset['method_prior'] == 0.5][metric].dropna()

        if len(auto_vals) == 0 or len(fixed_vals) == 0:
            continue

        auto_mean = auto_vals.mean()
        fixed_mean = fixed_vals.mean()
        diff = auto_mean - fixed_mean
        winner = 'auto' if diff > 0 else '0.5'

        # Paired t-test
        merged = subset.pivot_table(
            values=metric,
            index=['seed', 'c', 'true_prior'],
            columns='method_prior',
            aggfunc='first'
        ).dropna()

        if len(merged) > 1:
            t_stat, p_val = stats.ttest_rel(merged['auto'], merged[0.5])
        else:
            t_stat, p_val = np.nan, np.nan

        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'

        print(f"\n{dataset}:")
        print(f"  auto: {auto_mean:.4f} ± {auto_vals.std():.4f}")
        print(f"  0.5:  {fixed_mean:.4f} ± {fixed_vals.std():.4f}")
        print(f"  diff: {diff:+.4f} → {winner} ({sig})")


def by_true_prior_comparison(df):
    """Compare by true prior value"""
    print("\n" + "=" * 80)
    print("BY TRUE PRIOR COMPARISON")
    print("=" * 80)

    metric = 'test_ap'

    for true_prior in sorted(df['true_prior'].unique()):
        subset = df[df['true_prior'] == true_prior]

        auto_vals = subset[subset['method_prior'] == 'auto'][metric].dropna()
        fixed_vals = subset[subset['method_prior'] == 0.5][metric].dropna()

        if len(auto_vals) == 0 or len(fixed_vals) == 0:
            continue

        auto_mean = auto_vals.mean()
        fixed_mean = fixed_vals.mean()
        diff = auto_mean - fixed_mean
        winner = 'auto' if diff > 0 else '0.5'

        # Count wins per dataset
        dataset_wins = {}
        for dataset in subset['dataset'].unique():
            ds_subset = subset[subset['dataset'] == dataset]
            auto_ap = ds_subset[ds_subset['method_prior'] == 'auto'][metric].mean()
            fixed_ap = ds_subset[ds_subset['method_prior'] == 0.5][metric].mean()
            if not pd.isna(auto_ap) and not pd.isna(fixed_ap):
                dataset_wins[dataset] = 'auto' if auto_ap > fixed_ap else '0.5'

        auto_win_count = sum(1 for w in dataset_wins.values() if w == 'auto')
        fixed_win_count = len(dataset_wins) - auto_win_count

        print(f"\ntrue_prior={true_prior}:")
        print(f"  auto: {auto_mean:.4f}")
        print(f"  0.5:  {fixed_mean:.4f}")
        print(f"  diff: {diff:+.4f} → {winner}")
        print(f"  dataset wins: auto={auto_win_count}/{len(dataset_wins)}, 0.5={fixed_win_count}/{len(dataset_wins)}")


def by_c_value_comparison(df):
    """Compare by label frequency (c)"""
    print("\n" + "=" * 80)
    print("BY LABEL FREQUENCY (c) COMPARISON")
    print("=" * 80)

    metric = 'test_ap'

    for c in sorted(df['c'].unique()):
        subset = df[df['c'] == c]

        auto_vals = subset[subset['method_prior'] == 'auto'][metric].dropna()
        fixed_vals = subset[subset['method_prior'] == 0.5][metric].dropna()

        if len(auto_vals) == 0 or len(fixed_vals) == 0:
            continue

        auto_mean = auto_vals.mean()
        fixed_mean = fixed_vals.mean()
        diff = auto_mean - fixed_mean
        winner = 'auto' if diff > 0 else '0.5'

        # Count wins per dataset
        dataset_wins = {}
        for dataset in subset['dataset'].unique():
            ds_subset = subset[subset['dataset'] == dataset]
            auto_ap = ds_subset[ds_subset['method_prior'] == 'auto'][metric].mean()
            fixed_ap = ds_subset[ds_subset['method_prior'] == 0.5][metric].mean()
            if not pd.isna(auto_ap) and not pd.isna(fixed_ap):
                dataset_wins[dataset] = 'auto' if auto_ap > fixed_ap else '0.5'

        auto_win_count = sum(1 for w in dataset_wins.values() if w == 'auto')
        fixed_win_count = len(dataset_wins) - auto_win_count

        print(f"\nc={c}:")
        print(f"  auto: {auto_mean:.4f}")
        print(f"  0.5:  {fixed_mean:.4f}")
        print(f"  diff: {diff:+.4f} → {winner}")
        print(f"  dataset wins: auto={auto_win_count}/{len(dataset_wins)}, 0.5={fixed_win_count}/{len(dataset_wins)}")


def when_auto_wins_vs_loses(df):
    """Analyze conditions where auto wins vs loses"""
    print("\n" + "=" * 80)
    print("WHEN DOES AUTO WIN vs LOSE?")
    print("=" * 80)

    metric = 'test_ap'

    # Create paired dataset
    paired = df.pivot_table(
        values=metric,
        index=['dataset', 'seed', 'c', 'true_prior', 'measured_prior'],
        columns='method_prior',
        aggfunc='first'
    ).dropna()

    paired['diff'] = paired['auto'] - paired[0.5]
    paired['winner'] = paired['diff'].apply(lambda x: 'auto' if x > 0 else '0.5')
    paired = paired.reset_index()

    print(f"\nOverall wins:")
    print(f"  auto wins: {(paired['winner'] == 'auto').sum()} ({100 * (paired['winner'] == 'auto').mean():.1f}%)")
    print(f"  0.5 wins:  {(paired['winner'] == '0.5').sum()} ({100 * (paired['winner'] == '0.5').mean():.1f}%)")

    # Analyze by true prior
    print(f"\nBy true prior:")
    for tp in sorted(paired['true_prior'].unique()):
        subset = paired[paired['true_prior'] == tp]
        auto_win_rate = (subset['winner'] == 'auto').mean()
        mean_diff = subset['diff'].mean()
        print(f"  π_true={tp}: auto wins {100*auto_win_rate:.1f}% (mean diff={mean_diff:+.4f})")

    # Analyze by measured prior vs 0.5 distance
    print(f"\nBy prior distance from 0.5:")
    paired['prior_dist'] = (paired['measured_prior'] - 0.5).abs()
    for threshold in [0.1, 0.2, 0.3]:
        close = paired[paired['prior_dist'] <= threshold]
        far = paired[paired['prior_dist'] > threshold]

        print(f"  |π_measured - 0.5| <= {threshold}:")
        if len(close) > 0:
            print(f"    auto wins {100 * (close['winner'] == 'auto').mean():.1f}% (n={len(close)})")

        print(f"  |π_measured - 0.5| > {threshold}:")
        if len(far) > 0:
            print(f"    auto wins {100 * (far['winner'] == 'auto').mean():.1f}% (n={len(far)})")

    # Analyze by dataset
    print(f"\nBy dataset:")
    for dataset in sorted(paired['dataset'].unique()):
        subset = paired[paired['dataset'] == dataset]
        auto_win_rate = (subset['winner'] == 'auto').mean()
        mean_diff = subset['diff'].mean()
        print(f"  {dataset}: auto wins {100*auto_win_rate:.1f}% (mean diff={mean_diff:+.4f})")

    # Analyze by c value
    print(f"\nBy label frequency:")
    for c in sorted(paired['c'].unique()):
        subset = paired[paired['c'] == c]
        auto_win_rate = (subset['winner'] == 'auto').mean()
        mean_diff = subset['diff'].mean()
        print(f"  c={c}: auto wins {100*auto_win_rate:.1f}% (mean diff={mean_diff:+.4f})")

    return paired


def interaction_analysis(paired_df):
    """Analyze interactions between factors"""
    print("\n" + "=" * 80)
    print("INTERACTION ANALYSIS: Prior Distance × Label Frequency")
    print("=" * 80)

    paired_df['prior_dist_bin'] = pd.cut(
        paired_df['prior_dist'],
        bins=[0, 0.1, 0.2, 0.5],
        labels=['close (≤0.1)', 'medium (0.1-0.2)', 'far (>0.2)']
    )

    for c in sorted(paired_df['c'].unique()):
        print(f"\nc={c}:")
        subset = paired_df[paired_df['c'] == c]

        for bin_label in ['close (≤0.1)', 'medium (0.1-0.2)', 'far (>0.2)']:
            bin_subset = subset[subset['prior_dist_bin'] == bin_label]

            if len(bin_subset) == 0:
                continue

            auto_win_rate = (bin_subset['winner'] == 'auto').mean()
            mean_diff = bin_subset['diff'].mean()

            print(f"  {bin_label}: auto wins {100*auto_win_rate:.1f}% (diff={mean_diff:+.4f}, n={len(bin_subset)})")


if __name__ == "__main__":
    print("Loading Phase 1 results...")
    df = load_phase1_results()

    print(f"Loaded {len(df)} results")
    print(f"  auto: {len(df[df['method_prior'] == 'auto'])}")
    print(f"  0.5:  {len(df[df['method_prior'] == 0.5])}")
    print(f"  Datasets: {sorted(df['dataset'].unique())}")
    print(f"  True priors: {sorted(df['true_prior'].unique())}")
    print(f"  C values: {sorted(df['c'].unique())}")
    print()

    overall_comparison(df)
    by_dataset_comparison(df)
    by_true_prior_comparison(df)
    by_c_value_comparison(df)
    paired = when_auto_wins_vs_loses(df)
    interaction_analysis(paired)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nKey insights will be generated based on the analysis above.")
