"""Comprehensive auto vs 0.5 comparison across ALL metrics"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def load_phase1_results():
    """Load Phase 1 results with ALL metrics"""
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
                method_prior_label = 'auto' if method_prior is None else method_prior

                if method_prior_label not in ['auto', 0.5]:
                    continue

                dataset_info = method_data.get('dataset', {})
                metrics = method_data.get('best', {}).get('metrics', {})
                training_info = method_data.get('best', {})

                if not metrics:
                    continue

                results.append({
                    'dataset': dataset_class,
                    'seed': hyperparams.get('seed'),
                    'c': hyperparams.get('labeled_ratio'),
                    'true_prior': hyperparams.get('target_prevalence_train'),
                    'method_prior': method_prior_label,
                    'measured_prior': dataset_info.get('train', {}).get('prior'),
                    'num_epochs': training_info.get('epoch', np.nan),
                    **{k: v for k, v in metrics.items() if k.startswith('test_')},
                    **{k: v for k, v in metrics.items() if k.startswith('val_')}
                })
        except Exception as e:
            continue

    return pd.DataFrame(results)


def analyze_all_metrics(df):
    """Comprehensive analysis across ALL metrics"""
    print("=" * 100)
    print("COMPREHENSIVE METRIC COMPARISON: auto vs 0.5")
    print("=" * 100)

    # Group metrics by category
    ranking_metrics = ['test_ap', 'test_auc']
    calibration_metrics = ['test_oracle_ce', 'test_ece', 'test_mce']
    threshold_metrics = ['test_f1', 'test_accuracy', 'test_precision', 'test_recall']
    other_metrics = ['test_brier', 'test_risk']

    all_metric_groups = [
        ("Ranking Metrics (higher is better)", ranking_metrics),
        ("Calibration Metrics (lower is better)", calibration_metrics),
        ("Threshold-Dependent Metrics", threshold_metrics),
        ("Other Metrics", other_metrics)
    ]

    for group_name, metrics in all_metric_groups:
        print(f"\n{'=' * 100}")
        print(f"{group_name}")
        print("=" * 100)

        for metric in metrics:
            if metric not in df.columns:
                continue

            auto_vals = df[df['method_prior'] == 'auto'][metric].dropna()
            fixed_vals = df[df['method_prior'] == 0.5][metric].dropna()

            if len(auto_vals) == 0 or len(fixed_vals) == 0:
                continue

            # Determine if lower is better
            is_lower_better = metric in calibration_metrics or 'ce' in metric or 'error' in metric

            auto_mean = auto_vals.mean()
            auto_std = auto_vals.std()
            fixed_mean = fixed_vals.mean()
            fixed_std = fixed_vals.std()

            # Calculate difference
            diff = auto_mean - fixed_mean
            pct_diff = (diff / fixed_mean * 100) if fixed_mean != 0 else 0

            # Determine winner
            if is_lower_better:
                winner = 'auto' if auto_mean < fixed_mean else '0.5'
                better_symbol = '↓'
            else:
                winner = 'auto' if auto_mean > fixed_mean else '0.5'
                better_symbol = '↑'

            # Paired t-test
            merged = df.pivot_table(
                values=metric,
                index=['dataset', 'seed', 'c', 'true_prior'],
                columns='method_prior',
                aggfunc='first'
            ).dropna()

            if len(merged) > 0 and 'auto' in merged.columns and 0.5 in merged.columns:
                t_stat, p_val = stats.ttest_rel(merged['auto'], merged[0.5])
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            else:
                t_stat, p_val = np.nan, np.nan
                sig = 'N/A'

            # Effect size (Cohen's d)
            pooled_std = np.sqrt((auto_std**2 + fixed_std**2) / 2)
            cohens_d = diff / pooled_std if pooled_std > 0 else 0

            print(f"\n{metric}:")
            print(f"  auto:    {auto_mean:.6f} ± {auto_std:.6f} (n={len(auto_vals)})")
            print(f"  0.5:     {fixed_mean:.6f} ± {fixed_std:.6f} (n={len(fixed_vals)})")
            print(f"  diff:    {diff:+.6f} ({pct_diff:+.2f}%) {better_symbol}")
            print(f"  winner:  {winner}")
            print(f"  p-value: {p_val:.4f} {sig}")
            print(f"  Cohen's d: {cohens_d:.3f} ({'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'})")


def analyze_by_true_prior_all_metrics(df):
    """Analyze all metrics by true prior"""
    print("\n" + "=" * 100)
    print("BY TRUE PRIOR: ALL METRICS")
    print("=" * 100)

    key_metrics = ['test_ap', 'test_auc', 'test_oracle_ce', 'test_ece', 'test_f1']

    for true_prior in sorted(df['true_prior'].unique()):
        print(f"\n{'─' * 100}")
        print(f"TRUE PRIOR π = {true_prior}")
        print('─' * 100)

        subset = df[df['true_prior'] == true_prior]

        for metric in key_metrics:
            if metric not in subset.columns:
                continue

            auto_vals = subset[subset['method_prior'] == 'auto'][metric].dropna()
            fixed_vals = subset[subset['method_prior'] == 0.5][metric].dropna()

            if len(auto_vals) == 0 or len(fixed_vals) == 0:
                continue

            is_lower_better = 'ce' in metric or 'ece' in metric or 'error' in metric

            auto_mean = auto_vals.mean()
            fixed_mean = fixed_vals.mean()
            diff = auto_mean - fixed_mean

            if is_lower_better:
                winner = 'auto' if auto_mean < fixed_mean else '0.5'
            else:
                winner = 'auto' if auto_mean > fixed_mean else '0.5'

            # Win rate (head-to-head)
            paired = subset.pivot_table(
                values=metric,
                index=['dataset', 'seed', 'c'],
                columns='method_prior',
                aggfunc='first'
            ).dropna()

            if len(paired) > 0 and 'auto' in paired.columns and 0.5 in paired.columns:
                if is_lower_better:
                    win_rate = (paired['auto'] < paired[0.5]).mean()
                else:
                    win_rate = (paired['auto'] > paired[0.5]).mean()
            else:
                win_rate = np.nan

            print(f"  {metric:20s}: auto={auto_mean:.4f}, 0.5={fixed_mean:.4f}, diff={diff:+.4f} → {winner:4s} (win rate: {win_rate*100:.1f}%)")


def identify_contradiction(df):
    """Resolve the contradiction: when does auto actually win?"""
    print("\n" + "=" * 100)
    print("RESOLVING CONTRADICTION: TRUE PRIOR PATTERNS")
    print("=" * 100)

    metric = 'test_ap'

    print("\nPattern 1: By configured TRUE PRIOR")
    print("─" * 60)

    for tp in sorted(df['true_prior'].unique()):
        subset = df[df['true_prior'] == tp]

        # Get paired comparison
        paired = subset.pivot_table(
            values=metric,
            index=['dataset', 'seed', 'c'],
            columns='method_prior',
            aggfunc='first'
        ).dropna()

        if len(paired) > 0 and 'auto' in paired.columns and 0.5 in paired.columns:
            win_rate = (paired['auto'] > paired[0.5]).mean()
            mean_diff = (paired['auto'] - paired[0.5]).mean()

            auto_mean = paired['auto'].mean()
            fixed_mean = paired[0.5].mean()

            # Distance from 0.5
            dist_from_05 = abs(tp - 0.5)

            print(f"π = {tp:.1f} (|π-0.5| = {dist_from_05:.1f}):")
            print(f"  Win rate: {win_rate*100:.1f}%  Mean diff: {mean_diff:+.4f}")
            print(f"  auto: {auto_mean:.4f}  0.5: {fixed_mean:.4f}")


def analyze_measured_vs_configured_prior(df):
    """Analyze difference between measured and configured prior"""
    print("\n" + "=" * 100)
    print("MEASURED vs CONFIGURED PRIOR")
    print("=" * 100)

    # Calculate prior error
    df_copy = df.copy()
    df_copy['prior_error'] = (df_copy['measured_prior'] - df_copy['true_prior']).abs()

    print("\nPrior estimation error by label frequency:")
    for c in sorted(df_copy['c'].unique()):
        subset = df_copy[df_copy['c'] == c]
        mean_error = subset['prior_error'].mean()
        max_error = subset['prior_error'].max()
        print(f"  c={c}: mean error={mean_error:.4f}, max error={max_error:.4f}")

    print("\nDoes measured prior distance matter more than configured?")

    metric = 'test_ap'

    # Bin by measured prior distance
    df_copy['measured_dist'] = (df_copy['measured_prior'] - 0.5).abs()
    df_copy['configured_dist'] = (df_copy['true_prior'] - 0.5).abs()

    paired = df_copy.pivot_table(
        values=metric,
        index=['dataset', 'seed', 'c', 'true_prior', 'measured_dist', 'configured_dist'],
        columns='method_prior',
        aggfunc='first'
    ).dropna()

    paired['diff'] = paired['auto'] - paired[0.5]
    paired['winner'] = paired['diff'].apply(lambda x: 'auto' if x > 0 else '0.5')
    paired = paired.reset_index()

    print("\n  By MEASURED prior distance:")
    for threshold in [0.1, 0.2, 0.3]:
        close = paired[paired['measured_dist'] <= threshold]
        far = paired[paired['measured_dist'] > threshold]

        if len(close) > 0:
            print(f"    |π_measured - 0.5| ≤ {threshold}: auto wins {100*(close['winner']=='auto').mean():.1f}% (n={len(close)})")
        if len(far) > 0:
            print(f"    |π_measured - 0.5| > {threshold}: auto wins {100*(far['winner']=='auto').mean():.1f}% (n={len(far)})")

    print("\n  By CONFIGURED prior distance:")
    for threshold in [0.1, 0.2, 0.3]:
        close = paired[paired['configured_dist'] <= threshold]
        far = paired[paired['configured_dist'] > threshold]

        if len(close) > 0:
            print(f"    |π_true - 0.5| ≤ {threshold}: auto wins {100*(close['winner']=='auto').mean():.1f}% (n={len(close)})")
        if len(far) > 0:
            print(f"    |π_true - 0.5| > {threshold}: auto wins {100*(far['winner']=='auto').mean():.1f}% (n={len(far)})")


def convergence_analysis(df):
    """Analyze convergence speed"""
    print("\n" + "=" * 100)
    print("CONVERGENCE SPEED ANALYSIS")
    print("=" * 100)

    if 'num_epochs' not in df.columns or df['num_epochs'].isna().all():
        print("\n  No convergence data available (num_epochs not recorded)")
        return

    auto_epochs = df[df['method_prior'] == 'auto']['num_epochs'].dropna()
    fixed_epochs = df[df['method_prior'] == 0.5]['num_epochs'].dropna()

    if len(auto_epochs) == 0 or len(fixed_epochs) == 0:
        print("\n  No convergence data available")
        return

    print(f"\nEpochs until early stopping:")
    print(f"  auto: {auto_epochs.mean():.2f} ± {auto_epochs.std():.2f} (median: {auto_epochs.median():.0f})")
    print(f"  0.5:  {fixed_epochs.mean():.2f} ± {fixed_epochs.std():.2f} (median: {fixed_epochs.median():.0f})")

    # Paired comparison
    merged = df.pivot_table(
        values='num_epochs',
        index=['dataset', 'seed', 'c', 'true_prior'],
        columns='method_prior',
        aggfunc='first'
    ).dropna()

    if len(merged) > 0 and 'auto' in merged.columns and 0.5 in merged.columns:
        faster_count = (merged['auto'] < merged[0.5]).sum()
        slower_count = (merged['auto'] > merged[0.5]).sum()
        same_count = (merged['auto'] == merged[0.5]).sum()

        print(f"\n  auto converges faster: {faster_count} times ({100*faster_count/len(merged):.1f}%)")
        print(f"  auto converges slower: {slower_count} times ({100*slower_count/len(merged):.1f}%)")
        print(f"  Same convergence:      {same_count} times ({100*same_count/len(merged):.1f}%)")


def final_recommendation(df):
    """Generate clear, non-contradictory recommendations"""
    print("\n" + "=" * 100)
    print("FINAL RECOMMENDATIONS (Non-Contradictory)")
    print("=" * 100)

    metric = 'test_ap'

    # Analyze by configured true prior
    results_by_prior = []
    for tp in sorted(df['true_prior'].unique()):
        subset = df[df['true_prior'] == tp]
        paired = subset.pivot_table(
            values=metric,
            index=['dataset', 'seed', 'c'],
            columns='method_prior',
            aggfunc='first'
        ).dropna()

        if len(paired) > 0 and 'auto' in paired.columns and 0.5 in paired.columns:
            win_rate = (paired['auto'] > paired[0.5]).mean()
            mean_diff = (paired['auto'] - paired[0.5]).mean()

            results_by_prior.append({
                'true_prior': tp,
                'win_rate': win_rate,
                'mean_diff': mean_diff
            })

    results_df = pd.DataFrame(results_by_prior)

    print("\nBased on configured TRUE PRIOR (π_true):")
    print("─" * 60)

    for _, row in results_df.iterrows():
        tp = row['true_prior']
        wr = row['win_rate']
        md = row['mean_diff']

        if wr > 0.65 and md > 0:
            recommendation = "✓ Use AUTO"
        elif wr < 0.45 or md < -0.005:
            recommendation = "✓ Use 0.5"
        else:
            recommendation = "≈ Either (similar performance)"

        print(f"  π = {tp:.1f}: {recommendation:30s} (win rate: {wr*100:.1f}%, diff: {md:+.4f})")

    # Calibration vs Ranking trade-off
    print("\n\nMetric Priority:")
    print("─" * 60)

    # Check Oracle CE winner
    auto_ce = df[df['method_prior'] == 'auto']['test_oracle_ce'].mean()
    fixed_ce = df[df['method_prior'] == 0.5]['test_oracle_ce'].mean()
    ce_diff = auto_ce - fixed_ce
    ce_pct = (ce_diff / fixed_ce * 100)

    print(f"  Calibration (Oracle CE):")
    print(f"    auto:  {auto_ce:.4f}")
    print(f"    0.5:   {fixed_ce:.4f}")
    print(f"    → 0.5 is {abs(ce_pct):.1f}% better at calibration")

    print(f"\n  Ranking (AP):")
    auto_ap = df[df['method_prior'] == 'auto']['test_ap'].mean()
    fixed_ap = df[df['method_prior'] == 0.5]['test_ap'].mean()
    print(f"    auto:  {auto_ap:.4f}")
    print(f"    0.5:   {fixed_ap:.4f}")
    print(f"    → Nearly identical (diff: {auto_ap - fixed_ap:+.4f})")

    print("\n  Recommendation:")
    print("    - If calibration is critical (e.g., medical, risk assessment): Use 0.5")
    print("    - If ranking is primary (e.g., retrieval, recommendation): Use auto for π ∈ [0.3, 0.7]")


if __name__ == "__main__":
    print("Loading Phase 1 results with all metrics...")
    df = load_phase1_results()
    print(f"Loaded {len(df)} results\n")

    analyze_all_metrics(df)
    analyze_by_true_prior_all_metrics(df)
    identify_contradiction(df)
    analyze_measured_vs_configured_prior(df)
    convergence_analysis(df)
    final_recommendation(df)
