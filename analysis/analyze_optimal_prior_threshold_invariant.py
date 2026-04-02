#!/usr/bin/env python3
"""Re-analyze optimal prior using threshold-invariant metrics

Uses Average Precision (AP) instead of F1 for determining optimal prior,
since F1 depends on threshold choice and is not appropriate for method comparison.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np


def load_robustness_results(metric='test_ap'):
    """Load robustness results and extract specified metric

    Args:
        metric: One of 'test_ap', 'test_auc', 'test_max_f1' (threshold-invariant)
    """
    results = []

    results_dir = Path("results_robustness")
    json_files = list(results_dir.glob("seed_*/*.json"))

    print(f"Loading {len(json_files)} result files...")

    for json_file in json_files:
        try:
            with open(json_file) as f:
                data = json.load(f)

            # Process each method in the run
            for method_name, method_data in data.get('runs', {}).items():
                if method_name != 'vpu_nomixup_mean_prior':
                    continue  # Only analyze the prior-based method

                hyperparams = method_data.get('hyperparameters', {})
                dataset_info = method_data.get('dataset', {})
                best_metrics = method_data.get('best', {}).get('metrics', {})

                # Extract key information
                dataset = hyperparams.get('dataset_class')
                seed = hyperparams.get('seed')
                c = hyperparams.get('labeled_ratio')
                method_prior = hyperparams.get('method_prior')
                true_prior = dataset_info.get('train', {}).get('prior')

                # Get the threshold-invariant metric
                metric_value = best_metrics.get(metric)

                if all(v is not None for v in [dataset, seed, c, true_prior, metric_value]):
                    results.append({
                        'dataset': dataset,
                        'seed': seed,
                        'c': c,
                        'method_prior': method_prior if method_prior is not None else 'auto',
                        'true_prior': true_prior,
                        metric: metric_value,
                        # Also store other metrics for reference
                        'test_auc': best_metrics.get('test_auc'),
                        'test_max_f1': best_metrics.get('test_max_f1'),
                        'test_f1': best_metrics.get('test_f1'),
                    })

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

    df = pd.DataFrame(results)
    print(f"Loaded {len(df)} experiments")
    return df


def find_optimal_prior_per_config(df, metric='test_ap'):
    """Find optimal prior for each (dataset, seed, c) configuration"""

    results = []

    for (dataset, seed, c), group in df.groupby(['dataset', 'seed', 'c']):
        # Get true prior (should be same across all rows in group)
        true_prior = group['true_prior'].iloc[0]

        # Find the prior value that gives best metric
        best_idx = group[metric].idxmax()
        best_row = group.loc[best_idx]

        optimal_prior = best_row['method_prior']
        best_metric_value = best_row[metric]

        # Get baseline (auto prior) performance
        auto_rows = group[group['method_prior'] == 'auto']
        if not auto_rows.empty:
            baseline_metric = auto_rows[metric].iloc[0]
        else:
            baseline_metric = np.nan

        # Calculate difference
        if optimal_prior == 'auto':
            prior_diff = 0.0
            direction = 'auto'
            optimal_prior_value = true_prior
        else:
            optimal_prior_value = float(optimal_prior)
            prior_diff = optimal_prior_value - true_prior
            direction = 'higher' if prior_diff > 0 else ('lower' if prior_diff < 0 else 'exact')

        results.append({
            'dataset': dataset,
            'seed': seed,
            'c': c,
            'true_prior': true_prior,
            'optimal_prior': optimal_prior_value if optimal_prior != 'auto' else 'auto',
            'prior_diff': prior_diff,
            'direction': direction,
            f'best_{metric}': best_metric_value,
            f'baseline_{metric}': baseline_metric,
            'improvement': best_metric_value - baseline_metric if not np.isnan(baseline_metric) else 0,
            # Also include other metrics at optimal point
            'best_auc': best_row['test_auc'],
            'best_max_f1': best_row['test_max_f1'],
            'best_f1': best_row['test_f1'],
        })

    return pd.DataFrame(results)


def analyze_direction_by_c(df):
    """Analyze optimal direction stratified by label frequency"""

    summary = []

    for c in sorted(df['c'].unique()):
        subset = df[df['c'] == c]

        total = len(subset)
        higher = len(subset[subset['direction'] == 'higher'])
        lower = len(subset[subset['direction'] == 'lower'])
        auto = len(subset[subset['direction'] == 'auto'])

        # Among those that prefer higher/lower, compute mean difference
        higher_diff = subset[subset['direction'] == 'higher']['prior_diff'].mean()
        lower_diff = subset[subset['direction'] == 'lower']['prior_diff'].mean()
        overall_diff = subset[subset['direction'] != 'auto']['prior_diff'].mean()

        summary.append({
            'c': c,
            'total': total,
            'prefer_higher': higher,
            'prefer_lower': lower,
            'prefer_auto': auto,
            'pct_higher': f'{higher/total*100:.1f}%',
            'pct_lower': f'{lower/total*100:.1f}%',
            'pct_auto': f'{auto/total*100:.1f}%',
            'mean_diff_when_higher': higher_diff,
            'mean_diff_when_lower': lower_diff,
            'mean_diff_overall': overall_diff,
        })

    return pd.DataFrame(summary)


def compare_metrics(df_f1, df_ap, df_auc, df_max_f1):
    """Compare optimal prior selection across different metrics"""

    comparison = []

    for idx, row in df_f1.iterrows():
        key = (row['dataset'], row['seed'], row['c'])

        # Find corresponding rows in other dataframes
        ap_row = df_ap[(df_ap['dataset'] == key[0]) &
                       (df_ap['seed'] == key[1]) &
                       (df_ap['c'] == key[2])]
        auc_row = df_auc[(df_auc['dataset'] == key[0]) &
                         (df_auc['seed'] == key[1]) &
                         (df_auc['c'] == key[2])]
        max_f1_row = df_max_f1[(df_max_f1['dataset'] == key[0]) &
                               (df_max_f1['seed'] == key[1]) &
                               (df_max_f1['c'] == key[2])]

        comparison.append({
            'dataset': key[0],
            'seed': key[1],
            'c': key[2],
            'true_prior': row['true_prior'],
            'optimal_by_f1': row['optimal_prior'],
            'optimal_by_ap': ap_row['optimal_prior'].iloc[0] if not ap_row.empty else None,
            'optimal_by_auc': auc_row['optimal_prior'].iloc[0] if not auc_row.empty else None,
            'optimal_by_max_f1': max_f1_row['optimal_prior'].iloc[0] if not max_f1_row.empty else None,
            'agreement': None,  # Will compute
        })

    comp_df = pd.DataFrame(comparison)

    # Check agreement
    comp_df['agreement'] = comp_df.apply(
        lambda row: len(set([row['optimal_by_f1'], row['optimal_by_ap'],
                            row['optimal_by_auc'], row['optimal_by_max_f1']])) == 1,
        axis=1
    )

    return comp_df


def main():
    print("=" * 80)
    print("Optimal Prior Analysis (Threshold-Invariant Metrics)")
    print("=" * 80)
    print()

    # Load data for different metrics
    print("Loading data with different metrics...")
    df_raw_ap = load_robustness_results('test_ap')
    df_raw_auc = load_robustness_results('test_auc')
    df_raw_max_f1 = load_robustness_results('test_max_f1')
    df_raw_f1 = load_robustness_results('test_f1')
    print()

    # Find optimal for each metric
    print("Finding optimal prior for each metric...")
    df_opt_ap = find_optimal_prior_per_config(df_raw_ap, 'test_ap')
    df_opt_auc = find_optimal_prior_per_config(df_raw_auc, 'test_auc')
    df_opt_max_f1 = find_optimal_prior_per_config(df_raw_max_f1, 'test_max_f1')
    df_opt_f1 = find_optimal_prior_per_config(df_raw_f1, 'test_f1')
    print()

    # Save results
    output_dir = Path("results_robustness")

    df_opt_ap.to_csv(output_dir / "optimal_prior_by_ap.csv", index=False)
    df_opt_auc.to_csv(output_dir / "optimal_prior_by_auc.csv", index=False)
    df_opt_max_f1.to_csv(output_dir / "optimal_prior_by_max_f1.csv", index=False)

    print("Saved optimal prior analyses:")
    print(f"  - {output_dir / 'optimal_prior_by_ap.csv'}")
    print(f"  - {output_dir / 'optimal_prior_by_auc.csv'}")
    print(f"  - {output_dir / 'optimal_prior_by_max_f1.csv'}")
    print()

    # Analyze direction by c for each metric
    print("=" * 80)
    print("Direction Analysis by Label Frequency (c)")
    print("=" * 80)
    print()

    for metric_name, df_opt in [('AP', df_opt_ap), ('AUC', df_opt_auc),
                                 ('Max F1', df_opt_max_f1), ('F1', df_opt_f1)]:
        print(f"\n### Using {metric_name} ###")
        summary = analyze_direction_by_c(df_opt)
        print(summary.to_string(index=False))
        print()

    # Compare metrics
    print("=" * 80)
    print("Metric Agreement Analysis")
    print("=" * 80)
    print()

    comparison = compare_metrics(df_opt_f1, df_opt_ap, df_opt_auc, df_opt_max_f1)

    agreement_rate = comparison['agreement'].sum() / len(comparison) * 100
    print(f"Agreement rate (all 4 metrics choose same prior): {agreement_rate:.1f}%")
    print()

    # Show disagreements
    disagreements = comparison[~comparison['agreement']]
    if not disagreements.empty:
        print(f"Found {len(disagreements)} cases where metrics disagree:")
        print(disagreements[['dataset', 'seed', 'c', 'true_prior',
                            'optimal_by_f1', 'optimal_by_ap',
                            'optimal_by_auc', 'optimal_by_max_f1']].head(10).to_string(index=False))

    comparison.to_csv(output_dir / "metric_comparison.csv", index=False)
    print(f"\nSaved comparison to {output_dir / 'metric_comparison.csv'}")
    print()

    # Summary statistics
    print("=" * 80)
    print("Summary: Optimal Prior Direction (using AP)")
    print("=" * 80)
    print()

    df_numeric = df_opt_ap[df_opt_ap['direction'] != 'auto']

    print(f"Overall direction preference:")
    print(f"  Higher than true prior: {len(df_opt_ap[df_opt_ap['direction'] == 'higher'])} ({len(df_opt_ap[df_opt_ap['direction'] == 'higher'])/len(df_opt_ap)*100:.1f}%)")
    print(f"  Lower than true prior:  {len(df_opt_ap[df_opt_ap['direction'] == 'lower'])} ({len(df_opt_ap[df_opt_ap['direction'] == 'lower'])/len(df_opt_ap)*100:.1f}%)")
    print(f"  Auto (true prior best): {len(df_opt_ap[df_opt_ap['direction'] == 'auto'])} ({len(df_opt_ap[df_opt_ap['direction'] == 'auto'])/len(df_opt_ap)*100:.1f}%)")
    print()

    if not df_numeric.empty:
        print(f"Mean difference when not auto: {df_numeric['prior_diff'].mean():+.3f}")
        print(f"  When higher: {df_numeric[df_numeric['direction'] == 'higher']['prior_diff'].mean():+.3f}")
        print(f"  When lower:  {df_numeric[df_numeric['direction'] == 'lower']['prior_diff'].mean():+.3f}")


if __name__ == "__main__":
    main()
