#!/usr/bin/env python3
"""Analyze prior robustness experiment results

Loads robustness experiment data and calculates:
1. Performance metrics vs prior error
2. Degradation compared to true prior baseline
3. Method comparison (vpu_nomixup vs vpu_nomixup_mean vs vpu_nomixup_mean_prior)
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np


def load_robustness_results(results_dir="results_robustness"):
    """Load all robustness experiment results from JSON files"""
    results = []
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Error: {results_dir} does not exist")
        return pd.DataFrame()

    json_files = list(results_path.glob("seed_*/*.json"))
    print(f"Loading {len(json_files)} result files from {results_dir}/...")

    for json_file in json_files:
        try:
            with open(json_file) as f:
                data = json.load(f)

            # Extract experiment info
            exp_name = data.get('experiment', '')

            # Process each method's results
            for method, method_data in data.get('runs', {}).items():
                hyperparams = method_data.get('hyperparameters', {})
                dataset_info = method_data.get('dataset', {})
                best_metrics = method_data.get('best', {}).get('metrics', {})

                # Extract key parameters
                dataset_class = hyperparams.get('dataset_class')
                seed = hyperparams.get('seed')
                c = hyperparams.get('labeled_ratio')

                # Get true prior from training data
                true_prior = dataset_info.get('train', {}).get('prior')

                # Get method_prior (will be None for methods that don't use it)
                method_prior = hyperparams.get('method_prior')

                # Calculate prior error (only for methods with method_prior)
                if method_prior is not None and true_prior is not None:
                    prior_error = abs(method_prior - true_prior)
                else:
                    prior_error = 0.0  # No error for methods without priors

                # Extract all metrics
                result_row = {
                    'dataset': dataset_class,
                    'seed': seed,
                    'c': c,
                    'method': method,
                    'true_prior': true_prior,
                    'method_prior': method_prior,
                    'prior_error': prior_error,
                    'experiment': exp_name,
                }

                # Add all performance metrics
                for metric_name, metric_value in best_metrics.items():
                    result_row[metric_name] = metric_value

                results.append(result_row)

        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    df = pd.DataFrame(results)
    print(f"Loaded {len(df)} method runs across {len(json_files)} experiments")

    return df


def calculate_degradation(df):
    """Calculate performance drop vs true prior baseline"""
    results = []

    # Only analyze vpu_nomixup_mean_prior (the method using priors)
    df_prior = df[df['method'] == 'vpu_nomixup_mean_prior'].copy()

    # Group by dataset, seed, c
    for (dataset, seed, c), group in df_prior.groupby(['dataset', 'seed', 'c']):
        # Find baseline (method_prior=None means it used true prior, or auto)
        baseline = group[group['method_prior'].isna()]

        if baseline.empty:
            # If no None, look for one with prior_error=0
            baseline = group[group['prior_error'] == 0]

        if baseline.empty:
            print(f"Warning: No baseline found for {dataset}, seed={seed}, c={c}")
            continue

        baseline_f1 = baseline['test_f1'].values[0]
        baseline_ap = baseline['test_ap'].values[0]
        baseline_max_f1 = baseline['test_max_f1'].values[0]

        # Calculate degradation for each misspecified prior
        for _, row in group[group['prior_error'] > 0].iterrows():
            # Performance drops (absolute and relative)
            f1_drop_abs = baseline_f1 - row['test_f1']
            f1_drop_rel = (f1_drop_abs / baseline_f1 * 100) if baseline_f1 > 0 else 0

            ap_drop_abs = baseline_ap - row['test_ap']
            ap_drop_rel = (ap_drop_abs / baseline_ap * 100) if baseline_ap > 0 else 0

            max_f1_drop_abs = baseline_max_f1 - row['test_max_f1']
            max_f1_drop_rel = (max_f1_drop_abs / baseline_max_f1 * 100) if baseline_max_f1 > 0 else 0

            results.append({
                'dataset': dataset,
                'seed': seed,
                'c': c,
                'method': row['method'],
                'true_prior': row['true_prior'],
                'method_prior': row['method_prior'],
                'prior_error': row['prior_error'],
                # Baseline performance
                'baseline_f1': baseline_f1,
                'baseline_ap': baseline_ap,
                'baseline_max_f1': baseline_max_f1,
                # Current performance
                'test_f1': row['test_f1'],
                'test_ap': row['test_ap'],
                'test_max_f1': row['test_max_f1'],
                # Degradation metrics
                'f1_drop_abs': f1_drop_abs,
                'f1_drop_rel_pct': f1_drop_rel,
                'ap_drop_abs': ap_drop_abs,
                'ap_drop_rel_pct': ap_drop_rel,
                'max_f1_drop_abs': max_f1_drop_abs,
                'max_f1_drop_rel_pct': max_f1_drop_rel,
            })

    return pd.DataFrame(results)


def summarize_robustness(df_degradation):
    """Create summary statistics for robustness analysis"""

    print("\n" + "="*80)
    print("Prior Robustness Summary")
    print("="*80)

    # Overall statistics
    print(f"\nTotal experiments with misspecified priors: {len(df_degradation)}")
    print(f"Prior error range: [{df_degradation['prior_error'].min():.3f}, {df_degradation['prior_error'].max():.3f}]")

    # Performance degradation by prior error bins
    error_bins = [0, 0.1, 0.2, 0.3, 0.5, 1.0]
    df_degradation['error_bin'] = pd.cut(df_degradation['prior_error'], bins=error_bins)

    print("\n" + "-"*80)
    print("Mean F1 Drop (%) by Prior Error Level")
    print("-"*80)
    summary = df_degradation.groupby('error_bin')['f1_drop_rel_pct'].agg(['mean', 'std', 'min', 'max', 'count'])
    print(summary.to_string())

    print("\n" + "-"*80)
    print("Mean AP Drop (%) by Prior Error Level")
    print("-"*80)
    summary_ap = df_degradation.groupby('error_bin')['ap_drop_rel_pct'].agg(['mean', 'std', 'min', 'max', 'count'])
    print(summary_ap.to_string())

    # By dataset
    print("\n" + "-"*80)
    print("F1 Drop (%) by Dataset (mean across all prior errors)")
    print("-"*80)
    by_dataset = df_degradation.groupby('dataset')['f1_drop_rel_pct'].agg(['mean', 'std', 'count'])
    print(by_dataset.to_string())

    # By label frequency
    print("\n" + "-"*80)
    print("F1 Drop (%) by Label Frequency (c)")
    print("-"*80)
    by_c = df_degradation.groupby('c')['f1_drop_rel_pct'].agg(['mean', 'std', 'count'])
    print(by_c.to_string())

    return summary


def save_analysis_results(df_all, df_degradation, output_dir="results_robustness"):
    """Save analysis results to CSV"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save full results
    csv_path = output_path / "robustness_full_results.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"\n✓ Saved full results to {csv_path}")

    # Save degradation analysis
    csv_path_deg = output_path / "robustness_degradation.csv"
    df_degradation.to_csv(csv_path_deg, index=False)
    print(f"✓ Saved degradation analysis to {csv_path_deg}")


def main():
    # Load results
    df = load_robustness_results()

    if df.empty:
        print("No results found. Run experiments first with: bash scripts/run_prior_robustness.sh")
        return

    # Calculate degradation
    df_degradation = calculate_degradation(df)

    if df_degradation.empty:
        print("No degradation data available. Check that experiments included baseline (auto) prior.")
        return

    # Summarize
    summarize_robustness(df_degradation)

    # Save results
    save_analysis_results(df, df_degradation)

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
