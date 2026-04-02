#!/usr/bin/env python3
"""
Extract comprehensive metrics from JSON result files including:
- Standard metrics: F1, AUC, Accuracy, Precision, Recall
- Calibration metrics: A-NICE, S-NICE, ECE, MCE, Brier
- Threshold-independent metrics: Max F1, AP
"""

import json
import pandas as pd
from pathlib import Path
import re

def parse_experiment_name(exp_name):
    """Parse experiment name to extract dataset, c, prior, seed"""
    # Examples:
    # IMDB_case-control_random_c0.1_seed42_prior0.3
    # FashionMNIST_case-control_random_c0.9_seed42
    # mnist_c0.3_prior0.5_seed42

    parts = {}

    # Extract dataset (first part before _case-control or _c)
    dataset_match = re.match(r'^([A-Za-z0-9]+)', exp_name)
    if dataset_match:
        parts['dataset'] = dataset_match.group(1)

    # Extract c value
    c_match = re.search(r'_c([\d.]+)', exp_name)
    if c_match:
        parts['c'] = float(c_match.group(1))

    # Extract prior
    prior_match = re.search(r'_prior([\d.]+)', exp_name)
    if prior_match:
        parts['prior'] = float(prior_match.group(1))
    else:
        parts['prior'] = None

    # Extract seed
    seed_match = re.search(r'_seed(\d+)', exp_name)
    if seed_match:
        parts['seed'] = int(seed_match.group(1))

    return parts

def extract_metrics_from_json(json_path):
    """Extract all metrics from a single JSON result file"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    experiment_name = data.get('experiment', '')
    exp_info = parse_experiment_name(experiment_name)

    results = []

    # Process each method's results
    for method_name, method_data in data.get('runs', {}).items():
        if 'best' not in method_data:
            continue

        metrics = method_data['best'].get('metrics', {})

        # Extract all test metrics
        row = {
            'experiment': experiment_name,
            'dataset': exp_info.get('dataset', ''),
            'method': method_name,
            'c': exp_info.get('c'),
            'prior': exp_info.get('prior'),
            'seed': exp_info.get('seed'),

            # Standard metrics
            'test_f1': metrics.get('test_f1'),
            'test_auc': metrics.get('test_auc'),
            'test_accuracy': metrics.get('test_accuracy'),
            'test_error': metrics.get('test_error'),
            'test_precision': metrics.get('test_precision'),
            'test_recall': metrics.get('test_recall'),

            # Threshold-independent metrics
            'test_max_f1': metrics.get('test_max_f1'),
            'test_ap': metrics.get('test_ap'),

            # Calibration metrics
            'test_anice': metrics.get('test_anice'),
            'test_snice': metrics.get('test_snice'),
            'test_ece': metrics.get('test_ece'),
            'test_mce': metrics.get('test_mce'),
            'test_brier': metrics.get('test_brier'),

            # Timing and convergence
            'duration': method_data.get('timing', {}).get('duration_seconds'),
            'convergence_epoch': method_data['best'].get('epoch'),
        }

        results.append(row)

    return results

def main():
    """Extract metrics from all JSON files in results directory"""
    results_dir = Path(__file__).parent.parent / "results"

    all_results = []
    json_files = list(results_dir.glob('seed_*/*.json'))

    print(f"Found {len(json_files)} JSON files")

    for json_file in json_files:
        try:
            results = extract_metrics_from_json(json_file)
            all_results.extend(results)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Standardize dataset names
    dataset_mapping = {
        'mnist': 'MNIST',
        'fashionmnist': 'FashionMNIST',
        'imdb': 'IMDB',
        'IMDB': 'IMDB',
        'FashionMNIST': 'FashionMNIST',
        'MNIST': 'MNIST',
        'Mushrooms': 'Mushrooms',
        'mushrooms': 'Mushrooms',
        'Spambase': 'Spambase',
        'spambase': 'Spambase',
        'Connect4': 'Connect4',
        'connect4': 'Connect4',
        '20News': '20News',
        '20news': '20News',
        'AlzheimerMRI': 'AlzheimerMRI',
        'alzheimermri': 'AlzheimerMRI',
    }

    df['dataset'] = df['dataset'].map(lambda x: dataset_mapping.get(x, x))

    # Sort by dataset, method, c, prior, seed
    df = df.sort_values(['dataset', 'method', 'c', 'seed'])

    # Save to CSV
    output_path = results_dir / "comprehensive_metrics.csv"
    df.to_csv(output_path, index=False)

    print(f"\nExtracted {len(df)} rows")
    print(f"Datasets: {sorted(df['dataset'].unique())}")
    print(f"Methods: {sorted(df['method'].unique())}")
    print(f"Seeds: {sorted(df['seed'].unique())}")
    print(f"\nSaved to: {output_path}")

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY: Metrics availability")
    print("="*80)

    metrics_cols = [
        'test_f1', 'test_auc', 'test_accuracy', 'test_precision', 'test_recall',
        'test_max_f1', 'test_ap',
        'test_anice', 'test_snice', 'test_ece', 'test_mce', 'test_brier'
    ]

    for metric in metrics_cols:
        if metric in df.columns:
            non_null = df[metric].notna().sum()
            pct = 100 * non_null / len(df)
            print(f"  {metric:20s}: {non_null:4d}/{len(df)} ({pct:5.1f}%)")

if __name__ == "__main__":
    main()
