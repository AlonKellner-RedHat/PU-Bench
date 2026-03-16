#!/usr/bin/env python3
"""Create a comparison table for the 5 methods across all datasets."""

import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np

# Methods to compare
METHODS = ['oracle_bce', 'vpu', 'vpu_mean', 'vpu_nomixup', 'vpu_nomixup_mean']

# Datasets to analyze
DATASETS = ['MNIST', 'FashionMNIST', 'KMNIST', '20News', 'IMDB',
            'Connect4', 'DermNet', 'AlzheimerMRI', 'CIFAR10']

# Seeds
SEEDS = [42, 43, 44]

def load_results():
    """Load all results from JSON files."""
    results = defaultdict(lambda: defaultdict(list))

    for seed in SEEDS:
        seed_dir = Path(f'results/seed_{seed}')
        if not seed_dir.exists():
            continue

        for dataset in DATASETS:
            file_path = seed_dir / f'{dataset}_case-control_random_c0.1_seed{seed}.json'
            if not file_path.exists():
                continue

            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                for method in METHODS:
                    if method in data:
                        results[dataset][method].append(data[method])
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    return results

def extract_metrics(results):
    """Extract key metrics from results."""
    metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for dataset, methods_data in results.items():
        for method, runs in methods_data.items():
            for run in runs:
                if 'test_results' not in run:
                    continue

                test_res = run['test_results']

                # Performance metrics
                metrics[method]['f1'][dataset].append(test_res.get('f1', np.nan))
                metrics[method]['auc'][dataset].append(test_res.get('auc', np.nan))

                # Calibration metrics
                metrics[method]['ece'][dataset].append(test_res.get('ece', np.nan))
                metrics[method]['brier'][dataset].append(test_res.get('brier', np.nan))
                metrics[method]['a_nice'][dataset].append(test_res.get('a_nice', np.nan))
                metrics[method]['s_nice'][dataset].append(test_res.get('s_nice', np.nan))

                # Training metrics
                if 'training_info' in run:
                    train_info = run['training_info']
                    metrics[method]['epochs'][dataset].append(train_info.get('epochs_to_best_val', np.nan))
                    metrics[method]['train_time'][dataset].append(train_info.get('total_training_time_seconds', np.nan))

    return metrics

def compute_aggregates(metrics):
    """Compute mean and std for each metric across datasets."""
    aggregates = {}

    for method in METHODS:
        aggregates[method] = {}

        for metric_name in ['f1', 'auc', 'ece', 'brier', 'a_nice', 's_nice', 'epochs', 'train_time']:
            all_values = []
            for dataset_values in metrics[method][metric_name].values():
                all_values.extend([v for v in dataset_values if not np.isnan(v)])

            if all_values:
                aggregates[method][metric_name + '_mean'] = np.mean(all_values)
                aggregates[method][metric_name + '_std'] = np.std(all_values)
            else:
                aggregates[method][metric_name + '_mean'] = np.nan
                aggregates[method][metric_name + '_std'] = np.nan

    return aggregates

def print_table(aggregates):
    """Print a formatted comparison table."""
    print("\n" + "="*120)
    print("COMPREHENSIVE 5-METHOD COMPARISON TABLE")
    print("="*120)

    # Header
    print(f"\n{'Metric':<25} {'oracle_bce':<20} {'vpu':<20} {'vpu_mean':<20} {'vpu_nomixup':<20} {'vpu_nomixup_mean':<20}")
    print("-" * 120)

    # Performance metrics
    print("\n--- PERFORMANCE METRICS ---")
    for metric in ['f1', 'auc']:
        values = []
        for method in METHODS:
            mean_val = aggregates[method].get(f'{metric}_mean', np.nan)
            std_val = aggregates[method].get(f'{metric}_std', np.nan)
            values.append((mean_val, std_val))

        metric_name = metric.upper()
        print(f"{metric_name:<25}", end="")
        for mean_val, std_val in values:
            if not np.isnan(mean_val):
                print(f"{mean_val:.4f} ± {std_val:.4f}   ", end="")
            else:
                print(f"{'N/A':<20}", end="")
        print()

    # Calibration metrics
    print("\n--- CALIBRATION METRICS (lower is better) ---")
    for metric in ['ece', 'brier', 'a_nice', 's_nice']:
        values = []
        for method in METHODS:
            mean_val = aggregates[method].get(f'{metric}_mean', np.nan)
            std_val = aggregates[method].get(f'{metric}_std', np.nan)
            values.append((mean_val, std_val))

        metric_name = metric.upper().replace('_', '-')
        print(f"{metric_name:<25}", end="")
        for mean_val, std_val in values:
            if not np.isnan(mean_val):
                print(f"{mean_val:.4f} ± {std_val:.4f}   ", end="")
            else:
                print(f"{'N/A':<20}", end="")
        print()

    # Training metrics
    print("\n--- CONVERGENCE & SPEED METRICS ---")
    for metric, label in [('epochs', 'Epochs to Best'), ('train_time', 'Training Time (s)')]:
        values = []
        for method in METHODS:
            mean_val = aggregates[method].get(f'{metric}_mean', np.nan)
            std_val = aggregates[method].get(f'{metric}_std', np.nan)
            values.append((mean_val, std_val))

        print(f"{label:<25}", end="")
        for mean_val, std_val in values:
            if not np.isnan(mean_val):
                if metric == 'epochs':
                    print(f"{mean_val:.1f} ± {std_val:.1f}       ", end="")
                else:
                    print(f"{mean_val:.2f} ± {std_val:.2f}      ", end="")
            else:
                print(f"{'N/A':<20}", end="")
        print()

    print("\n" + "="*120)

if __name__ == '__main__':
    print("Loading results...")
    results = load_results()

    print(f"Found results for {len(results)} datasets")
    for dataset, methods in results.items():
        print(f"  {dataset}: {list(methods.keys())}")

    print("\nExtracting metrics...")
    metrics = extract_metrics(results)

    print("\nComputing aggregates...")
    aggregates = compute_aggregates(metrics)

    print_table(aggregates)
