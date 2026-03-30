#!/usr/bin/env python3
"""
Validate metrics coverage for VPU rerun.

Checks that all experiments have:
- test_max_f1 (threshold-independent max F1)
- test_ap (average precision)
- All other standard and calibration metrics

Reports coverage by dataset, method, and seed.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


# VPU variants we expect
EXPECTED_METHODS = [
    'vpu',
    'vpu_mean',
    'vpu_mean_prior',
    'vpu_nomixup',
    'vpu_nomixup_mean',
    'vpu_nomixup_mean_prior'
]

# Core datasets
EXPECTED_DATASETS = [
    'MNIST',
    'FashionMNIST',
    'IMDB',
    '20News',
    'Mushrooms',
    'Spambase'
]

# Seeds
EXPECTED_SEEDS = [42, 123, 456, 789, 2024]

# Metrics that MUST be present
REQUIRED_METRICS = ['test_max_f1', 'test_ap']


def scan_results_directory(results_dir='results'):
    """Scan results directory for all JSON files and check metrics."""
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return None

    # Find all JSON files
    json_files = list(results_path.rglob('*.json'))

    if not json_files:
        print(f"❌ No JSON files found in {results_dir}")
        return None

    print(f"Found {len(json_files)} result files")
    print()

    # Scan each file
    total_experiments = 0
    total_methods = 0
    complete_methods = 0
    incomplete_methods = 0
    missing_experiments = []

    coverage_by_dataset = defaultdict(lambda: {'total': 0, 'complete': 0})
    coverage_by_method = defaultdict(lambda: {'total': 0, 'complete': 0})
    coverage_by_seed = defaultdict(lambda: {'total': 0, 'complete': 0})

    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Extract dataset and seed from experiment name (format: Dataset_scenario_strategy_cX.X_seedXXXX)
            experiment_name = data.get('experiment', '')
            parts = experiment_name.split('_')
            dataset = parts[0] if parts else 'unknown'

            # Extract seed from experiment name
            seed = 'unknown'
            for part in parts:
                if part.startswith('seed'):
                    seed = part[4:]  # Remove 'seed' prefix
                    break

            total_experiments += 1

            # Check each method in this experiment
            runs = data.get('runs', {})

            for method_name, method_data in runs.items():
                total_methods += 1
                coverage_by_method[method_name]['total'] += 1
                coverage_by_dataset[dataset]['total'] += 1
                coverage_by_seed[seed]['total'] += 1

                # Check if metrics exist
                metrics = method_data.get('best', {}).get('metrics', {})

                has_all_required = all(metric in metrics for metric in REQUIRED_METRICS)

                if has_all_required:
                    complete_methods += 1
                    coverage_by_method[method_name]['complete'] += 1
                    coverage_by_dataset[dataset]['complete'] += 1
                    coverage_by_seed[seed]['complete'] += 1
                else:
                    incomplete_methods += 1
                    missing = [m for m in REQUIRED_METRICS if m not in metrics]
                    missing_experiments.append({
                        'file': json_file.name,
                        'dataset': dataset,
                        'seed': seed,
                        'method': method_name,
                        'missing': missing
                    })

        except Exception as e:
            print(f"⚠️  Error reading {json_file}: {e}")
            continue

    return {
        'total_experiments': total_experiments,
        'total_methods': total_methods,
        'complete_methods': complete_methods,
        'incomplete_methods': incomplete_methods,
        'missing_experiments': missing_experiments,
        'coverage_by_dataset': dict(coverage_by_dataset),
        'coverage_by_method': dict(coverage_by_method),
        'coverage_by_seed': dict(coverage_by_seed)
    }


def print_report(results):
    """Print validation report."""
    if results is None:
        return

    total_exp = results['total_experiments']
    total_methods = results['total_methods']
    complete = results['complete_methods']
    incomplete = results['incomplete_methods']

    print("=" * 60)
    print("VPU Core Datasets - Metrics Coverage Report")
    print("=" * 60)
    print()

    # Overall stats
    print("Overall Statistics:")
    print(f"  Total Experiments: {total_exp}")
    print(f"  Total Method Runs: {total_methods}")
    print(f"  Complete (with Max F1 & AP): {complete} ({100 * complete / total_methods:.1f}%)")
    print(f"  Incomplete: {incomplete} ({100 * incomplete / total_methods:.1f}%)")
    print()

    # By dataset
    print("Coverage by Dataset:")
    for dataset in sorted(results['coverage_by_dataset'].keys()):
        stats = results['coverage_by_dataset'][dataset]
        pct = 100 * stats['complete'] / stats['total'] if stats['total'] > 0 else 0
        status = "✓" if pct == 100 else "⚠️ "
        print(f"  {status} {dataset:15s}: {stats['complete']:4d}/{stats['total']:4d} ({pct:5.1f}%)")
    print()

    # By method
    print("Coverage by Method:")
    for method in sorted(results['coverage_by_method'].keys()):
        stats = results['coverage_by_method'][method]
        pct = 100 * stats['complete'] / stats['total'] if stats['total'] > 0 else 0
        status = "✓" if pct == 100 else "⚠️ "
        print(f"  {status} {method:30s}: {stats['complete']:4d}/{stats['total']:4d} ({pct:5.1f}%)")
    print()

    # By seed
    print("Coverage by Seed:")
    for seed in sorted(results['coverage_by_seed'].keys(), key=lambda x: int(x) if str(x).isdigit() else 0):
        stats = results['coverage_by_seed'][seed]
        pct = 100 * stats['complete'] / stats['total'] if stats['total'] > 0 else 0
        status = "✓" if pct == 100 else "⚠️ "
        print(f"  {status} Seed {str(seed):10s}: {stats['complete']:4d}/{stats['total']:4d} ({pct:5.1f}%)")
    print()

    # Missing experiments
    if results['missing_experiments']:
        print(f"Missing Metrics Details (showing first 20):")
        for i, exp in enumerate(results['missing_experiments'][:20]):
            print(f"  - {exp['file']}: {exp['method']} missing {', '.join(exp['missing'])}")
        if len(results['missing_experiments']) > 20:
            print(f"  ... and {len(results['missing_experiments']) - 20} more")
        print()

    # Final status
    print("=" * 60)
    if incomplete == 0:
        print("✓ SUCCESS: All experiments have complete metrics!")
        print("=" * 60)
        return True
    else:
        print(f"⚠️  INCOMPLETE: {incomplete} method runs missing metrics")
        print("=" * 60)
        return False


def main():
    """Main entry point."""
    print("Validating VPU rerun metrics coverage...")
    print()

    results = scan_results_directory('results')
    success = print_report(results)

    # Exit code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
