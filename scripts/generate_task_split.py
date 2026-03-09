"""Generate balanced task split from profiling results.

This script creates a YAML file with training/test task splits where:
- Training: 2 tabular, 2 image, 1 text datasets
- Testing: remaining datasets
- Each dataset has 9 configs (3 c_values × 3 prior_values)

Usage:
    python scripts/generate_task_split.py --input task_speeds.json --output task_split.yaml
"""

import argparse
import json
import yaml
from pathlib import Path


def load_profiling_results(input_path: str) -> dict:
    """Load profiling results from JSON file."""
    with open(input_path, 'r') as f:
        return json.load(f)


def generate_task_configs(c_values: list, prior_values: list) -> list:
    """Generate all config combinations for a dataset.

    Args:
        c_values: List of c (label ratio) values
        prior_values: List of prior (class prior) values

    Returns:
        List of config dicts with c and prior values
    """
    configs = []
    for c in c_values:
        for prior in prior_values:
            configs.append({'c': c, 'prior': prior})
    return configs


def create_task_split(profiling_data: dict,
                     c_values: list = [0.3, 0.5, 0.7],
                     prior_values: list = [0.3, 0.5, 0.7]) -> dict:
    """Create balanced train/test split.

    Args:
        profiling_data: Profiling results with task metadata
        c_values: List of c values to use for grid
        prior_values: List of prior values to use for grid

    Returns:
        Dict with training_tasks and test_tasks lists
    """
    # Extract dataset info from profiling results
    datasets = {}
    for task_id, metadata in profiling_data['task_metadata'].items():
        dataset = metadata['dataset']
        datasets[dataset] = {
            'data_type': metadata['data_type'],
            'speed': metadata['time_per_epoch']
        }

    # Group by data type
    by_type = {'image': [], 'tabular': [], 'text': []}
    for dataset, info in datasets.items():
        by_type[info['data_type']].append({
            'dataset': dataset,
            'speed': info['speed']
        })

    # Sort each type by speed (fastest first)
    for data_type in by_type:
        by_type[data_type].sort(key=lambda x: x['speed'])

    # Select training datasets: 2 tabular, 2 image, 1 text
    training_datasets = []
    testing_datasets = []

    # Take 2 fastest tabular for training
    training_datasets.extend([d['dataset'] for d in by_type['tabular'][:2]])
    testing_datasets.extend([d['dataset'] for d in by_type['tabular'][2:]])

    # Take 2 fastest image for training
    training_datasets.extend([d['dataset'] for d in by_type['image'][:2]])
    testing_datasets.extend([d['dataset'] for d in by_type['image'][2:]])

    # Take 1 fastest text for training
    training_datasets.extend([d['dataset'] for d in by_type['text'][:1]])
    testing_datasets.extend([d['dataset'] for d in by_type['text'][1:]])

    # Generate configs for each dataset
    configs = generate_task_configs(c_values, prior_values)

    # Build training tasks
    training_tasks = []
    for dataset in training_datasets:
        data_type = datasets[dataset]['data_type']
        speed = datasets[dataset]['speed']
        for config in configs:
            training_tasks.append({
                'dataset': dataset,
                'c_value': config['c'],
                'prior': config['prior'],
                'data_type': data_type,
                'speed': round(speed, 2)
            })

    # Build test tasks
    test_tasks = []
    for dataset in testing_datasets:
        data_type = datasets[dataset]['data_type']
        speed = datasets[dataset]['speed']
        for config in configs:
            test_tasks.append({
                'dataset': dataset,
                'c_value': config['c'],
                'prior': config['prior'],
                'data_type': data_type,
                'speed': round(speed, 2)
            })

    # Create summary
    train_by_type = {'image': 0, 'tabular': 0, 'text': 0}
    test_by_type = {'image': 0, 'tabular': 0, 'text': 0}

    for task in training_tasks:
        train_by_type[task['data_type']] += 1
    for task in test_tasks:
        test_by_type[task['data_type']] += 1

    train_datasets_by_type = {
        'image': [d for d in training_datasets if datasets[d]['data_type'] == 'image'],
        'tabular': [d for d in training_datasets if datasets[d]['data_type'] == 'tabular'],
        'text': [d for d in training_datasets if datasets[d]['data_type'] == 'text']
    }

    test_datasets_by_type = {
        'image': [d for d in testing_datasets if datasets[d]['data_type'] == 'image'],
        'tabular': [d for d in testing_datasets if datasets[d]['data_type'] == 'tabular'],
        'text': [d for d in testing_datasets if datasets[d]['data_type'] == 'text']
    }

    return {
        'training_tasks': training_tasks,
        'test_tasks': test_tasks,
        'summary': {
            'training': {
                'total_tasks': len(training_tasks),
                'total_datasets': len(training_datasets),
                'datasets_by_type': train_datasets_by_type,
                'tasks_by_type': train_by_type,
                'configs_per_dataset': len(configs)
            },
            'testing': {
                'total_tasks': len(test_tasks),
                'total_datasets': len(testing_datasets),
                'datasets_by_type': test_datasets_by_type,
                'tasks_by_type': test_by_type,
                'configs_per_dataset': len(configs)
            }
        }
    }


def save_task_split(split_data: dict, output_path: str):
    """Save task split to YAML file with header comment."""

    summary = split_data['summary']

    header = f"""# Balanced train/test split generated from profiling results
# Generated programmatically to ensure correct structure
#
# Configuration:
# - c_values: [0.3, 0.5, 0.7]
# - prior_values: [0.3, 0.5, 0.7]
# - Configs per dataset: {summary['training']['configs_per_dataset']} (3×3 grid)
#
# Training set ({summary['training']['total_tasks']} tasks from {summary['training']['total_datasets']} datasets):
#   - {summary['training']['tasks_by_type']['tabular']} tabular tasks from {len(summary['training']['datasets_by_type']['tabular'])} datasets: {', '.join(summary['training']['datasets_by_type']['tabular'])}
#   - {summary['training']['tasks_by_type']['image']} image tasks from {len(summary['training']['datasets_by_type']['image'])} datasets: {', '.join(summary['training']['datasets_by_type']['image'])}
#   - {summary['training']['tasks_by_type']['text']} text tasks from {len(summary['training']['datasets_by_type']['text'])} datasets: {', '.join(summary['training']['datasets_by_type']['text'])}
#
# Test set ({summary['testing']['total_tasks']} tasks from {summary['testing']['total_datasets']} datasets):
#   - {summary['testing']['tasks_by_type']['tabular']} tabular tasks from {len(summary['testing']['datasets_by_type']['tabular'])} datasets: {', '.join(summary['testing']['datasets_by_type']['tabular'])}
#   - {summary['testing']['tasks_by_type']['image']} image tasks from {len(summary['testing']['datasets_by_type']['image'])} datasets: {', '.join(summary['testing']['datasets_by_type']['image'])}
#   - {summary['testing']['tasks_by_type']['text']} text tasks from {len(summary['testing']['datasets_by_type']['text'])} datasets: {', '.join(summary['testing']['datasets_by_type']['text'])}

"""

    # Prepare YAML structure
    yaml_data = {
        'training_tasks': split_data['training_tasks'],
        'test_tasks': split_data['test_tasks']
    }

    # Write YAML with header
    with open(output_path, 'w') as f:
        f.write(header)
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    print(f"\n{'='*70}")
    print("TASK SPLIT GENERATED")
    print('='*70)
    print(f"\nOutput file: {output_path}")
    print(f"\nTraining: {summary['training']['total_tasks']} tasks from {summary['training']['total_datasets']} datasets")
    print(f"  - Tabular: {summary['training']['tasks_by_type']['tabular']} tasks ({len(summary['training']['datasets_by_type']['tabular'])} datasets)")
    print(f"  - Image: {summary['training']['tasks_by_type']['image']} tasks ({len(summary['training']['datasets_by_type']['image'])} datasets)")
    print(f"  - Text: {summary['training']['tasks_by_type']['text']} tasks ({len(summary['training']['datasets_by_type']['text'])} datasets)")
    print(f"\nTesting: {summary['testing']['total_tasks']} tasks from {summary['testing']['total_datasets']} datasets")
    print(f"  - Tabular: {summary['testing']['tasks_by_type']['tabular']} tasks ({len(summary['testing']['datasets_by_type']['tabular'])} datasets)")
    print(f"  - Image: {summary['testing']['tasks_by_type']['image']} tasks ({len(summary['testing']['datasets_by_type']['image'])} datasets)")
    print(f"  - Text: {summary['testing']['tasks_by_type']['text']} tasks ({len(summary['testing']['datasets_by_type']['text'])} datasets)")
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate task split from profiling results")
    parser.add_argument("--input", type=str, default="task_speeds.json",
                        help="Input JSON file with profiling results")
    parser.add_argument("--output", type=str, default="task_split.yaml",
                        help="Output YAML file for task split")
    parser.add_argument("--c-values", type=float, nargs="+", default=[0.3, 0.5, 0.7],
                        help="C values for grid (default: 0.3 0.5 0.7)")
    parser.add_argument("--prior-values", type=float, nargs="+", default=[0.3, 0.5, 0.7],
                        help="Prior values for grid (default: 0.3 0.5 0.7)")
    args = parser.parse_args()

    # Load profiling results
    print(f"Loading profiling results from {args.input}...")
    profiling_data = load_profiling_results(args.input)

    # Generate task split
    print(f"Generating task split with grid: c={args.c_values}, prior={args.prior_values}...")
    split_data = create_task_split(profiling_data, args.c_values, args.prior_values)

    # Save to YAML
    save_task_split(split_data, args.output)


if __name__ == "__main__":
    main()
