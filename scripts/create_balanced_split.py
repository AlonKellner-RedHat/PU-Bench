"""Create balanced training/test split from profiling results.

This script creates a balanced split where:
- Each dataset (all configs) goes EITHER to training OR testing
- Training: 2 unique image datasets + 2 tabular + 1 text = 5 datasets × 9 configs = 45 tasks
- Testing: 2 unique image datasets + 1 tabular + 1 text = 4 datasets × 9 configs = 36 tasks
- Each dataset contributes all 9 configs (3 c_values × 3 prior_values) to either training or testing

Usage:
    python scripts/create_balanced_split.py --input task_speeds.json --output task_split.yaml
"""

import argparse
import json
import yaml
from collections import defaultdict


def load_profiling_results(input_file: str):
    """Load profiling results from JSON file."""
    with open(input_file, 'r') as f:
        data = json.load(f)
    return data['task_metadata']


def group_tasks_by_dataset(task_metadata: dict):
    """Group tasks by dataset name.

    Returns:
        dict mapping dataset -> list of task configs
    """
    grouped = defaultdict(list)

    for task_id, meta in task_metadata.items():
        dataset = meta['dataset']
        grouped[dataset].append({
            'dataset': dataset,
            'c_value': meta['c_value'],
            'prior': meta['prior'],
            'data_type': meta['data_type'],
            'speed': meta['time_per_epoch']
        })

    # Sort each dataset's tasks by speed (fastest first)
    for dataset in grouped:
        grouped[dataset].sort(key=lambda x: x['speed'])

    return grouped


def create_balanced_split(grouped_tasks: dict):
    """Create balanced split satisfying user's constraints.

    User requirements:
    - 2 unique image datasets for training, rest for testing
    - 2 unique tabular datasets for training, rest for testing
    - 1 unique text dataset for training, rest for testing
    - All configs of a dataset go together (EITHER training OR testing)
    - All datasets have equal number of configs (9 each)
    - More training tasks than testing tasks (45 vs 36)
    """
    # Separate datasets by type
    image_datasets = {k: v for k, v in grouped_tasks.items()
                     if v[0]['data_type'] == 'image'}
    tabular_datasets = {k: v for k, v in grouped_tasks.items()
                       if v[0]['data_type'] == 'tabular'}
    text_datasets = {k: v for k, v in grouped_tasks.items()
                    if v[0]['data_type'] == 'text'}

    print("Dataset inventory:")
    print(f"  Image datasets: {list(image_datasets.keys())}")
    print(f"  Tabular datasets: {list(tabular_datasets.keys())}")
    print(f"  Text datasets: {list(text_datasets.keys())}")
    print()

    # Select fastest datasets for training (for efficiency)
    # Sort by average speed
    def avg_speed(tasks):
        return sum(t['speed'] for t in tasks) / len(tasks)

    image_by_speed = sorted(image_datasets.items(), key=lambda x: avg_speed(x[1]))
    tabular_by_speed = sorted(tabular_datasets.items(), key=lambda x: avg_speed(x[1]))
    text_by_speed = sorted(text_datasets.items(), key=lambda x: avg_speed(x[1]))

    # Training: 2 fastest image + 2 fastest tabular + 1 fastest text
    # This gives more training tasks than testing tasks (45 vs 36)
    training_datasets = []
    testing_datasets = []

    # Image: 2 for training, rest for testing
    for i, (dataset, tasks) in enumerate(image_by_speed):
        if i < 2:  # First 2 (fastest)
            training_datasets.append((dataset, tasks))
        else:
            testing_datasets.append((dataset, tasks))

    # Tabular: 2 for training, rest for testing
    for i, (dataset, tasks) in enumerate(tabular_by_speed):
        if i < 2:  # First 2 (fastest)
            training_datasets.append((dataset, tasks))
        else:
            testing_datasets.append((dataset, tasks))

    # Text: 1 for training, rest for testing
    for i, (dataset, tasks) in enumerate(text_by_speed):
        if i < 1:  # First 1 (fastest)
            training_datasets.append((dataset, tasks))
        else:
            testing_datasets.append((dataset, tasks))

    return training_datasets, testing_datasets


def create_split_yaml(training_datasets, testing_datasets, output_file: str):
    """Create YAML file with training/test split."""

    # Flatten tasks
    training_tasks = []
    for dataset, tasks in training_datasets:
        training_tasks.extend(tasks)

    testing_tasks = []
    for dataset, tasks in testing_datasets:
        testing_tasks.extend(tasks)

    # Create YAML structure
    split_data = {
        'training_tasks': training_tasks,
        'test_tasks': testing_tasks
    }

    # Add summary comment
    training_datasets_names = [d for d, _ in training_datasets]
    testing_datasets_names = [d for d, _ in testing_datasets]

    print("=" * 70)
    print("BALANCED TASK SPLIT")
    print("=" * 70)
    print()
    print(f"Training datasets ({len(training_datasets_names)}): {training_datasets_names}")
    print(f"  Total tasks: {len(training_tasks)}")
    for dataset, tasks in training_datasets:
        print(f"    {dataset}: {len(tasks)} configs ({tasks[0]['data_type']})")
    print()
    print(f"Testing datasets ({len(testing_datasets_names)}): {testing_datasets_names}")
    print(f"  Total tasks: {len(testing_tasks)}")
    for dataset, tasks in testing_datasets:
        print(f"    {dataset}: {len(tasks)} configs ({tasks[0]['data_type']})")
    print()
    print(f"Saved to: {output_file}")
    print("=" * 70)

    # Save to file
    with open(output_file, 'w') as f:
        f.write(f"# Balanced task split for meta-learning\n")
        f.write(f"# Training: {len(training_datasets_names)} datasets = {len(training_tasks)} tasks\n")
        f.write(f"# Testing: {len(testing_datasets_names)} datasets = {len(testing_tasks)} tasks\n")
        f.write(f"#\n")
        f.write(f"# Training datasets: {', '.join(training_datasets_names)}\n")
        f.write(f"# Testing datasets: {', '.join(testing_datasets_names)}\n")
        f.write(f"#\n")
        f.write(f"# Each dataset contributes ALL configs to either training or testing\n")
        f.write(f"# All datasets have 9 configs (3 c_values × 3 prior_values) for perfect balance\n\n")
        yaml.dump(split_data, f, default_flow_style=False, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description="Create balanced training/test split")
    parser.add_argument("--input", type=str, default="task_speeds.json",
                        help="Input profiling results JSON")
    parser.add_argument("--output", type=str, default="task_split.yaml",
                        help="Output split YAML file")
    args = parser.parse_args()

    # Load profiling results
    print(f"Loading profiling results from: {args.input}")
    task_metadata = load_profiling_results(args.input)
    print(f"  Total tasks profiled: {len(task_metadata)}")
    print()

    # Group by dataset
    grouped_tasks = group_tasks_by_dataset(task_metadata)

    # Create balanced split
    training_datasets, testing_datasets = create_balanced_split(grouped_tasks)

    # Save to YAML
    create_split_yaml(training_datasets, testing_datasets, args.output)


if __name__ == "__main__":
    main()
