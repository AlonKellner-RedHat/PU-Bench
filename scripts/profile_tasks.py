"""Profile all candidate tasks to measure training speed.

This script trains one epoch with PN naive loss for each candidate task configuration
and measures the time required. Results are saved to task_speeds.json for use in
creating the training/test split.

Usage:
    python scripts/profile_tasks.py --output task_speeds.json
"""

import argparse
import json
import time
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from config.run_param_sweep import load_dataset_config, expand_dataset_grid


# Candidate tasks organized by data type
# For profiling, we only need one config per dataset to measure speed
# Training speed should be similar across different c/prior values for same dataset
# Total: 9 datasets × 1 config = 9 tasks

# Map dataset names to their config file paths
DATASET_CONFIG_PATHS = {
    "mnist": "config/datasets_typical/param_sweep_mnist_single.yaml",
    "fashionmnist": "config/datasets_typical/param_sweep_fashionmnist_single.yaml",
    "cifar10": "config/datasets_typical/param_sweep_cifar10_single.yaml",
    "alzheimermri": "config/datasets_typical/param_sweep_alzheimermri_single.yaml",
    "mushrooms": "config/datasets_typical/param_sweep_mushrooms_single.yaml",
    "spambase": "config/datasets_typical/param_sweep_spambase_single.yaml",
    "connect4": "config/datasets_typical/param_sweep_connect4_single.yaml",
    "20news": "config/datasets_typical/param_sweep_20news_single.yaml",
    "imdb": "config/datasets_typical/param_sweep_imdb_single.yaml",
}

CANDIDATE_TASKS = {}

# Single representative config for profiling (c=0.5, prior=0.5)
PROFILE_CONFIG = [{"c": 0.5, "prior": 0.5}]

# Image tasks (4 datasets × 1 config = 4 tasks)
for dataset_name, dataset_class in [
    ("mnist", "MNIST"),
    ("fashionmnist", "FashionMNIST"),
    ("cifar10", "CIFAR10"),
    ("alzheimermri", "AlzheimerMRI"),
]:
    CANDIDATE_TASKS[dataset_name] = {
        "dataset_class": dataset_class,
        "configs": PROFILE_CONFIG,
        "config_path": DATASET_CONFIG_PATHS[dataset_name]
    }

# Tabular tasks (3 datasets × 1 config = 3 tasks)
for dataset_name, dataset_class in [
    ("mushrooms", "Mushrooms"),
    ("spambase", "Spambase"),
    ("connect4", "Connect4"),
]:
    CANDIDATE_TASKS[dataset_name] = {
        "dataset_class": dataset_class,
        "configs": PROFILE_CONFIG,
        "config_path": DATASET_CONFIG_PATHS[dataset_name]
    }

# Text tasks (2 datasets × 1 config = 2 tasks)
for dataset_name, dataset_class in [
    ("20news", "20News"),
    ("imdb", "IMDB"),
]:
    CANDIDATE_TASKS[dataset_name] = {
        "dataset_class": dataset_class,
        "configs": PROFILE_CONFIG,
        "config_path": DATASET_CONFIG_PATHS[dataset_name]
    }


def get_data_type(dataset_name: str) -> str:
    """Get data type for a dataset."""
    if dataset_name in ["mnist", "fashionmnist", "cifar10", "alzheimermri"]:
        return "image"
    elif dataset_name in ["mushrooms", "spambase", "connect4"]:
        return "tabular"
    elif dataset_name in ["20news", "imdb"]:
        return "text"
    else:
        return "unknown"


def format_task_id(dataset: str, c_value: float, prior: float) -> str:
    """Format task identifier."""
    return f"{dataset}_c{c_value:.1f}_prior{prior:.1f}"


def profile_task(dataset_name: str, c_value: float, prior: float, batch_size: int = 256):
    """Profile a single task configuration using PNNaiveTrainer.

    Args:
        dataset_name: Name of the dataset
        c_value: Label noise level
        prior: Class prior
        batch_size: Batch size for training

    Returns:
        Time in seconds for one epoch
    """
    from train.pn_naive_trainer import PNNaiveTrainer
    import copy

    task_id = format_task_id(dataset_name, c_value, prior)

    try:
        # Get dataset info
        dataset_info = CANDIDATE_TASKS[dataset_name]
        config_path = dataset_info["config_path"]

        # Load dataset config (same as benchmark)
        dataset_cfg = load_dataset_config(config_path)
        dataset_class, data_runs = expand_dataset_grid(dataset_cfg)

        # Get the first run config (we only need one for profiling)
        base_params = copy.deepcopy(data_runs[0]) if data_runs else {}

        # Override with profiling-specific values
        params = base_params
        params.update({
            # Override dataset params for profiling
            'labeled_ratio': c_value,
            'target_prevalence': prior,
            'random_seed': 42,
            'val_ratio': 0.1,

            # Training config
            'num_epochs': 1,  # Single epoch for profiling
            'batch_size': batch_size,
            'lr': 0.001,
            'weight_decay': 1e-4,
            'optimizer': 'adam',

            # Model config
            'init_bias_from_prior': True,

            # Logging (suppress during profiling)
            'log_interval': 999,
            'silence_metrics_before_epoch': 999,
        })

        # Create experiment name
        exp_name = f"profile_{task_id}"

        # Train using PNNaiveTrainer for 1 epoch
        trainer = PNNaiveTrainer(
            method='pn_naive',
            experiment=exp_name,
            params=params
        )

        # Time the single epoch training
        start_time = time.time()
        trainer.run()
        elapsed = time.time() - start_time

        return elapsed

    except Exception as e:
        print(f"  ERROR profiling {task_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Profile candidate tasks")
    parser.add_argument("--output", type=str, default="task_speeds.json",
                        help="Output JSON file for speeds")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for profiling")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    # Setup
    print(f"Batch size: {args.batch_size}")
    print(f"Random seed: {args.seed}")
    print()
    print("Note: Device selection handled automatically by PNNaiveTrainer (MPS → CUDA → CPU)")
    print()

    # Profile all tasks
    task_speeds = {}
    task_metadata = {}

    print("=" * 70)
    print("TASK PROFILING")
    print("=" * 70)
    print()

    for dataset_name, dataset_info in CANDIDATE_TASKS.items():
        data_type = get_data_type(dataset_name)
        print(f"Dataset: {dataset_name} ({data_type})")

        for config in dataset_info["configs"]:
            c_value = config["c"]
            prior = config["prior"]
            task_id = format_task_id(dataset_name, c_value, prior)

            print(f"  Profiling {task_id}...", end=" ", flush=True)

            elapsed = profile_task(dataset_name, c_value, prior, args.batch_size)

            if elapsed is not None:
                task_speeds[task_id] = elapsed
                task_metadata[task_id] = {
                    "dataset": dataset_name,
                    "data_type": data_type,
                    "c_value": c_value,
                    "prior": prior,
                    "time_per_epoch": elapsed
                }
                print(f"{elapsed:.2f}s/epoch")
            else:
                print("FAILED")

        print()

    # Summary
    print("=" * 70)
    print("PROFILING SUMMARY")
    print("=" * 70)
    print()

    # Sort by speed
    sorted_tasks = sorted(task_speeds.items(), key=lambda x: x[1])

    print(f"Total tasks profiled: {len(task_speeds)}")
    print()

    print("Tasks by speed (fastest first):")
    for i, (task_id, speed) in enumerate(sorted_tasks, 1):
        meta = task_metadata[task_id]
        print(f"  {i:2d}. {task_id:30s}  {speed:6.2f}s/epoch  ({meta['data_type']})")

    print()

    # Save results
    output_data = {
        "task_speeds": task_speeds,
        "task_metadata": task_metadata,
        "sorted_task_ids": [task_id for task_id, _ in sorted_tasks],
        "profiling_config": {
            "batch_size": args.batch_size,
            "seed": args.seed,
            "device": "auto (MPS → CUDA → CPU)"
        }
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {args.output}")
    print()

    # Data type distribution
    print("Data type distribution:")
    data_type_counts = {}
    for meta in task_metadata.values():
        dt = meta["data_type"]
        data_type_counts[dt] = data_type_counts.get(dt, 0) + 1

    for data_type, count in sorted(data_type_counts.items()):
        print(f"  {data_type:10s}: {count} tasks")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
