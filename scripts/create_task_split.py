"""Create training/test split from profiling results.

This script loads task profiling results and creates a training/test split that:
1. Prioritizes fast tasks for training
2. Ensures at least 1 task from each data type (image, tabular, text)
3. Maintains diversity in c_values and priors
4. Reserves at least 50% of tasks for testing

Usage:
    python scripts/create_task_split.py --input task_speeds.json --output task_split.yaml
"""

import argparse
import json
import yaml
from collections import defaultdict
from pathlib import Path


def load_profiling_results(input_file: str):
    """Load profiling results from JSON."""
    with open(input_file, 'r') as f:
        data = json.load(f)
    return data


def select_diverse_training_tasks(
    task_metadata: dict,
    sorted_task_ids: list,
    target_size: int = 8,
    min_per_type: int = 1
):
    """Select diverse training tasks ensuring data type coverage.

    Args:
        task_metadata: Metadata for each task
        sorted_task_ids: Task IDs sorted by speed (fastest first)
        target_size: Target number of training tasks
        min_per_type: Minimum tasks per data type

    Returns:
        List of selected training task IDs
    """
    # Group tasks by data type
    tasks_by_type = defaultdict(list)
    for task_id in sorted_task_ids:
        data_type = task_metadata[task_id]["data_type"]
        tasks_by_type[data_type].append(task_id)

    # Ensure we have all data types
    required_types = ["image", "tabular", "text"]
    for dt in required_types:
        if dt not in tasks_by_type or len(tasks_by_type[dt]) < min_per_type:
            raise ValueError(f"Not enough {dt} tasks in profiling results!")

    # Select tasks ensuring coverage
    selected = []

    # Phase 1: Select min_per_type from each data type (prioritize speed)
    for data_type in required_types:
        candidates = tasks_by_type[data_type]
        selected.extend(candidates[:min_per_type])

    # Phase 2: Fill remaining slots with fastest tasks not yet selected
    remaining_slots = target_size - len(selected)
    if remaining_slots > 0:
        for task_id in sorted_task_ids:
            if task_id not in selected:
                selected.append(task_id)
                remaining_slots -= 1
                if remaining_slots == 0:
                    break

    # Phase 3: Ensure diversity in c_values and priors
    selected_meta = [task_metadata[tid] for tid in selected]
    c_values = set(meta["c_value"] for meta in selected_meta)
    priors = set(meta["prior"] for meta in selected_meta)

    print(f"Selected {len(selected)} training tasks:")
    print(f"  Data type coverage: {sum(1 for m in selected_meta if m['data_type']=='image')} image, "
          f"{sum(1 for m in selected_meta if m['data_type']=='tabular')} tabular, "
          f"{sum(1 for m in selected_meta if m['data_type']=='text')} text")
    print(f"  C-value diversity: {len(c_values)} unique values")
    print(f"  Prior diversity: {len(priors)} unique values")

    return selected


def create_split(profiling_file: str, output_file: str, target_train_size: int = 8):
    """Create training/test split from profiling results.

    Args:
        profiling_file: Path to task_speeds.json
        output_file: Path to output task_split.yaml
        target_train_size: Target number of training tasks
    """
    # Load profiling results
    data = load_profiling_results(profiling_file)
    task_speeds = data["task_speeds"]
    task_metadata = data["task_metadata"]
    sorted_task_ids = data["sorted_task_ids"]

    print("=" * 70)
    print("CREATING TRAINING/TEST SPLIT")
    print("=" * 70)
    print()

    print(f"Total tasks: {len(sorted_task_ids)}")
    print(f"Target training size: {target_train_size}")
    print()

    # Select training tasks
    training_task_ids = select_diverse_training_tasks(
        task_metadata,
        sorted_task_ids,
        target_size=target_train_size,
        min_per_type=1
    )

    # Remaining tasks are for testing
    test_task_ids = [tid for tid in sorted_task_ids if tid not in training_task_ids]

    print()
    print(f"Test tasks: {len(test_task_ids)}")
    print(f"Held-out percentage: {100 * len(test_task_ids) / len(sorted_task_ids):.1f}%")
    print()

    # Convert to structured format
    training_tasks = []
    for task_id in training_task_ids:
        meta = task_metadata[task_id]
        training_tasks.append({
            "dataset": meta["dataset"],
            "c_value": meta["c_value"],
            "prior": meta["prior"],
            "speed": meta["time_per_epoch"],
            "data_type": meta["data_type"]
        })

    test_tasks = []
    for task_id in test_task_ids:
        meta = task_metadata[task_id]
        test_tasks.append({
            "dataset": meta["dataset"],
            "c_value": meta["c_value"],
            "prior": meta["prior"],
            "speed": meta["time_per_epoch"],
            "data_type": meta["data_type"]
        })

    # Create output YAML
    split_data = {
        "meta": {
            "total_tasks": len(sorted_task_ids),
            "training_tasks": len(training_task_ids),
            "test_tasks": len(test_task_ids),
            "held_out_percentage": 100 * len(test_task_ids) / len(sorted_task_ids),
            "profiling_file": profiling_file
        },
        "training_tasks": training_tasks,
        "test_tasks": test_tasks
    }

    # Save to YAML
    with open(output_file, 'w') as f:
        yaml.dump(split_data, f, default_flow_style=False, sort_keys=False)

    print(f"Split saved to: {output_file}")
    print()

    # Print detailed breakdown
    print("=" * 70)
    print("TRAINING TASKS DETAIL")
    print("=" * 70)
    print()

    for i, task in enumerate(training_tasks, 1):
        print(f"{i:2d}. {task['dataset']:15s} c={task['c_value']:.1f} prior={task['prior']:.1f}  "
              f"{task['speed']:6.2f}s/epoch  ({task['data_type']})")

    print()
    print("=" * 70)
    print("TEST TASKS DETAIL")
    print("=" * 70)
    print()

    for i, task in enumerate(test_tasks, 1):
        print(f"{i:2d}. {task['dataset']:15s} c={task['c_value']:.1f} prior={task['prior']:.1f}  "
              f"{task['speed']:6.2f}s/epoch  ({task['data_type']})")

    print()

    # Verification
    print("=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    print()

    # Check data type coverage
    training_types = set(task["data_type"] for task in training_tasks)
    required_types = {"image", "tabular", "text"}

    if training_types >= required_types:
        print("✓ All data types covered in training set")
    else:
        missing = required_types - training_types
        print(f"✗ WARNING: Missing data types in training: {missing}")

    # Check held-out percentage
    held_out_pct = 100 * len(test_task_ids) / len(sorted_task_ids)
    if held_out_pct >= 50:
        print(f"✓ Held-out percentage ({held_out_pct:.1f}%) >= 50%")
    else:
        print(f"✗ WARNING: Held-out percentage ({held_out_pct:.1f}%) < 50%")

    # Check diversity
    train_c_values = set(task["c_value"] for task in training_tasks)
    train_priors = set(task["prior"] for task in training_tasks)

    print(f"✓ Training set has {len(train_c_values)} unique c_values")
    print(f"✓ Training set has {len(train_priors)} unique priors")

    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Create training/test split from profiling")
    parser.add_argument("--input", type=str, default="task_speeds.json",
                        help="Input JSON file with profiling results")
    parser.add_argument("--output", type=str, default="task_split.yaml",
                        help="Output YAML file for split")
    parser.add_argument("--train-size", type=int, default=8,
                        help="Target number of training tasks")
    args = parser.parse_args()

    create_split(args.input, args.output, args.train_size)


if __name__ == "__main__":
    main()
