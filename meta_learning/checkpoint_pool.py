"""Checkpoint pool management for multi-task meta-learning.

This module manages a pool of pre-trained model checkpoints from multiple PU tasks.
Checkpoints are sampled during meta-training to optimize the learned loss function.
"""

import os
import pickle
import random
from typing import List, Dict, Any, Tuple
from pathlib import Path
from copy import deepcopy

import torch
import yaml


# Architecture mapping: dataset name -> model architecture
DATASET_TO_ARCH = {
    'mnist': 'LeNet',
    'fashionmnist': 'LeNet',
    'cifar10': 'CNN_CIFAR10',
    'alzheimermri': 'CNN_Medical',
    'mushrooms': 'MLP_Tabular',
    'spambase': 'MLP_Tabular',
    'connect4': 'MLP_Tabular',
    '20news': 'MLP_Text',
    'imdb': 'MLP_Text'
}


class CheckpointPool:
    """Manages checkpoint pool for multi-task meta-learning.

    The pool contains checkpoints from multiple tasks at various training stages.
    Initially populated from PN naive training, then gradually refreshed with
    checkpoints trained using the learned loss (curriculum strategy).

    Attributes:
        checkpoint_dir: Directory for storing checkpoints
        checkpoints: List of checkpoint dictionaries

    Each checkpoint contains:
        - task_id: str, e.g., "mnist_c0.3_prior0.5_seed42"
        - epoch: int, training epoch when checkpoint was saved
        - seed: int, random seed used for training
        - model_state: OrderedDict, model state_dict
        - dataset: str, dataset name
        - c_value: float, label noise level
        - prior: float, class prior
        - source: str, 'pn_naive' or 'learned_loss'
        - created_at: int, meta-iteration when created
    """

    def __init__(self, checkpoint_dir: str):
        """Initialize checkpoint pool.

        Args:
            checkpoint_dir: Directory for storing checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints = []

    def add_checkpoint(self, checkpoint: Dict[str, Any]):
        """Add a checkpoint to the pool.

        Args:
            checkpoint: Checkpoint dictionary
        """
        self.checkpoints.append(checkpoint)

    def sample_meta_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample random checkpoints for meta-batch.

        Args:
            batch_size: Number of checkpoints to sample

        Returns:
            List of sampled checkpoint dictionaries
        """
        if len(self.checkpoints) < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} checkpoints from pool of size {len(self.checkpoints)}"
            )

        indices = random.sample(range(len(self.checkpoints)), batch_size)
        return [self.checkpoints[i] for i in indices]

    def refresh_pool(
        self,
        learned_loss,
        percent: float,
        current_iteration: int,
        trainer_factory,
        device
    ):
        """Replace random N% of checkpoints with ones trained using learned_loss.

        This implements the curriculum strategy: gradually replace PN naive
        checkpoints with new ones adapted to the current learned loss. Using
        random selection creates natural decay toward learned_loss distribution.

        Args:
            learned_loss: Current learned loss module
            percent: Percentage of checkpoints to replace (e.g., 10 for 10%)
            current_iteration: Current meta-iteration number
            trainer_factory: Function to create trainer given task config and loss
            device: Device to use for training
        """
        num_to_replace = max(1, int(len(self.checkpoints) * percent / 100))

        # Randomly select checkpoints to replace (natural decay)
        import random
        to_replace = random.sample(self.checkpoints, num_to_replace)

        print(f"\nRefreshing checkpoint pool: replacing {num_to_replace} oldest checkpoints")
        print(f"  Iteration: {current_iteration}")
        print()

        for i, old_checkpoint in enumerate(to_replace, 1):
            task_id = old_checkpoint['task_id']
            target_epoch = old_checkpoint['epoch']

            print(f"  [{i}/{num_to_replace}] Training {task_id} to epoch {target_epoch}...", end=" ", flush=True)

            # Train new checkpoint using current learned loss
            try:
                new_model_state = trainer_factory(
                    task_id=task_id,
                    target_epoch=target_epoch,
                    loss_fn=learned_loss,
                    device=device
                )

                # Create new checkpoint
                new_checkpoint = {
                    **old_checkpoint,  # Keep task metadata
                    'model_state': new_model_state,
                    'source': 'learned_loss',
                    'created_at': current_iteration
                }

                # Replace in pool
                self.checkpoints.remove(old_checkpoint)
                self.checkpoints.append(new_checkpoint)

                print("Done")

            except Exception as e:
                print(f"Failed: {e}")

        print()

    def save_to_disk(self, filename: str = "checkpoint_pool.pkl"):
        """Save checkpoint pool to disk.

        Args:
            filename: Filename for the pool pickle file
        """
        filepath = self.checkpoint_dir / filename

        # Save pool metadata and checkpoints
        pool_data = {
            'checkpoints': self.checkpoints,
            'pool_size': len(self.checkpoints)
        }

        with open(filepath, 'wb') as f:
            pickle.dump(pool_data, f)

        print(f"Checkpoint pool saved to: {filepath}")

        # Also save individual checkpoint files for inspection
        checkpoints_dir = self.checkpoint_dir / "individual_checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)

        for checkpoint in self.checkpoints:
            # task_id already includes seed if present
            checkpoint_filename = (
                f"{checkpoint['task_id']}_epoch{checkpoint['epoch']}_"
                f"{checkpoint['source']}.pt"
            )
            checkpoint_path = checkpoints_dir / checkpoint_filename

            torch.save(checkpoint['model_state'], checkpoint_path)

    def load_from_disk(self, filename: str = "checkpoint_pool.pkl"):
        """Load checkpoint pool from disk.

        Args:
            filename: Filename of the pool pickle file
        """
        filepath = self.checkpoint_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint pool file not found: {filepath}")

        with open(filepath, 'rb') as f:
            pool_data = pickle.load(f)

        self.checkpoints = pool_data['checkpoints']

        print(f"Loaded {len(self.checkpoints)} checkpoints from: {filepath}")

    def get_architecture_from_dataset(self, dataset: str) -> str:
        """Get architecture name for dataset.

        Args:
            dataset: Dataset name

        Returns:
            Architecture name
        """
        return DATASET_TO_ARCH.get(dataset, 'unknown')

    def get_dataset_from_task_id(self, task_id: str) -> str:
        """Extract dataset name from task_id.

        Args:
            task_id: Task identifier (e.g., "mnist_c0.3_prior0.5_seed42")

        Returns:
            Dataset name (e.g., "mnist")
        """
        # Task ID format: {dataset}_c{c_value}_prior{prior}_seed{seed}
        # or {dataset}_c{c_value}_prior{prior}
        return task_id.split('_c')[0]

    def get_architecture_from_task_id(self, task_id: str) -> str:
        """Get architecture for task.

        Args:
            task_id: Task identifier

        Returns:
            Architecture name
        """
        dataset = self.get_dataset_from_task_id(task_id)
        return self.get_architecture_from_dataset(dataset)

    def filter_by_architecture(self, architecture: str) -> List[Dict]:
        """Get all checkpoints for a specific architecture.

        Args:
            architecture: Architecture name (e.g., 'LeNet', 'MLP_Tabular')

        Returns:
            List of checkpoints with matching architecture
        """
        filtered = []
        for cp in self.checkpoints:
            task_arch = self.get_architecture_from_task_id(cp['task_id'])
            if task_arch == architecture:
                filtered.append(cp)
        return filtered

    def group_by_architecture(self, checkpoints: List[Dict] = None) -> Dict[str, List[Dict]]:
        """Group checkpoints by architecture.

        Args:
            checkpoints: List of checkpoints to group (default: all checkpoints)

        Returns:
            Dictionary mapping architecture -> list of checkpoints
        """
        if checkpoints is None:
            checkpoints = self.checkpoints

        groups = {}
        for cp in checkpoints:
            arch = self.get_architecture_from_task_id(cp['task_id'])
            if arch not in groups:
                groups[arch] = []
            groups[arch].append(cp)

        return groups

    def get_architecture_stats(self) -> Dict[str, int]:
        """Get checkpoint counts by architecture.

        Returns:
            Dictionary mapping architecture name -> count
        """
        groups = self.group_by_architecture()
        return {arch: len(cps) for arch, cps in groups.items()}

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the checkpoint pool.

        Returns:
            Dictionary with pool statistics
        """
        if not self.checkpoints:
            return {"pool_size": 0}

        # Count by source
        source_counts = {}
        for cp in self.checkpoints:
            source = cp['source']
            source_counts[source] = source_counts.get(source, 0) + 1

        # Count by task
        task_counts = {}
        for cp in self.checkpoints:
            task_id = cp['task_id']
            task_counts[task_id] = task_counts.get(task_id, 0) + 1

        # Count by dataset
        dataset_counts = {}
        for cp in self.checkpoints:
            dataset = cp['dataset']
            dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1

        # Count by epoch
        epoch_counts = {}
        for cp in self.checkpoints:
            epoch = cp['epoch']
            epoch_counts[epoch] = epoch_counts.get(epoch, 0) + 1

        # Count by seed
        seed_counts = {}
        for cp in self.checkpoints:
            seed = cp.get('seed', 'unknown')
            seed_counts[seed] = seed_counts.get(seed, 0) + 1

        return {
            "pool_size": len(self.checkpoints),
            "source_counts": source_counts,
            "task_counts": task_counts,
            "dataset_counts": dataset_counts,
            "epoch_counts": epoch_counts,
            "seed_counts": seed_counts,
            "unique_tasks": len(task_counts),
            "unique_datasets": len(dataset_counts),
            "unique_epochs": len(epoch_counts),
            "unique_seeds": len(seed_counts)
        }

    def print_statistics(self):
        """Print checkpoint pool statistics."""
        stats = self.get_statistics()

        print("=" * 70)
        print("CHECKPOINT POOL STATISTICS")
        print("=" * 70)
        print()

        print(f"Total checkpoints: {stats['pool_size']}")
        print(f"Unique tasks: {stats.get('unique_tasks', 0)}")
        print(f"Unique datasets: {stats.get('unique_datasets', 0)}")
        print(f"Unique epochs: {stats.get('unique_epochs', 0)}")
        print(f"Unique seeds: {stats.get('unique_seeds', 0)}")
        print()

        if 'source_counts' in stats:
            print("By source:")
            for source, count in stats['source_counts'].items():
                pct = 100 * count / stats['pool_size']
                print(f"  {source:15s}: {count:3d} ({pct:5.1f}%)")
            print()

        # Add architecture stats
        arch_stats = self.get_architecture_stats()
        if arch_stats:
            print("By architecture:")
            for arch, count in sorted(arch_stats.items()):
                pct = 100 * count / stats['pool_size']
                print(f"  {arch:15s}: {count:3d} ({pct:5.1f}%)")
            print()

        if 'dataset_counts' in stats:
            print("By dataset:")
            for dataset, count in sorted(stats['dataset_counts'].items()):
                print(f"  {dataset:15s}: {count:3d}")
            print()

        if 'epoch_counts' in stats:
            print("By epoch:")
            for epoch, count in sorted(stats['epoch_counts'].items()):
                print(f"  Epoch {epoch:2d}: {count:3d}")
            print()

        if 'seed_counts' in stats:
            print("By seed:")
            for seed, count in sorted(stats['seed_counts'].items()):
                print(f"  Seed {seed}: {count:3d}")
            print()

        print("=" * 70)


def load_task_split(split_file: str) -> Tuple[List[Dict], List[Dict]]:
    """Load training/test split from YAML file.

    Args:
        split_file: Path to task_split.yaml

    Returns:
        (training_tasks, test_tasks) as lists of task dictionaries
    """
    with open(split_file, 'r') as f:
        split_data = yaml.safe_load(f)

    training_tasks = split_data['training_tasks']
    test_tasks = split_data['test_tasks']

    return training_tasks, test_tasks


def format_task_id(dataset: str, c_value: float, prior: float, seed: int = None) -> str:
    """Format task identifier.

    Args:
        dataset: Dataset name
        c_value: Label noise level
        prior: Class prior
        seed: Optional random seed

    Returns:
        Task ID string
    """
    base_id = f"{dataset}_c{c_value:.1f}_prior{prior:.1f}"
    if seed is not None:
        return f"{base_id}_seed{seed}"
    return base_id
