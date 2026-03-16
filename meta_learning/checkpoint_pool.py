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

    def __init__(self, checkpoint_dir: str, lazy_load: bool = True, meta_val_split: float = 0.2):
        """Initialize checkpoint pool.

        Args:
            checkpoint_dir: Directory for storing checkpoints
            lazy_load: If True, load only metadata and defer loading model states
            meta_val_split: Fraction of checkpoints to use for meta-validation (default: 0.2)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints = []
        self.lazy_load = lazy_load
        self._checkpoint_metadata = []  # Lightweight metadata without model states
        self.meta_val_split = meta_val_split
        self._train_indices = []  # Indices for meta-training
        self._val_indices = []    # Indices for meta-validation

    def add_checkpoint(self, checkpoint: Dict[str, Any]):
        """Add a checkpoint to the pool.

        Args:
            checkpoint: Checkpoint dictionary
        """
        self.checkpoints.append(checkpoint)

    def _load_checkpoint_state(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Load model state for a checkpoint from disk.

        Args:
            metadata: Checkpoint metadata

        Returns:
            Full checkpoint with model_state loaded
        """
        checkpoint_filename = (
            f"{metadata['task_id']}_epoch{metadata['epoch']}_"
            f"{metadata['source']}.pt"
        )
        checkpoint_path = self.checkpoint_dir / "individual_checkpoints" / checkpoint_filename

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        model_state = torch.load(checkpoint_path, map_location='cpu')

        # Return full checkpoint
        return {**metadata, 'model_state': model_state}

    def _split_train_val(self):
        """Split checkpoints into train/val sets based on meta_val_split."""
        if self.lazy_load:
            num_total = len(self._checkpoint_metadata)
        else:
            num_total = len(self.checkpoints)

        num_val = int(num_total * self.meta_val_split)

        # Create shuffled indices
        indices = list(range(num_total))
        random.seed(42)  # Fixed seed for reproducibility
        random.shuffle(indices)

        self._val_indices = indices[:num_val]
        self._train_indices = indices[num_val:]

        print(f"Split checkpoint pool: {len(self._train_indices)} train, {len(self._val_indices)} val")

    def sample_meta_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample random checkpoints for meta-batch (backward compatibility).

        Uses training set only if train/val split is active.

        Args:
            batch_size: Number of checkpoints to sample

        Returns:
            List of sampled checkpoint dictionaries
        """
        return self.sample_meta_train_batch(batch_size)

    def sample_meta_train_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample random checkpoints from training set.

        Args:
            batch_size: Number of checkpoints to sample

        Returns:
            List of sampled checkpoint dictionaries
        """
        if self.lazy_load:
            # Sample from metadata and load states on-demand
            if len(self._train_indices) < batch_size:
                raise ValueError(
                    f"Cannot sample {batch_size} checkpoints from training set of size {len(self._train_indices)}"
                )

            sample_indices = random.sample(self._train_indices, batch_size)
            sampled = []
            for i in sample_indices:
                metadata = self._checkpoint_metadata[i]
                checkpoint = self._load_checkpoint_state(metadata)
                sampled.append(checkpoint)

            return sampled
        else:
            # Original behavior: sample from fully loaded checkpoints
            if len(self._train_indices) < batch_size:
                raise ValueError(
                    f"Cannot sample {batch_size} checkpoints from training set of size {len(self._train_indices)}"
                )

            sample_indices = random.sample(self._train_indices, batch_size)
            return [self.checkpoints[i] for i in sample_indices]

    def get_fixed_val_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Get a fixed validation batch (deterministic, same each time).

        Uses first N checkpoints from validation set for reproducible validation.

        Args:
            batch_size: Number of checkpoints to return

        Returns:
            List of checkpoint dictionaries (always the same for given batch_size)
        """
        if len(self._val_indices) < batch_size:
            raise ValueError(
                f"Cannot get {batch_size} checkpoints from validation set of size {len(self._val_indices)}"
            )

        # Always return first batch_size checkpoints from validation set
        sample_indices = self._val_indices[:batch_size]

        if self.lazy_load:
            sampled = []
            for i in sample_indices:
                metadata = self._checkpoint_metadata[i]
                checkpoint = self._load_checkpoint_state(metadata)
                sampled.append(checkpoint)
            return sampled
        else:
            return [self.checkpoints[i] for i in sample_indices]

    def sample_meta_val_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample random checkpoints from validation set.

        Args:
            batch_size: Number of checkpoints to sample

        Returns:
            List of sampled checkpoint dictionaries
        """
        if self.lazy_load:
            # Sample from metadata and load states on-demand
            if len(self._val_indices) < batch_size:
                raise ValueError(
                    f"Cannot sample {batch_size} checkpoints from validation set of size {len(self._val_indices)}"
                )

            sample_indices = random.sample(self._val_indices, batch_size)
            sampled = []
            for i in sample_indices:
                metadata = self._checkpoint_metadata[i]
                checkpoint = self._load_checkpoint_state(metadata)
                sampled.append(checkpoint)

            return sampled
        else:
            # Original behavior: sample from fully loaded checkpoints
            if len(self._val_indices) < batch_size:
                raise ValueError(
                    f"Cannot sample {batch_size} checkpoints from validation set of size {len(self._val_indices)}"
                )

            sample_indices = random.sample(self._val_indices, batch_size)
            return [self.checkpoints[i] for i in sample_indices]

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
        # Use appropriate checkpoint list based on lazy_load mode
        checkpoint_list = self._checkpoint_metadata if self.lazy_load else self.checkpoints
        num_to_replace = max(1, int(len(checkpoint_list) * percent / 100))

        # Randomly select checkpoints to replace (natural decay)
        import random
        to_replace_indices = random.sample(range(len(checkpoint_list)), num_to_replace)
        to_replace = [checkpoint_list[i] for i in to_replace_indices]

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

                # Create new checkpoint metadata
                new_metadata = {
                    k: v for k, v in old_checkpoint.items() if k != 'model_state'
                }
                new_metadata['source'] = 'learned_loss'
                new_metadata['created_at'] = current_iteration

                if self.lazy_load:
                    # Save model state to disk
                    checkpoint_filename = (
                        f"{task_id}_epoch{target_epoch}_learned_loss.pt"
                    )
                    checkpoint_path = self.checkpoint_dir / "individual_checkpoints" / checkpoint_filename
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(new_model_state, checkpoint_path)

                    # Find and replace in metadata list
                    # Match by task_id, epoch, and source (to handle multiple seeds)
                    old_source = old_checkpoint.get('source', 'pn_naive')
                    old_seed = old_checkpoint.get('seed', 42)
                    for idx, meta in enumerate(self._checkpoint_metadata):
                        if (meta['task_id'] == task_id and
                            meta['epoch'] == target_epoch and
                            meta.get('source', 'pn_naive') == old_source and
                            meta.get('seed', 42) == old_seed):
                            self._checkpoint_metadata[idx] = new_metadata
                            break
                else:
                    # Full checkpoint with model state
                    new_checkpoint = {
                        **new_metadata,
                        'model_state': new_model_state
                    }
                    # Replace in checkpoints list
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

        # Save full pool data (for backward compatibility)
        pool_data = {
            'checkpoints': self.checkpoints,
            'pool_size': len(self.checkpoints)
        }

        with open(filepath, 'wb') as f:
            pickle.dump(pool_data, f)

        print(f"Checkpoint pool saved to: {filepath}")

        # Save lightweight metadata-only version (for lazy loading)
        metadata_only = []
        for cp in self.checkpoints:
            meta = {k: v for k, v in cp.items() if k != 'model_state'}
            metadata_only.append(meta)

        metadata_path = self.checkpoint_dir / "checkpoint_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump({'metadata': metadata_only, 'pool_size': len(metadata_only)}, f)

        print(f"Checkpoint metadata saved to: {metadata_path}")

        # Also save individual checkpoint files for lazy loading
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
        if self.lazy_load:
            # Load only metadata (fast: ~50KB vs ~900MB)
            metadata_path = self.checkpoint_dir / "checkpoint_metadata.pkl"

            if metadata_path.exists():
                print(f"Using lazy loading (metadata only)...")
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)

                self._checkpoint_metadata = metadata['metadata']
                print(f"Loaded metadata for {len(self._checkpoint_metadata)} checkpoints from: {metadata_path}")
                print("Model states will be loaded on-demand when sampled")

                # Split into train/val sets
                self._split_train_val()
                return
            else:
                print(f"Metadata file not found at {metadata_path}, falling back to full load")
                self.lazy_load = False

        # Full load (backward compatibility or if metadata doesn't exist)
        filepath = self.checkpoint_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint pool file not found: {filepath}")

        print(f"Loading full checkpoint pool (this may take a few minutes)...")
        with open(filepath, 'rb') as f:
            pool_data = pickle.load(f)

        self.checkpoints = pool_data['checkpoints']

        print(f"Loaded {len(self.checkpoints)} checkpoints from: {filepath}")

        # Split into train/val sets
        self._split_train_val()

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
        # Use metadata if lazy loading, otherwise use full checkpoints
        data = self._checkpoint_metadata if self.lazy_load else self.checkpoints

        filtered = []
        for cp in data:
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
            # Use metadata if lazy loading, otherwise use full checkpoints
            checkpoints = self._checkpoint_metadata if self.lazy_load else self.checkpoints

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
        # Use metadata if lazy loading, otherwise use full checkpoints
        data = self._checkpoint_metadata if self.lazy_load else self.checkpoints

        if not data:
            return {"pool_size": 0}

        # Count by source
        source_counts = {}
        for cp in data:
            source = cp['source']
            source_counts[source] = source_counts.get(source, 0) + 1

        # Count by task
        task_counts = {}
        for cp in data:
            task_id = cp['task_id']
            task_counts[task_id] = task_counts.get(task_id, 0) + 1

        # Count by dataset
        dataset_counts = {}
        for cp in data:
            dataset = cp['dataset']
            dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1

        # Count by epoch
        epoch_counts = {}
        for cp in data:
            epoch = cp['epoch']
            epoch_counts[epoch] = epoch_counts.get(epoch, 0) + 1

        # Count by seed
        seed_counts = {}
        for cp in data:
            seed = cp.get('seed', 'unknown')
            seed_counts[seed] = seed_counts.get(seed, 0) + 1

        return {
            "pool_size": len(data),
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
