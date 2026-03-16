#!/usr/bin/env python3
"""Gradient Matching Checkpoint Pool.

Maintains a pool of ~100 model checkpoints at various training stages.
Between meta-iterations:
- 90% of checkpoints persist and advance one training step
- 10% are replaced with fresh random initializations
- Creates natural distribution across the training trajectory
"""

import random
import numpy as np
from typing import List, Dict, Any
import torch
import torch.nn as nn

from models.simple_mlp import SimpleMLP


def generate_random_task_config(config: dict) -> dict:
    """Generate a random task configuration from distribution ranges."""
    mean_separations = config.get('mean_separations', [2.0, 2.5, 3.0, 3.5])
    stds = config.get('stds', [0.8, 1.0])
    labeling_freqs = config.get('labeling_freqs', [0.3])
    priors = config.get('priors', [0.5])

    return {
        'num_dimensions': config.get('num_dimensions', 2),
        'mean_separation': float(np.random.choice(mean_separations)),
        'std': float(np.random.choice(stds)),
        'prior': float(np.random.choice(priors)),
        'labeling_freq': float(np.random.choice(labeling_freqs)),
        'num_samples': config.get('num_samples_per_task', 1000),
        'mode': 'pu',
        'negative_labeling_freq': 0.3,
        'seed': np.random.randint(0, 1000000),
    }


class GradientMatchingCheckpointPool:
    """Pool of model checkpoints at various training stages.

    Maintains ~100 checkpoints that persist and advance through training.
    Between meta-iterations:
    - Sample batch of checkpoints
    - Each checkpoint advances one training step
    - 90% of checkpoints persist with updated state
    - 10% are replaced with fresh random initializations

    This creates a natural curriculum distribution from early to late training.
    """

    def __init__(
        self,
        config: dict,
        pool_size: int = 100,
        input_dim: int = 2,
        hidden_dims: List[int] = None,
        inner_lr: float = 0.01,
        inner_momentum: float = 0.9,
    ):
        """Initialize checkpoint pool.

        Args:
            config: Task generation configuration
            pool_size: Number of checkpoints to maintain
            input_dim: Model input dimension
            hidden_dims: Model hidden layer dimensions
            inner_lr: Learning rate for inner optimization
            inner_momentum: Momentum for inner optimizer
        """
        self.pool_size = pool_size
        self.config = config
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [32, 32]
        self.inner_lr = inner_lr
        self.inner_momentum = inner_momentum

        self.checkpoints: List[Dict[str, Any]] = []
        self._next_task_id = 0

    def initialize_pool(self, device: str):
        """Create pool of random checkpoints at step 0.

        Args:
            device: Device to create models on ('cpu', 'cuda', 'mps')
        """
        print(f"Initializing checkpoint pool with {self.pool_size} checkpoints...")

        # Available training objectives
        objectives = ['oracle_bce', 'pudra', 'vpu', 'naive']

        for i in range(self.pool_size):
            # Generate random task configuration
            task_config = generate_random_task_config(self.config)

            # Randomly assign training objective
            objective = random.choice(objectives)

            # Create model with random initialization
            model = SimpleMLP(
                input_dim=self.input_dim,
                hidden_dims=self.hidden_dims,
            ).to(device)

            # Create optimizer (SGD with momentum)
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.inner_lr,
                momentum=self.inner_momentum,
            )

            # Create checkpoint
            checkpoint = {
                'task_id': f"task_{self._next_task_id}",
                'task_config': task_config,
                'objective': objective,  # NEW: training objective for this checkpoint
                'model_state': {k: v.cpu().clone() for k, v in model.state_dict().items()},
                'optimizer_state': {
                    k: v.cpu().clone() if torch.is_tensor(v) else v
                    for k, v in optimizer.state_dict().items()
                },
                'step_count': 0,
                'training_history': [],
                'last_updated_iteration': -1,
            }

            self.checkpoints.append(checkpoint)
            self._next_task_id += 1

        # Print objective distribution
        obj_counts = {obj: sum(1 for c in self.checkpoints if c['objective'] == obj) for obj in objectives}
        print(f"✓ Pool initialized with {len(self.checkpoints)} checkpoints")
        print(f"  Objective distribution: {obj_counts}")

        print(f"✓ Pool initialized with {len(self.checkpoints)} checkpoints")

    def sample_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Randomly sample checkpoints from pool.

        Args:
            batch_size: Number of checkpoints to sample

        Returns:
            List of checkpoint dictionaries
        """
        if batch_size > len(self.checkpoints):
            raise ValueError(
                f"Cannot sample {batch_size} checkpoints from pool of size {len(self.checkpoints)}"
            )

        return random.sample(self.checkpoints, batch_size)

    def update_pool(
        self,
        updated_checkpoints: List[Dict[str, Any]],
        num_to_refresh: int = 8,
        current_iteration: int = 0,
        device: str = 'cpu',
    ):
        """Update pool with new checkpoint states.

        Args:
            updated_checkpoints: Checkpoints that were used this iteration
            num_to_refresh: Number of random checkpoints to replace with fresh init (default 8)
            current_iteration: Current meta-iteration number
            device: Device for creating new checkpoints
        """
        # 1. Replace updated checkpoints in pool
        updated_task_ids = {ckpt['task_id'] for ckpt in updated_checkpoints}

        for i, checkpoint in enumerate(self.checkpoints):
            if checkpoint['task_id'] in updated_task_ids:
                # Find corresponding updated checkpoint
                updated_ckpt = next(
                    c for c in updated_checkpoints
                    if c['task_id'] == checkpoint['task_id']
                )
                # Replace with updated version
                self.checkpoints[i] = updated_ckpt

        # 2. Refresh N random checkpoints with fresh random initializations
        if num_to_refresh > 0 and num_to_refresh <= len(self.checkpoints):
            # Randomly select checkpoints to replace
            replace_indices = random.sample(range(len(self.checkpoints)), num_to_refresh)

            # Replace with fresh random initializations
            objectives = ['oracle_bce', 'pudra', 'vpu', 'naive']

            for idx in replace_indices:
                task_config = generate_random_task_config(self.config)

                # Randomly assign training objective
                objective = random.choice(objectives)

                model = SimpleMLP(
                    input_dim=self.input_dim,
                    hidden_dims=self.hidden_dims,
                ).to(device)

                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=self.inner_lr,
                    momentum=self.inner_momentum,
                )

                # Create fresh checkpoint
                fresh_checkpoint = {
                    'task_id': f"task_{self._next_task_id}",
                    'task_config': task_config,
                    'objective': objective,  # NEW: training objective for this checkpoint
                    'model_state': {k: v.cpu().clone() for k, v in model.state_dict().items()},
                    'optimizer_state': {
                        k: v.cpu().clone() if torch.is_tensor(v) else v
                        for k, v in optimizer.state_dict().items()
                    },
                    'step_count': 0,
                    'training_history': [],
                    'last_updated_iteration': current_iteration,
                }

                self.checkpoints[idx] = fresh_checkpoint
                self._next_task_id += 1

    def get_statistics(self) -> Dict[str, float]:
        """Compute pool statistics for logging.

        Returns:
            Dictionary with statistics about checkpoint distribution
        """
        if not self.checkpoints:
            return {
                'min_steps': 0,
                'max_steps': 0,
                'mean_steps': 0.0,
                'std_steps': 0.0,
                'median_steps': 0.0,
            }

        step_counts = [ckpt['step_count'] for ckpt in self.checkpoints]

        return {
            'min_steps': min(step_counts),
            'max_steps': max(step_counts),
            'mean_steps': np.mean(step_counts),
            'std_steps': np.std(step_counts),
            'median_steps': np.median(step_counts),
        }

    def get_step_distribution(self, num_bins: int = 10) -> Dict[str, List]:
        """Get histogram of step counts.

        Args:
            num_bins: Number of bins for histogram

        Returns:
            Dictionary with 'bins' and 'counts'
        """
        step_counts = [ckpt['step_count'] for ckpt in self.checkpoints]

        if not step_counts:
            return {'bins': [], 'counts': []}

        counts, bin_edges = np.histogram(step_counts, bins=num_bins)

        return {
            'bins': bin_edges.tolist(),
            'counts': counts.tolist(),
        }

    def save_pool(self, filepath: str):
        """Save entire pool to disk.

        Args:
            filepath: Path to save checkpoint pool
        """
        torch.save({
            'checkpoints': self.checkpoints,
            'config': self.config,
            'pool_size': self.pool_size,
            'next_task_id': self._next_task_id,
        }, filepath)

    def load_pool(self, filepath: str):
        """Load pool from disk.

        Args:
            filepath: Path to saved checkpoint pool
        """
        data = torch.load(filepath)
        self.checkpoints = data['checkpoints']
        self.config = data['config']
        self.pool_size = data['pool_size']
        self._next_task_id = data['next_task_id']

    def __len__(self) -> int:
        """Return number of checkpoints in pool."""
        return len(self.checkpoints)

    def __repr__(self) -> str:
        """String representation of pool."""
        stats = self.get_statistics()
        return (
            f"GradientMatchingCheckpointPool(\n"
            f"  pool_size={len(self.checkpoints)},\n"
            f"  step_count_range=[{stats['min_steps']}, {stats['max_steps']}],\n"
            f"  mean_steps={stats['mean_steps']:.1f},\n"
            f"  std_steps={stats['std_steps']:.1f}\n"
            f")"
        )
