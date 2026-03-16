"""Checkpoint pool management for toy meta-learning."""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from torch.utils.data import DataLoader
import copy
import hashlib
import json
import pickle

from tasks.gaussian_task import GaussianBlobTask
from models.simple_mlp import SimpleMLP
from loss.baseline_losses import PUDRaNaiveLoss, VPUNoMixUpLoss


class CheckpointPool:
    """Manages a pool of model checkpoints for meta-learning.

    Each checkpoint contains:
    - Task configuration (mean_sep, std, prior, labeling_freq)
    - Model weights at a specific training epoch
    - Task dataloaders
    """

    def __init__(
        self,
        config: Dict,
        save_dir: str = './toy_checkpoints',
    ):
        """Initialize checkpoint pool.

        Args:
            config: Configuration dictionary
            save_dir: Directory to save/load checkpoints
        """
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoints = []
        self.checkpoint_metadata = []

    def _get_pool_cache_path(self) -> Path:
        """Generate unique cache path based on config hash.

        Returns:
            Path to cache file
        """
        # Create deterministic hash from relevant config parameters
        cache_key = {
            'mean_separations': self.config.get('mean_separations', [2.0]),
            'stds': self.config.get('stds', [1.0]),
            'priors': self.config.get('priors', [0.5]),
            'labeling_freqs': self.config.get('labeling_freqs', [0.3]),
            'checkpoint_seeds': self.config.get('checkpoint_seeds', [42]),
            'checkpoint_epochs': self.config.get('checkpoint_epochs', [1, 5, 10]),
            'training_methods': self.config.get('training_methods', ['oracle']),
            'num_dimensions': self.config.get('num_dimensions', 2),
            'num_samples_per_task': self.config.get('num_samples_per_task', 1000),
            'model_hidden_dims': self.config.get('model_hidden_dims', [32, 32]),
        }

        # Create hash
        cache_str = json.dumps(cache_key, sort_keys=True)
        cache_hash = hashlib.md5(cache_str.encode()).hexdigest()[:8]

        return self.save_dir / f"checkpoint_pool_{cache_hash}.pkl"

    def save_checkpoint_pool(self) -> Path:
        """Save checkpoint pool to disk.

        Returns:
            Path where pool was saved
        """
        cache_path = self._get_pool_cache_path()

        pool_data = {
            'checkpoints': self.checkpoints,
            'checkpoint_metadata': self.checkpoint_metadata,
            'config': self.config,
        }

        with open(cache_path, 'wb') as f:
            pickle.dump(pool_data, f)

        print(f"Saved checkpoint pool to: {cache_path}")
        return cache_path

    def load_checkpoint_pool(self) -> bool:
        """Load checkpoint pool from disk if available.

        Returns:
            True if loaded successfully, False if cache doesn't exist
        """
        cache_path = self._get_pool_cache_path()

        if not cache_path.exists():
            return False

        try:
            with open(cache_path, 'rb') as f:
                pool_data = pickle.load(f)

            self.checkpoints = pool_data['checkpoints']
            self.checkpoint_metadata = pool_data['checkpoint_metadata']

            print("="*70)
            print("LOADED CACHED CHECKPOINT POOL")
            print("="*70)
            print(f"Loaded from: {cache_path}")
            print(f"Total checkpoints: {len(self.checkpoints)}")
            print("="*70)
            print()

            return True
        except Exception as e:
            print(f"Warning: Failed to load cached pool: {e}")
            return False

    def generate_task_grid(self) -> List[Dict]:
        """Generate grid of task configurations.

        Returns:
            List of task configuration dictionaries
        """
        tasks = []

        mean_seps = self.config.get('mean_separations', [2.0])
        stds = self.config.get('stds', [1.0])
        priors = self.config.get('priors', [0.5])
        labeling_freqs = self.config.get('labeling_freqs', [0.3])
        seeds = self.config.get('checkpoint_seeds', [42])
        mode = self.config.get('mode', 'pu')
        negative_labeling_freq = self.config.get('negative_labeling_freq', 0.3)
        training_methods = self.config.get('training_methods', ['oracle'])  # 'oracle' or 'naive'

        # Generate all combinations
        for mean_sep in mean_seps:
            for std in stds:
                for prior in priors:
                    for labeling_freq in labeling_freqs:
                        for seed in seeds:
                            for training_method in training_methods:
                                task_config = {
                                    'num_dimensions': self.config.get('num_dimensions', 2),
                                    'mean_separation': mean_sep,
                                    'std': std,
                                    'prior': prior,
                                    'labeling_freq': labeling_freq,
                                    'num_samples': self.config.get('num_samples_per_task', 1000),
                                    'seed': seed,
                                    'mode': mode,
                                    'negative_labeling_freq': negative_labeling_freq,
                                    'training_method': training_method,  # oracle or naive
                                }
                                tasks.append(task_config)

        return tasks

    def create_checkpoint_pool(self, device: str = 'cpu'):
        """Create checkpoint pool by training models on tasks.

        Args:
            device: Device to train on
        """
        print("="* 70)
        print("CREATING CHECKPOINT POOL")
        print("=" * 70)

        task_configs = self.generate_task_grid()
        checkpoint_epochs = self.config.get('checkpoint_epochs', [1, 5, 10])

        print(f"Task configurations: {len(task_configs)}")
        print(f"Checkpoint epochs: {checkpoint_epochs}")
        print(f"Total checkpoints: {len(task_configs) * len(checkpoint_epochs)}")

        checkpoint_id = 0

        for task_idx, task_config in enumerate(task_configs):
            print(f"\n[{task_idx + 1}/{len(task_configs)}] Task: sep={task_config['mean_separation']:.1f}, "
                  f"std={task_config['std']:.1f}, prior={task_config['prior']:.1f}, "
                  f"c={task_config['labeling_freq']:.1f}, seed={task_config['seed']}, "
                  f"method={task_config['training_method']}")

            # Create task (exclude training_method from task creation)
            task_params = {k: v for k, v in task_config.items() if k != 'training_method'}
            task = GaussianBlobTask(**task_params)

            # Get dataloaders
            dataloaders = task.get_dataloaders(
                batch_size=self.config.get('inner_batch_size', 64),
                num_train=task_config['num_samples'],
                num_val=task_config['num_samples'] // 2,
                num_test=task_config['num_samples'] // 2,
            )

            # Create model
            model = SimpleMLP(
                input_dim=task_config['num_dimensions'],
                hidden_dims=self.config.get('model_hidden_dims', [32, 32]),
                activation=self.config.get('model_activation', 'relu'),
            ).to(device)

            # Create loss based on training method
            training_method = task_config.get('training_method', 'oracle')
            if training_method in ['oracle', 'naive']:
                # BCE for oracle and naive methods
                criterion = torch.nn.BCEWithLogitsLoss()
            elif training_method == 'pudra_naive':
                # PUDRa-naive baseline
                criterion = PUDRaNaiveLoss()
            elif training_method == 'vpu_nomixup':
                # VPU-NoMixUp baseline
                criterion = VPUNoMixUpLoss()
            else:
                raise ValueError(f"Unknown training method: {training_method}")

            # Create optimizer
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.config.get('checkpoint_train_lr', 0.001),
            )

            # Train and save checkpoints at specified epochs
            current_epoch = 0
            for target_epoch in sorted(checkpoint_epochs):
                epochs_to_train = target_epoch - current_epoch

                # Train for epochs_to_train epochs
                for epoch in range(epochs_to_train):
                    model.train()
                    for batch in dataloaders['train']:
                        # Unpack batch: (X, y_true, y_pu)
                        batch_x = batch[0].to(device)
                        y_true = batch[1].to(device)
                        y_pu = batch[2].to(device)

                        # Use appropriate labels and loss based on training_method
                        training_method = task_config.get('training_method', 'oracle')

                        optimizer.zero_grad()
                        outputs = model(batch_x).squeeze(-1)  # Only squeeze last dim

                        if training_method == 'oracle':
                            # Oracle: Train with ground truth PN labels using BCE
                            batch_y = y_true
                            loss = criterion(outputs, batch_y)
                        elif training_method == 'naive':
                            # Naive: Train with PU labels (only some positives labeled)
                            batch_y = y_pu
                            # For naive PU, filter out unlabeled samples (y_pu == -1)
                            labeled_mask = batch_y != -1
                            if labeled_mask.sum() == 0:
                                continue
                            outputs_filtered = outputs[labeled_mask]
                            batch_y_filtered = batch_y[labeled_mask]
                            loss = criterion(outputs_filtered, batch_y_filtered)
                        elif training_method in ['pudra_naive', 'vpu_nomixup']:
                            # PU baseline losses: Use all samples with PU labels
                            # PU labels: 1 for positive, -1 for unlabeled
                            loss = criterion(outputs, y_pu, mode='pu')
                        else:
                            raise ValueError(f"Unknown training method: {training_method}")

                        loss.backward()
                        optimizer.step()

                current_epoch = target_epoch

                # Save checkpoint
                checkpoint = {
                    'model_state_dict': copy.deepcopy(model.state_dict()),
                    'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                    'task_config': task_config,
                    'epoch': current_epoch,
                    'checkpoint_id': checkpoint_id,
                }

                metadata = {
                    'checkpoint_id': checkpoint_id,
                    'task_idx': task_idx,
                    'task_id': task.task_id,
                    'epoch': current_epoch,
                    **task_config,
                }

                self.checkpoints.append(checkpoint)
                self.checkpoint_metadata.append(metadata)

                checkpoint_id += 1

            print(f"  Created {len(checkpoint_epochs)} checkpoints at epochs {checkpoint_epochs}")

        print(f"\n{'='*70}")
        print(f"CHECKPOINT POOL CREATED: {len(self.checkpoints)} total checkpoints")
        print(f"{'='*70}\n")

        # Save checkpoint pool to disk for future reuse
        self.save_checkpoint_pool()

    def get_checkpoint(self, idx: int) -> Tuple[Dict, GaussianBlobTask, Dict]:
        """Get a specific checkpoint.

        Args:
            idx: Checkpoint index

        Returns:
            checkpoint: Checkpoint dictionary with model/optimizer states
            task: GaussianBlobTask instance
            dataloaders: Dict of dataloaders
        """
        checkpoint = self.checkpoints[idx]
        task_config = checkpoint['task_config']

        # Recreate task (exclude training_method)
        task_params = {k: v for k, v in task_config.items() if k != 'training_method'}
        task = GaussianBlobTask(**task_params)

        # Get dataloaders
        dataloaders = task.get_dataloaders(
            batch_size=self.config.get('inner_batch_size', 64),
            num_train=task_config['num_samples'],
            num_val=task_config['num_samples'] // 2,
            num_test=task_config['num_samples'] // 2,
        )

        return checkpoint, task, dataloaders

    def sample_batch(self, batch_size: int) -> List[int]:
        """Sample a batch of checkpoint indices.

        Args:
            batch_size: Number of checkpoints to sample

        Returns:
            List of checkpoint indices
        """
        indices = np.random.choice(
            len(self.checkpoints),
            size=min(batch_size, len(self.checkpoints)),
            replace=False,
        )
        return indices.tolist()

    def __len__(self):
        """Number of checkpoints in pool."""
        return len(self.checkpoints)
