"""Gaussian blob task generation for toy meta-learning example."""

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, Dict


class GaussianBlobTask:
    """Generate binary classification tasks using Gaussian blobs.

    Creates a PN learning task where:
    - Class 0 (negative): Gaussian centered at origin
    - Class 1 (positive): Gaussian centered at (mean_separation, 0, ...)
    - PU labels: some positives labeled (1), rest unlabeled (-1)
    """

    def __init__(
        self,
        num_dimensions: int = 2,
        mean_separation: float = 2.0,
        std: float = 1.0,
        prior: float = 0.5,
        labeling_freq: float = 0.3,
        num_samples: int = 1000,
        seed: int = 42,
        mode: str = 'pu',  # 'pu' or 'pn'
        negative_labeling_freq: float = 0.3,  # For PN mode
    ):
        """Initialize Gaussian blob task.

        Args:
            num_dimensions: Feature dimensionality
            mean_separation: Distance between positive and negative class centers
            std: Standard deviation of both Gaussians
            prior: Class prior (proportion of positives)
            labeling_freq: Proportion of positives that are labeled
            num_samples: Total number of samples per split
            seed: Random seed for reproducibility
            mode: 'pu' (PU learning) or 'pn' (PN learning with labeled negatives)
            negative_labeling_freq: Proportion of negatives that are labeled (PN mode only)
        """
        self.num_dimensions = num_dimensions
        self.mean_separation = mean_separation
        self.std = std
        self.prior = prior
        self.labeling_freq = labeling_freq
        self.negative_labeling_freq = negative_labeling_freq
        self.num_samples = num_samples
        self.seed = seed
        self.mode = mode

        # Task identifier
        self.task_id = (
            f"gaussian_{mode}_d{num_dimensions}_sep{mean_separation:.1f}_"
            f"std{std:.1f}_pi{prior:.1f}_c{labeling_freq:.1f}_s{seed}"
        )

    def generate_data(
        self,
        num_samples: int,
        seed: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate synthetic data.

        Args:
            num_samples: Number of samples to generate
            seed: Random seed

        Returns:
            X: Features [num_samples, num_dimensions]
            y_true: True binary labels [num_samples] (0 or 1)
            y_pu: PU labels [num_samples] (1 for labeled positive, -1 for unlabeled)
        """
        rng = np.random.RandomState(seed)

        # Generate class labels according to prior
        num_positive = int(num_samples * self.prior)
        num_negative = num_samples - num_positive

        # Generate features
        # Negative class: centered at origin
        X_neg = rng.randn(num_negative, self.num_dimensions) * self.std

        # Positive class: centered at (mean_separation, 0, ...)
        mean_pos = np.zeros(self.num_dimensions)
        mean_pos[0] = self.mean_separation
        X_pos = rng.randn(num_positive, self.num_dimensions) * self.std + mean_pos

        # Combine
        X = np.vstack([X_neg, X_pos])
        y_true = np.hstack([np.zeros(num_negative), np.ones(num_positive)])

        # Create PU or PN labels
        if self.mode == 'pu':
            # PU mode: Only label a fraction of positives, rest unlabeled (-1)
            positive_indices = np.where(y_true == 1)[0]
            num_labeled = int(len(positive_indices) * self.labeling_freq)
            labeled_indices = rng.choice(positive_indices, num_labeled, replace=False)

            # Set PU labels: labeled positives = 1, everything else = -1
            y_pu = -np.ones_like(y_true)
            y_pu[labeled_indices] = 1

        elif self.mode == 'pn':
            # PN mode: Label fraction of positives and fraction of negatives
            positive_indices = np.where(y_true == 1)[0]
            negative_indices = np.where(y_true == 0)[0]

            num_labeled_pos = int(len(positive_indices) * self.labeling_freq)
            num_labeled_neg = int(len(negative_indices) * self.negative_labeling_freq)

            labeled_pos_indices = rng.choice(positive_indices, num_labeled_pos, replace=False)
            labeled_neg_indices = rng.choice(negative_indices, num_labeled_neg, replace=False)

            # Set PN labels: labeled positives = 1, labeled negatives = 0, unlabeled = -1
            y_pu = -np.ones_like(y_true)
            y_pu[labeled_pos_indices] = 1
            y_pu[labeled_neg_indices] = 0
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Must be 'pu' or 'pn'.")

        # Shuffle
        perm = rng.permutation(num_samples)
        X = X[perm]
        y_true = y_true[perm]
        y_pu = y_pu[perm]

        return (
            torch.FloatTensor(X),
            torch.FloatTensor(y_true),
            torch.FloatTensor(y_pu),
        )

    def get_dataloaders(
        self,
        batch_size: int = 64,
        num_train: int = 1000,
        num_val: int = 500,
        num_test: int = 500,
    ) -> Dict[str, DataLoader]:
        """Generate train/val/test dataloaders.

        Args:
            batch_size: Batch size for dataloaders
            num_train: Number of training samples
            num_val: Number of validation samples
            num_test: Number of test samples

        Returns:
            Dictionary with 'train', 'val', 'test' dataloaders
        """
        # Generate data with different seeds for each split
        X_train, y_true_train, y_pu_train = self.generate_data(
            num_train, seed=self.seed
        )
        X_val, y_true_val, y_pu_val = self.generate_data(
            num_val, seed=self.seed + 1000
        )
        X_test, y_true_test, y_pu_test = self.generate_data(
            num_test, seed=self.seed + 2000
        )

        # Create datasets
        # Include BOTH true labels and PU labels for meta-learning
        # Format: (X, y_true, y_pu)
        train_dataset = TensorDataset(X_train, y_true_train, y_pu_train)
        val_dataset = TensorDataset(X_val, y_true_val, y_pu_val)
        test_dataset = TensorDataset(X_test, y_true_test, y_pu_test)

        # Create dataloaders with deterministic shuffling
        # Use task seed to create deterministic generator for shuffling
        generator = torch.Generator()
        generator.manual_seed(self.seed + 10000)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=generator,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        # Also store full datasets for meta-learning
        # (need true labels for BCE computation in meta-objective)
        val_dataset_with_pu = TensorDataset(X_val, y_pu_val, y_true_val)
        val_loader_with_pu = DataLoader(
            val_dataset_with_pu,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
            'val_with_pu': val_loader_with_pu,  # For meta-learning
        }

    def get_task_info(self) -> Dict:
        """Get task configuration info."""
        return {
            'task_id': self.task_id,
            'num_dimensions': self.num_dimensions,
            'mean_separation': self.mean_separation,
            'std': self.std,
            'prior': self.prior,
            'labeling_freq': self.labeling_freq,
            'num_samples': self.num_samples,
            'seed': self.seed,
        }
