"""Meta-trainer for MonotonicBasisLoss using checkpoint-based meta-learning.

This module implements the "models as data" paradigm where pre-trained model
checkpoints are treated as data points to be sampled and evaluated. Unlike
traditional meta-learning with inner loops, this approach:

1. Treats checkpoints as frozen "data samples" (model_state, task_data, val_data)
2. Samples checkpoint batches like a dataloader samples (x, y) batches
3. Optimizes only loss parameters, not model weights
4. Enables memory-efficient meta-learning without higher-order gradients

The meta-objective is to learn loss parameters that minimize validation loss
(BCE or S-NICE) on PN data when the loss is applied to PU training.
"""

import os
import gc
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.func import functional_call, vmap

from loss.loss_monotonic_basis import MonotonicBasisLoss
from loss.loss_polynomial_basis import PolynomialBasisLoss
from train.train_utils import select_model, prepare_loaders
from meta_learning.vmap_utils import (
    stack_state_dicts,
    get_device,
    check_vmap_compatibility
)
from meta_learning.checkpoint_pool import DATASET_TO_ARCH


class MetaTrainer:
    """Meta-learning trainer for MonotonicBasisLoss.

    Implements checkpoint-based meta-learning where pre-trained models are
    treated as data points. Each meta-iteration:
    1. Samples a batch of checkpoints from the pool
    2. Loads each checkpoint's model state (frozen)
    3. Computes training loss (PU data) + validation loss (PN data)
    4. Updates only the learned loss parameters via backprop

    Attributes:
        config: Configuration dictionary
        pool: CheckpointPool instance
        learned_loss: MonotonicBasisLoss being optimized
        optimizer_loss: Optimizer for loss parameters
        model_cache: Cache of models by dataset
        loader_cache: Cache of data loaders by task_id
        device: Torch device (cuda/cpu)
        iteration: Current meta-iteration
    """

    def __init__(
        self,
        config: Dict[str, Any],
        checkpoint_pool,
        device: Optional[torch.device] = None
    ):
        """Initialize meta-trainer.

        Args:
            config: Configuration dictionary with meta-learning settings
            checkpoint_pool: CheckpointPool instance
            device: Torch device (defaults to cuda if available)
        """
        self.config = config
        self.pool = checkpoint_pool
        # Device selection: MPS → CUDA → CPU (prioritize MPS on Mac)
        if device is not None:
            self.device = device
        elif config.get('device') == 'auto' or config.get('device') is None:
            self.device = get_device()  # MPS → CUDA → CPU from vmap_utils
        else:
            self.device = torch.device(config.get('device', 'cpu'))

        # Initialize learned loss (select type from config)
        loss_type = config.get('loss_type', 'monotonic_basis')

        if loss_type == 'polynomial_basis':
            self.learned_loss = PolynomialBasisLoss(
                num_repetitions=config.get('num_repetitions', 3),
                use_prior=config.get('use_prior', True),
                prior=0.5,  # Default, will be updated per task
                oracle_mode=config.get('oracle_mode', False),
                init_scale=config.get('init_scale', 0.01),
                init_mode=config.get('init_mode', 'random'),
                init_noise_scale=config.get('init_noise_scale', 0.0),
                l1_weight=config.get('l1_weight', 1e-4),
                l2_weight=config.get('l2_weight', 1e-3),
            ).to(self.device)
        elif loss_type == 'monotonic_basis':
            self.learned_loss = MonotonicBasisLoss(
                num_repetitions=config.get('num_repetitions', 3),
                num_fourier=config.get('num_fourier', 16),
                use_prior=config.get('use_prior', True),
                l1_weight=config.get('l1_weight', 1e-4),
                l2_weight=config.get('l2_weight', 1e-3),
                oracle_mode=config.get('oracle_mode', False),
                init_scale=config.get('init_scale', 0.01),
                init_mode=config.get('init_mode', 'random'),
                num_integration_points=config.get('num_integration_points', 20),
                integration_chunk_size=config.get('integration_chunk_size', None)
            ).to(self.device)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}. Must be 'polynomial_basis' or 'monotonic_basis'")

        # Optimizer for loss parameters only
        meta_optimizer = config.get('meta_optimizer', 'adamw').lower()
        if meta_optimizer == 'sgd':
            self.optimizer_loss = torch.optim.SGD(
                self.learned_loss.parameters(),
                lr=config.get('meta_lr', 1e-4),
                momentum=config.get('meta_momentum', 0.9),
                weight_decay=config.get('meta_weight_decay', 0.0)
            )
        elif meta_optimizer == 'adam':
            self.optimizer_loss = torch.optim.Adam(
                self.learned_loss.parameters(),
                lr=config.get('meta_lr', 1e-4)
            )
        else:  # default: adamw
            self.optimizer_loss = torch.optim.AdamW(
                self.learned_loss.parameters(),
                lr=config.get('meta_lr', 1e-4),
                weight_decay=config.get('meta_weight_decay', 0.01)
            )

        # Learning rate scheduler (optional)
        scheduler_type = config.get('meta_lr_scheduler', None)
        if scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_loss,
                T_max=config.get('meta_iterations', 1000),
                eta_min=config.get('meta_lr_min', 1e-6)
            )
        elif scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer_loss,
                step_size=config.get('lr_step_size', 500),
                gamma=config.get('lr_gamma', 0.5)
            )
        else:
            self.scheduler = None

        # Caches: treat models and loaders as reusable resources
        self.model_cache: Dict[str, nn.Module] = {}  # dataset -> model
        self.loader_cache: Dict[str, tuple] = {}  # task_id -> (train_loader, val_loader)

        # State
        self.iteration = 0
        self.best_meta_loss = float('inf')

        # Gradient accumulation
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        self._accumulated_steps = 0  # Counter for accumulation

        # Learning rate warmup
        self.warmup_iters = config.get('meta_lr_warmup_iters', 0)
        self.base_lr = config.get('meta_lr', 1e-4)
        if self.warmup_iters > 0:
            self.warmup_start_lr = self.base_lr / 3.0  # Start at 1/3 of target LR

        # Meta-objective type
        self.meta_objective = config.get('meta_objective', 'bce')
        assert self.meta_objective in ['bce', 'snice', 'anice'], \
            f"Invalid meta_objective: {self.meta_objective}"

        # Logging
        self.log_freq = config.get('log_freq', 10)
        self.save_freq = config.get('save_freq', 50)
        self.loss_checkpoint_dir = Path(config.get('loss_checkpoint_dir', './learned_losses'))
        self.loss_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint refresh
        self.refresh_freq = config.get('checkpoint_refresh_freq', 50)
        self.refresh_percent = config.get('checkpoint_refresh_percent', 10)

        # Adaptive batch sizing
        self.adaptive_batch_sizes = config.get('adaptive_batch_sizes', {})

        print(f"MetaTrainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Loss parameters: {sum(p.numel() for p in self.learned_loss.parameters())}")
        print(f"  Meta-objective: {self.meta_objective}")
        print(f"  Meta LR: {config.get('meta_lr', 1e-4)}")
        print(f"  L1 weight: {config.get('l1_weight', 1e-4)}")
        print(f"  L2 weight: {config.get('l2_weight', 1e-3)}")
        print()

        # Pre-cache data loaders if requested
        if config.get('precache_loaders', False):
            self._precache_all_loaders()

    def _precache_all_loaders(self):
        """Pre-cache data loaders for all unique tasks in the pool.

        This trades initialization time for faster iterations. With 180 unique tasks,
        pre-caching takes ~5-10 minutes but makes all subsequent iterations instant.
        """
        print("=" * 70)
        print("PRE-CACHING DATA LOADERS")
        print("=" * 70)
        print()

        # Get all unique task_ids from pool
        if self.pool.lazy_load:
            unique_tasks = {cp['task_id'] for cp in self.pool._checkpoint_metadata}
        else:
            unique_tasks = {cp['task_id'] for cp in self.pool.checkpoints}

        print(f"Pre-caching loaders for {len(unique_tasks)} unique tasks...")
        print(f"This will take ~5-10 minutes but makes iterations much faster.")
        print()

        import time
        start_time = time.time()

        for idx, task_id in enumerate(sorted(unique_tasks), 1):
            # Create a dummy checkpoint dict with task metadata
            # Extract from pool metadata
            if self.pool.lazy_load:
                cp_meta = next(cp for cp in self.pool._checkpoint_metadata if cp['task_id'] == task_id)
            else:
                cp_meta = next(cp for cp in self.pool.checkpoints if cp['task_id'] == task_id)

            # Log each task being cached
            print(f"  [{idx}/{len(unique_tasks)}] Caching {task_id}...", end=" ", flush=True)
            task_start = time.time()

            # Get loaders (this will cache them)
            self._get_loaders(cp_meta)

            task_time = time.time() - task_start
            elapsed = time.time() - start_time
            eta = elapsed / idx * (len(unique_tasks) - idx) if idx < len(unique_tasks) else 0
            print(f"Done ({task_time:.1f}s) | Total: {elapsed:.1f}s | ETA: {eta:.1f}s")

        total_time = time.time() - start_time
        print()
        print(f"✓ Pre-cached {len(unique_tasks)} data loaders in {total_time:.1f}s")
        print(f"  Cache size: {len(self.loader_cache)} tasks")
        print("=" * 70)
        print()

    def _get_model(self, dataset: str) -> nn.Module:
        """Get or create model for dataset.

        Models are cached by dataset since all tasks from the same dataset
        share the same architecture. Checkpoints provide the trained weights.

        Args:
            dataset: Dataset name (e.g., 'mnist', 'fashionmnist')

        Returns:
            Model instance (on device, in eval mode)
        """
        if dataset not in self.model_cache:
            # Create model with default parameters
            model_params = {
                'dataset_class': dataset.upper(),  # MNIST, FashionMNIST, etc.
                'optimizer': self.config.get('optimizer', 'adam'),
                'lr': self.config.get('lr', 3e-4),
                'weight_decay': self.config.get('weight_decay', 1e-4)
            }

            # Use a default prior (will be set in loss anyway)
            # NOTE: Checkpoints were created with 'pn_naive' method, so use that for compatibility
            model = select_model('pn_naive', model_params, prior=0.5)
            model = model.to(self.device)
            model.eval()  # Always in eval mode (frozen)

            self.model_cache[dataset] = model
            print(f"Created model for dataset: {dataset}")

        return self.model_cache[dataset]

    def _get_loaders(self, checkpoint: Dict[str, Any]) -> tuple:
        """Get or create data loaders for checkpoint's task.

        Loaders are cached by task_id to avoid recreating them for each
        checkpoint from the same task (different epochs, same data).

        Args:
            checkpoint: Checkpoint dictionary

        Returns:
            (train_loader, val_loader) tuple
        """
        task_id = checkpoint['task_id']

        if task_id not in self.loader_cache:
            dataset = checkpoint['dataset']
            c_value = checkpoint['c_value']
            prior = checkpoint['prior']

            # Create data config
            data_config = {
                'dataset_class': dataset.upper(),  # MNIST, FASHIONMNIST, etc.
                'c_values': [c_value],
                'scenarios': ['case-control'],
                'selection_strategies': ['random'],
                'val_ratio': 0.1,
                'target_prevalence': prior,
                'num_workers': self.config.get('num_workers', 0),  # Configurable (0=no multiprocessing, safer)
                'prefetch_factor': self.config.get('prefetch_factor', 2) if self.config.get('num_workers', 0) > 0 else None,
                'persistent_workers': self.config.get('persistent_workers', False) and self.config.get('num_workers', 0) > 0,
                'pin_memory': self.config.get('pin_memory', False)
            }

            # Add SBERT settings for text datasets
            if dataset.lower() == 'imdb':
                data_config.update({
                    'sbert_model_name': 'all-MiniLM-L6-v2',
                    'sbert_model_path': './scripts/models/all-MiniLM-L6-v2',
                    'sbert_embeddings_path': './scripts/embeddings/imdb_sbert_embeddings.npz'
                })
            elif dataset.lower() == '20news':
                data_config.update({
                    'sbert_model_name': 'all-MiniLM-L6-v2',
                    'sbert_model_path': './scripts/models/all-MiniLM-L6-v2',
                    'sbert_embeddings_path': './scripts/embeddings/20news_sbert_embeddings.npz'
                })

            # Prepare loaders
            train_loader, val_loader, test_loader, actual_prior, _ = prepare_loaders(
                dataset_name=dataset,
                data_config=data_config,
                batch_size=self.config.get('batch_size', 128),
                method='pn_naive'  # Match checkpoint creation method
            )

            self.loader_cache[task_id] = (train_loader, val_loader)

        return self.loader_cache[task_id]

    def _compute_meta_objective(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        prior: float
    ) -> torch.Tensor:
        """Compute meta-objective (validation loss).

        Args:
            model: Model (frozen, in eval mode)
            val_loader: Validation data loader (PN labels)
            prior: Class prior for S-NICE/A-NICE

        Returns:
            Scalar validation loss
        """
        model.eval()
        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                # PUDataset returns: (features, pu_labels, true_labels, indices, pseudo_labels)
                x = batch[0].to(self.device)  # features
                y = batch[2].to(self.device)  # true_labels (not pu_labels!)

                # Forward pass
                outputs = model(x).view(-1)
                probs = torch.sigmoid(outputs)

                # Compute loss based on objective type
                if self.meta_objective == 'bce':
                    loss = F.binary_cross_entropy(probs, y.float(), reduction='mean')
                elif self.meta_objective == 'snice':
                    # S-NICE: simplified version
                    # L = -[y * log(p) + (1-y) * log(1-p)]
                    # This is just BCE, but we can add sample weighting if needed
                    loss = F.binary_cross_entropy(probs, y.float(), reduction='mean')
                elif self.meta_objective == 'anice':
                    # A-NICE: asymmetric version
                    # Weight positive and negative samples differently
                    pos_mask = y == 1
                    neg_mask = y == 0
                    if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                        pos_loss = F.binary_cross_entropy(
                            probs[pos_mask],
                            torch.ones_like(probs[pos_mask]),
                            reduction='mean'
                        )
                        neg_loss = F.binary_cross_entropy(
                            probs[neg_mask],
                            torch.zeros_like(probs[neg_mask]),
                            reduction='mean'
                        )
                        loss = prior * pos_loss + (1 - prior) * neg_loss
                    else:
                        loss = F.binary_cross_entropy(probs, y.float(), reduction='mean')

                val_loss += loss.item()
                num_batches += 1

        return torch.tensor(val_loss / num_batches if num_batches > 0 else 0.0)

    def meta_train_step(self, meta_batch: List[Dict[str, Any]]) -> Dict[str, float]:
        """Single meta-training iteration.

        Treats each checkpoint as a "data sample" containing:
        - model_state: Pre-trained weights (frozen "input")
        - train_data: PU data for computing training loss
        - val_data: PN data for computing meta-objective

        For each checkpoint in the meta-batch:
        1. Load pre-trained model weights (frozen)
        2. Compute training loss on PU data using learned loss
        3. Compute validation loss on PN data (meta-objective)
        4. Accumulate gradients w.r.t. loss parameters

        After processing all checkpoints:
        5. Add regularization
        6. Update loss parameters

        Args:
            meta_batch: List of checkpoint dictionaries

        Returns:
            Dictionary with loss components
        """
        self.learned_loss.train()

        total_train_loss = 0.0
        total_val_loss = 0.0
        total_reg_loss = 0.0
        meta_loss = 0.0

        # Process each checkpoint (model is "data", loss is "learner")
        for i, checkpoint in enumerate(meta_batch):
            dataset = checkpoint['dataset']
            prior = checkpoint['prior']

            # Get model and loaders (cached)
            model = self._get_model(dataset)
            train_loader, val_loader = self._get_loaders(checkpoint)

            # Load checkpoint weights (frozen snapshot of training state)
            model.load_state_dict(checkpoint['model_state'])
            model.eval()

            # Set prior in learned loss
            self.learned_loss.set_prior(prior)

            # === Training loss (for gradient flow) ===
            # Only use a single batch for efficiency
            try:
                # PUDataset returns: (features, pu_labels, true_labels, indices, pseudo_labels)
                batch_train = next(iter(train_loader))
                x_train = batch_train[0].to(self.device)  # features
                t_train = batch_train[1].to(self.device)  # pu_labels

                # Forward pass through model (no grad)
                with torch.no_grad():
                    outputs_train = model(x_train).view(-1)

                # Compute loss (with grad for loss params)
                train_loss = self.learned_loss(outputs_train, t_train)

                # Accumulate
                meta_loss = meta_loss + train_loss
                total_train_loss += train_loss.item()

            except StopIteration:
                print(f"Warning: Empty train loader for checkpoint {i}")

            # === Validation loss (meta-objective) ===
            # Use a single batch for efficiency
            try:
                # PUDataset returns: (features, pu_labels, true_labels, indices, pseudo_labels)
                batch_val = next(iter(val_loader))
                x_val = batch_val[0].to(self.device)  # features
                y_val = batch_val[2].to(self.device)  # true_labels (not pu_labels!)

                # Forward pass through model (no grad)
                with torch.no_grad():
                    outputs_val = model(x_val).view(-1)
                    probs_val = torch.sigmoid(outputs_val)

                # Compute validation loss
                if self.meta_objective == 'bce':
                    val_loss = F.binary_cross_entropy(probs_val, y_val.float(), reduction='mean')
                elif self.meta_objective == 'snice':
                    val_loss = F.binary_cross_entropy(probs_val, y_val.float(), reduction='mean')
                elif self.meta_objective == 'anice':
                    pos_mask = y_val == 1
                    neg_mask = y_val == 0
                    if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                        pos_loss = F.binary_cross_entropy(
                            probs_val[pos_mask],
                            torch.ones_like(probs_val[pos_mask]),
                            reduction='mean'
                        )
                        neg_loss = F.binary_cross_entropy(
                            probs_val[neg_mask],
                            torch.zeros_like(probs_val[neg_mask]),
                            reduction='mean'
                        )
                        val_loss = prior * pos_loss + (1 - prior) * neg_loss
                    else:
                        val_loss = F.binary_cross_entropy(probs_val, y_val.float(), reduction='mean')

                # Accumulate
                meta_loss = meta_loss + val_loss
                total_val_loss += val_loss.item()

            except StopIteration:
                print(f"Warning: Empty val loader for checkpoint {i}")

        # === Regularization ===
        reg_loss = self.learned_loss.compute_regularization()
        meta_loss = meta_loss + reg_loss
        total_reg_loss = reg_loss.item()

        # === Normalize by batch size ===
        meta_loss = meta_loss / len(meta_batch)
        total_train_loss /= len(meta_batch)
        total_val_loss /= len(meta_batch)
        # reg_loss already normalized (sum over all params)

        # === Backward and update ===
        self.optimizer_loss.zero_grad()
        meta_loss.backward()
        self.optimizer_loss.step()

        return {
            'meta_loss': meta_loss.item(),
            'train_loss': total_train_loss,
            'val_loss': total_val_loss,
            'reg_loss': total_reg_loss
        }

    def train(self, num_iterations: int):
        """Run full meta-training loop.

        Args:
            num_iterations: Number of meta-iterations to run
        """
        print("=" * 70)
        print("META-TRAINING")
        print("=" * 70)
        print()
        print(f"Total iterations: {num_iterations}")
        print(f"Meta-batch size: {self.config.get('meta_batch_size', 8)}")
        print(f"Checkpoint pool size: {len(self.pool.checkpoints)}")
        print()

        start_time = time.time()

        for iteration in range(num_iterations):
            self.iteration = iteration

            # Sample meta-batch (like dataloader sampling (x, y) batches)
            meta_batch = self.pool.sample_meta_batch(
                self.config.get('meta_batch_size', 8)
            )

            # Meta-training step
            losses = self.meta_train_step(meta_batch)

            # Logging
            if iteration % self.log_freq == 0:
                elapsed = time.time() - start_time
                iter_per_sec = (iteration + 1) / elapsed if elapsed > 0 else 0

                print(f"Iteration {iteration:4d}/{num_iterations} | "
                      f"Meta: {losses['meta_loss']:.4f} | "
                      f"Train: {losses['train_loss']:.4f} | "
                      f"Val: {losses['val_loss']:.4f} | "
                      f"Reg: {losses['reg_loss']:.6f} | "
                      f"{iter_per_sec:.2f} it/s")

            # Save checkpoint
            if iteration > 0 and iteration % self.save_freq == 0:
                self._save_loss_checkpoint(iteration, losses)

            # Refresh checkpoint pool (curriculum evolution)
            if iteration > 0 and iteration % self.refresh_freq == 0:
                print()
                self._refresh_pool(iteration)
                print()

            # Track best
            if losses['meta_loss'] < self.best_meta_loss:
                self.best_meta_loss = losses['meta_loss']
                self._save_loss_checkpoint('best', losses)

        print()
        print("=" * 70)
        print("META-TRAINING COMPLETE")
        print("=" * 70)
        print(f"Best meta-loss: {self.best_meta_loss:.4f}")
        print(f"Total time: {time.time() - start_time:.1f}s")
        print()

        # Save final checkpoint
        self._save_loss_checkpoint('final', losses)

    def _save_loss_checkpoint(self, iteration: Any, losses: Dict[str, float]):
        """Save learned loss checkpoint.

        Args:
            iteration: Iteration number or string ('best', 'final')
            losses: Dictionary of loss components
        """
        checkpoint_path = self.loss_checkpoint_dir / f"loss_iter_{iteration}.pt"

        checkpoint = {
            'iteration': self.iteration,
            'state_dict': self.learned_loss.state_dict(),
            'optimizer_state': self.optimizer_loss.state_dict(),
            'losses': losses,
            'config': self.config,
            'best_meta_loss': self.best_meta_loss
        }

        torch.save(checkpoint, checkpoint_path)

        if iteration == 'best' or iteration == 'final':
            print(f"Saved {iteration} loss checkpoint: {checkpoint_path}")

    def _refresh_pool(self, iteration: int):
        """Refresh checkpoint pool with curriculum strategy.

        Args:
            iteration: Current meta-iteration
        """
        print(f"Refreshing checkpoint pool at iteration {iteration}...")

        # Create trainer factory for checkpoint refresh
        def trainer_factory(task_id: str, target_epoch: int, loss_fn, device):
            """Train a model to target_epoch using loss_fn.

            This is a minimal trainer for checkpoint refresh.
            """
            # Parse task_id to extract task info
            # Format: "mnist_c0.3_prior0.5_seed42"
            parts = task_id.rsplit('_seed', 1)
            base_id = parts[0]
            seed = int(parts[1]) if len(parts) > 1 else 42

            # Parse base_id: "mnist_c0.3_prior0.5"
            dataset = base_id.split('_c')[0]
            c_value = float(base_id.split('_c')[1].split('_prior')[0])
            prior = float(base_id.split('_prior')[1])

            # Create data config
            data_config = {
                'dataset_class': dataset.upper(),
                'c_values': [c_value],
                'scenarios': ['case-control'],
                'selection_strategies': ['random'],
                'val_ratio': 0.1,
                'target_prevalence': prior,
                'num_workers': self.config.get('num_workers', 0),  # Configurable (0=no multiprocessing, safer)
                'prefetch_factor': self.config.get('prefetch_factor', 2) if self.config.get('num_workers', 0) > 0 else None,
                'persistent_workers': self.config.get('persistent_workers', False) and self.config.get('num_workers', 0) > 0,
                'pin_memory': self.config.get('pin_memory', False)
            }

            # Add precomputed embeddings paths for text datasets
            if dataset == 'imdb':
                data_config['sbert_embeddings_path'] = './scripts/embeddings/imdb_sbert_embeddings.npz'
            elif dataset == '20news':
                data_config['sbert_embeddings_path'] = './scripts/embeddings/20news_sbert_embeddings.npz'

            # Prepare loaders
            train_loader, _, _, actual_prior, _ = prepare_loaders(
                dataset_name=dataset,
                data_config=data_config,
                batch_size=self.config.get('batch_size', 128),
                method='pn_naive'  # Match checkpoint creation method
            )

            # Create fresh model
            model_params = {
                'optimizer': self.config.get('optimizer', 'adam'),
                'lr': self.config.get('lr', 3e-4),
                'weight_decay': self.config.get('weight_decay', 1e-4)
            }
            model = select_model('pn_naive', model_params, prior=actual_prior)
            model = model.to(device)

            # Optimizer
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=model_params['lr'],
                weight_decay=model_params['weight_decay']
            )

            # Train to target epoch
            loss_fn_copy = deepcopy(loss_fn)
            loss_fn_copy.set_prior(prior)
            loss_fn_copy.eval()  # Frozen

            for epoch in range(target_epoch + 1):  # Include epoch 0
                model.train()
                for batch in train_loader:
                    x, t = batch[0].to(device), batch[1].to(device)

                    optimizer.zero_grad()
                    outputs = model(x).view(-1)
                    loss = loss_fn_copy(outputs, t)
                    loss.backward()
                    optimizer.step()

            return model.state_dict()

        # Call pool refresh
        self.pool.refresh_pool(
            learned_loss=self.learned_loss,
            percent=self.refresh_percent,
            current_iteration=iteration,
            trainer_factory=trainer_factory,
            device=self.device
        )

    # ==================== K=3 Inner Loop Methods ====================

    def _group_by_dataset(self, meta_batch: List[Dict]) -> Dict[str, List[Dict]]:
        """Group checkpoints by dataset for vmapping.

        Each dataset has unique input dimensions, so we must group by dataset
        (not just architecture) to ensure state dicts are compatible.

        Args:
            meta_batch: List of checkpoint dictionaries

        Returns:
            Dictionary mapping dataset name -> list of checkpoints
        """
        groups = {}
        for cp in meta_batch:
            dataset = cp.get('dataset', self.pool.get_dataset_from_task_id(cp['task_id']))

            if dataset not in groups:
                groups[dataset] = []
            groups[dataset].append(cp)

        return groups

    def _get_meta_batch_size_for_dataset(self, dataset: str) -> int:
        """Get optimal meta-batch size for dataset.

        Args:
            dataset: Dataset name (e.g., 'mnist', 'mushrooms')

        Returns:
            Optimal batch size for this dataset
        """
        return self.adaptive_batch_sizes.get(
            dataset.lower(),
            self.config.get('meta_batch_size', 8)  # Default fallback
        )

    def sample_meta_batch_adaptive(self) -> List[Dict]:
        """Sample meta-batch with dataset-specific sizes.

        Returns:
            List of sampled checkpoints with adaptive sizing
        """
        checkpoints = []

        # Get available datasets
        if self.pool.lazy_load:
            all_datasets = set(cp['dataset'] for cp in self.pool._checkpoint_metadata)
        else:
            all_datasets = set(cp['dataset'] for cp in self.pool.checkpoints)

        # Sample from each dataset proportionally to its optimal batch size
        total_budget = self.config.get('meta_batch_size', 32)

        # Simple strategy: cycle through datasets and sample according to their sizes
        datasets_list = sorted(all_datasets)
        samples_per_dataset = {}

        for dataset in datasets_list:
            optimal_size = self._get_meta_batch_size_for_dataset(dataset)
            # Allocate proportionally (can be improved with smarter scheduling)
            samples_per_dataset[dataset] = min(optimal_size, total_budget // len(datasets_list))

        # Sample from each dataset
        for dataset, num_samples in samples_per_dataset.items():
            if num_samples == 0:
                continue

            # Get checkpoints for this dataset
            if self.pool.lazy_load:
                dataset_checkpoints = [cp for cp in self.pool._checkpoint_metadata if cp['dataset'] == dataset]
            else:
                dataset_checkpoints = [cp for cp in self.pool.checkpoints if cp['dataset'] == dataset]

            # Sample
            if len(dataset_checkpoints) >= num_samples:
                import random
                sampled_indices = random.sample(range(len(dataset_checkpoints)), num_samples)

                for idx in sampled_indices:
                    if self.pool.lazy_load:
                        checkpoint = self.pool._load_checkpoint_state(dataset_checkpoints[idx])
                    else:
                        checkpoint = dataset_checkpoints[idx]
                    checkpoints.append(checkpoint)

        return checkpoints

    def _prepare_batched_data(
        self,
        checkpoints: List[Dict]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[float]]:
        """Sample and stack data batches from checkpoint loaders.

        Args:
            checkpoints: List of checkpoint dicts

        Returns:
            Tuple of (x_train_batched, t_train_batched, x_val_batched, y_val_batched, priors)
        """
        x_train_list, t_train_list = [], []
        x_val_list, y_val_list = [], []
        priors = []

        for cp in checkpoints:
            # Get loaders
            train_loader, val_loader = self._get_loaders(cp)

            # Sample batch from train loader (PU data)
            # PUDataset returns: (features, pu_labels, true_labels, indices, pseudo_labels)
            batch_train = next(iter(train_loader))
            x_train = batch_train[0]  # features
            t_train = batch_train[1]  # pu_labels
            x_train_list.append(x_train)
            t_train_list.append(t_train)

            # Sample batch from val loader (PN data)
            # For validation, use true_labels instead of pu_labels
            batch_val = next(iter(val_loader))
            x_val = batch_val[0]  # features
            y_val = batch_val[2]  # true_labels (not pu_labels!)
            x_val_list.append(x_val)
            y_val_list.append(y_val)

            # Get prior
            priors.append(cp['prior'])

        # Stack into [num_ckpts, batch_size, ...]
        x_train_batched = torch.stack(x_train_list).to(self.device)
        t_train_batched = torch.stack(t_train_list).to(self.device)
        x_val_batched = torch.stack(x_val_list).to(self.device)
        y_val_batched = torch.stack(y_val_list).to(self.device)

        return x_train_batched, t_train_batched, x_val_batched, y_val_batched, priors

    def _vmapped_inner_loops(
        self,
        checkpoints: List[Dict],
        K: int = 3
    ) -> torch.Tensor:
        """Process multiple checkpoints with K=3 inner loop using batched operations.

        This optimized version combines all N checkpoints' batches into one large batch
        for better GPU utilization. Instead of processing N small batches sequentially,
        we process one large batch of size (N × batch_size), which better saturates GPU.

        Args:
            checkpoints: List of checkpoint dicts (all same dataset)
            K: Number of inner loop gradient steps

        Returns:
            improvements: [num_ckpts] tensor with gradients to loss params
        """
        if not checkpoints:
            return torch.tensor(0.0, device=self.device)

        # Use vmap if enabled, otherwise fall back to sequential
        use_vmap = self.config.get('use_vmap', True)

        if not use_vmap:
            return self._vmapped_inner_loops_sequential(checkpoints, K)

        # Get dataset
        dataset = checkpoints[0].get('dataset',
                                     self.pool.get_dataset_from_task_id(checkpoints[0]['task_id']))

        if self.config.get('verbose', False):
            print(f"\n      Loading data loaders...", end=" ", flush=True)

        # Prepare batched data
        x_train_batched, t_train_batched, x_val_batched, y_val_batched, priors = \
            self._prepare_batched_data(checkpoints)

        priors_tensor = torch.tensor(priors, device=self.device)

        N = len(checkpoints)
        batch_size = x_train_batched.shape[1]

        if self.config.get('verbose', False):
            print(f"done")
            print(f"      Running K={K} inner loops for {N} checkpoints (batched)...", end=" ", flush=True)

        # OPTIMIZATION: Combine all N checkpoints' batches into one large batch
        # Preserve all dimensions except combining first two: [N, batch_size, ...] -> [N*batch_size, ...]
        # This works for both tabular ([N, B, features]) and image data ([N, B, C, H, W])
        x_train_combined = x_train_batched.view(N * batch_size, *x_train_batched.shape[2:])
        t_train_combined = t_train_batched.view(N * batch_size)
        x_val_combined = x_val_batched.view(N * batch_size, *x_val_batched.shape[2:])
        y_val_combined = y_val_batched.view(N * batch_size)

        # Process each checkpoint with its slice of the combined batch
        # This gives better GPU utilization than processing one at a time
        improvements = []
        final_losses = []

        for i in range(N):
            checkpoint_dataset = checkpoints[i].get('dataset',
                                                    self.pool.get_dataset_from_task_id(checkpoints[i]['task_id']))

            # Create fresh model for this checkpoint's dataset
            model_params = {
                'dataset_class': checkpoint_dataset.upper(),
                'optimizer': self.config.get('optimizer', 'adam'),
                'lr': self.config.get('lr', 3e-4),
                'weight_decay': self.config.get('weight_decay', 1e-4)
            }
            model = select_model('pn_naive', model_params, prior=0.5)
            model = model.to(self.device)

            # For dynamic models (like MLP_20News), build the model first
            if hasattr(model, '_build') and not model.built:
                first_layer_key = 'layers.0.0.weight'
                if first_layer_key in checkpoints[i]['model_state']:
                    in_features = checkpoints[i]['model_state'][first_layer_key].shape[1]
                    model._build(in_features)
                    model = model.to(self.device)

            # Load checkpoint state
            model.load_state_dict(checkpoints[i]['model_state'])
            model.train()  # Enable gradients

            # Fresh optimizer for inner loop
            inner_lr = self.config.get('inner_lr', 1e-3)
            inner_optimizer = self.config.get('inner_optimizer', 'adam').lower()
            if inner_optimizer == 'sgd':
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=inner_lr,
                    momentum=self.config.get('inner_momentum', 0.9),
                    weight_decay=self.config.get('weight_decay', 0.0)
                )
            else:  # default: adam
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=inner_lr,
                    weight_decay=self.config.get('weight_decay', 1e-4)
                )

            # Get this checkpoint's data slice from combined batch
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            x_train = x_train_combined[start_idx:end_idx]
            t_train = t_train_combined[start_idx:end_idx]
            x_val = x_val_combined[start_idx:end_idx]
            y_val = y_val_combined[start_idx:end_idx]
            prior = priors_tensor[i]

            # Baseline validation loss (before K steps)
            with torch.no_grad():
                val_out_before = model(x_val).view(-1)
                val_loss_before = F.binary_cross_entropy_with_logits(val_out_before, y_val.float())

            # K gradient steps with learned loss
            for _ in range(K):
                optimizer.zero_grad()
                train_out = model(x_train).view(-1)

                # Set prior for this checkpoint
                self.learned_loss.set_prior(prior.item())

                # Compute loss with learned loss
                train_loss = self.learned_loss(train_out, t_train)

                # Backward with second-order gradients
                train_loss.backward(create_graph=True)

                optimizer.step()

            # Final validation loss (after K steps, WITH gradients)
            val_out_after = model(x_val).view(-1)
            val_loss_after = F.binary_cross_entropy_with_logits(val_out_after, y_val.float())

            # Improvement (positive = better)
            # Detach baseline to ensure gradients flow only through val_loss_after
            improvement = val_loss_before.detach() - val_loss_after
            improvements.append(improvement)
            final_losses.append(val_loss_after)

            # CRITICAL: Clear gradients to break reference cycles and prevent memory leaks
            # PyTorch warning: "reset the .grad fields of your parameters to None after use"
            for param in model.parameters():
                param.grad = None

            # Explicitly delete model and optimizer to free memory
            del model, optimizer

        improvements = torch.stack(improvements)
        final_losses = torch.stack(final_losses)

        # Clear GPU cache after processing all checkpoints in this dataset group
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()

        if self.config.get('verbose', False):
            avg_imp = improvements.mean().item()
            avg_final_loss = final_losses.mean().item()
            print(f"done (avg_imp={avg_imp:+.4f}, avg_final_bce={avg_final_loss:.4f})")

        return improvements, final_losses

    def _vmapped_inner_loops_sequential(
        self,
        checkpoints: List[Dict],
        K: int = 3
    ) -> torch.Tensor:
        """Sequential checkpoint processing (original implementation).

        This is the original implementation - kept for testing and as fallback.
        Processes checkpoints one-by-one in a for loop.

        Args:
            checkpoints: List of checkpoint dicts (all same dataset)
            K: Number of inner loop gradient steps

        Returns:
            improvements: [num_ckpts] tensor with gradients to loss params
        """
        if not checkpoints:
            return torch.tensor(0.0, device=self.device)

        # Get dataset
        dataset = checkpoints[0].get('dataset',
                                     self.pool.get_dataset_from_task_id(checkpoints[0]['task_id']))

        if self.config.get('verbose', False):
            print(f"\n      Loading data loaders...", end=" ", flush=True)

        # Prepare batched data
        x_train_batched, t_train_batched, x_val_batched, y_val_batched, priors = \
            self._prepare_batched_data(checkpoints)

        priors_tensor = torch.tensor(priors, device=self.device)

        if self.config.get('verbose', False):
            print(f"done")
            print(f"      Running K={K} inner loops for {len(checkpoints)} checkpoints...", end=" ", flush=True)

        # Process checkpoints sequentially (simpler than vmap for now)
        improvements = []
        final_losses = []
        for i in range(len(checkpoints)):
            checkpoint_dataset = checkpoints[i].get('dataset',
                                                    self.pool.get_dataset_from_task_id(checkpoints[i]['task_id']))

            # Create fresh model for this checkpoint's dataset
            model_params = {
                'dataset_class': checkpoint_dataset.upper(),
                'optimizer': self.config.get('optimizer', 'adam'),
                'lr': self.config.get('lr', 3e-4),
                'weight_decay': self.config.get('weight_decay', 1e-4)
            }
            model = select_model('pn_naive', model_params, prior=0.5)
            model = model.to(self.device)

            # For dynamic models (like MLP_20News), build the model first
            if hasattr(model, '_build') and not model.built:
                # Get input dimension from checkpoint state dict
                first_layer_key = 'layers.0.0.weight'
                if first_layer_key in checkpoints[i]['model_state']:
                    in_features = checkpoints[i]['model_state'][first_layer_key].shape[1]
                    model._build(in_features)
                    model = model.to(self.device)

            # Load checkpoint state (use original state dict, not flattened)
            model.load_state_dict(checkpoints[i]['model_state'])
            model.train()  # Enable gradients

            # Fresh optimizer for inner loop
            inner_lr = self.config.get('inner_lr', 1e-3)
            inner_optimizer = self.config.get('inner_optimizer', 'adam').lower()
            if inner_optimizer == 'sgd':
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=inner_lr,
                    momentum=self.config.get('inner_momentum', 0.9),
                    weight_decay=self.config.get('weight_decay', 0.0)
                )
            else:  # default: adam
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=inner_lr,
                    weight_decay=self.config.get('weight_decay', 1e-4)
                )

            # Get this checkpoint's data
            x_train = x_train_batched[i]
            t_train = t_train_batched[i]
            x_val = x_val_batched[i]
            y_val = y_val_batched[i]
            prior = priors_tensor[i]

            # Baseline validation loss (before K steps)
            with torch.no_grad():
                val_out_before = model(x_val).view(-1)
                val_loss_before = F.binary_cross_entropy_with_logits(val_out_before, y_val.float())

            # K gradient steps with learned loss
            for _ in range(K):
                optimizer.zero_grad()
                train_out = model(x_train).view(-1)

                # Set prior for this checkpoint
                self.learned_loss.set_prior(prior.item())

                # Compute loss with learned loss
                train_loss = self.learned_loss(train_out, t_train)

                # Backward with second-order gradients
                train_loss.backward(create_graph=True)

                optimizer.step()

            # Final validation loss (after K steps, WITH gradients)
            val_out_after = model(x_val).view(-1)
            val_loss_after = F.binary_cross_entropy_with_logits(val_out_after, y_val.float())

            # Improvement (positive = better)
            # Detach baseline to ensure gradients flow only through val_loss_after
            improvement = val_loss_before.detach() - val_loss_after
            improvements.append(improvement)
            final_losses.append(val_loss_after)

            # CRITICAL: Clear gradients to break reference cycles and prevent memory leaks
            # PyTorch warning: "reset the .grad fields of your parameters to None after use"
            for param in model.parameters():
                param.grad = None

            # Explicitly delete model and optimizer to free memory
            del model, optimizer

        improvements = torch.stack(improvements)
        final_losses = torch.stack(final_losses)

        if self.config.get('verbose', False):
            avg_imp = improvements.mean().item()
            avg_final_loss = final_losses.mean().item()
            print(f"done (avg_imp={avg_imp:+.4f}, avg_final_bce={avg_final_loss:.4f})")

        return improvements, final_losses

    def meta_train_step_k3(self, meta_batch: List[Dict]) -> Dict[str, float]:
        """Single meta-training iteration with K=3 inner loop and gradient accumulation.

        Processes checkpoints grouped by dataset (each dataset has unique input dims).

        Args:
            meta_batch: List of checkpoint dictionaries

        Returns:
            Dictionary with training metrics
        """
        print(f"  [Training] Processing meta-batch with {len(meta_batch)} checkpoints...")

        # Group by dataset (not architecture) since each dataset has unique input dimensions
        dataset_groups = self._group_by_dataset(meta_batch)

        print(f"  [Training] Grouped into {len(dataset_groups)} datasets: {list(dataset_groups.keys())}")

        # Zero gradients only at start of accumulation cycle
        if self._accumulated_steps == 0:
            self.optimizer_loss.zero_grad()

        total_improvement = 0.0
        total_oracle_loss = 0.0
        num_checkpoints = 0

        # Get oracle loss weight from config (default: 1.0 for equal weight)
        oracle_loss_weight = self.config.get('oracle_loss_weight', 1.0)

        # Process each dataset group
        for idx, (dataset_name, checkpoints) in enumerate(dataset_groups.items(), 1):
            if not checkpoints:
                continue

            print(f"    [{idx}/{len(dataset_groups)}] Processing {dataset_name}: {len(checkpoints)} checkpoints...", end=" ", flush=True)

            # Vmap K=3 inner loops for this dataset
            improvements, final_bce_losses = self._vmapped_inner_loops(
                checkpoints,
                K=self.config.get('K_inner_steps', 3)
            )

            # Combine two objectives:
            # 1. Maximize improvement (negative for minimization)
            # 2. Minimize final oracle BCE loss
            dataset_improvement = improvements.sum()
            dataset_oracle_loss = final_bce_losses.mean()  # Average BCE across checkpoints

            # Combined meta-loss: -improvement + lambda * oracle_loss
            meta_loss = -dataset_improvement + oracle_loss_weight * dataset_oracle_loss * len(checkpoints)

            # Accumulate gradients (don't scale yet - will scale before step)
            meta_loss.backward()

            total_improvement += dataset_improvement.item()
            total_oracle_loss += dataset_oracle_loss.item() * len(checkpoints)
            num_checkpoints += len(checkpoints)

            avg_imp = dataset_improvement.item() / len(checkpoints)
            avg_oracle = dataset_oracle_loss.item()
            print(f"Done (avg_imp={avg_imp:+.4f}, avg_oracle_bce={avg_oracle:.4f})")

            # Clear GPU cache and run garbage collection after each dataset group
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            gc.collect()

        # Add regularization (accumulates more gradients)
        reg_loss = self.learned_loss.compute_regularization()
        reg_loss.backward()

        # Increment accumulation counter
        self._accumulated_steps += 1

        # Optimizer step only after N accumulation steps
        if self._accumulated_steps >= self.gradient_accumulation_steps:
            # Scale gradients by number of accumulation steps
            # This keeps effective learning rate constant
            for param in self.learned_loss.parameters():
                if param.grad is not None:
                    param.grad /= self.gradient_accumulation_steps

            # Gradient clipping (if enabled)
            if self.config.get('gradient_clip_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.learned_loss.parameters(),
                    max_norm=self.config['gradient_clip_norm']
                )

            self.optimizer_loss.step()
            self._accumulated_steps = 0  # Reset counter

            # Learning rate warmup (linear warmup from warmup_start_lr to base_lr)
            if self.warmup_iters > 0 and self.iteration < self.warmup_iters:
                warmup_progress = self.iteration / self.warmup_iters
                current_lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * warmup_progress
                for param_group in self.optimizer_loss.param_groups:
                    param_group['lr'] = current_lr
            # Step LR scheduler after warmup
            elif self.scheduler is not None:
                self.scheduler.step()

        # Compute average metrics
        avg_improvement = total_improvement / max(num_checkpoints, 1)
        avg_oracle_loss = total_oracle_loss / max(num_checkpoints, 1)

        return {
            'avg_improvement': avg_improvement,
            'total_improvement': total_improvement,
            'avg_oracle_loss': avg_oracle_loss,
            'total_oracle_loss': total_oracle_loss,
            'num_checkpoints': num_checkpoints,
            'reg_loss': reg_loss.item(),
            'accumulated_steps': self._accumulated_steps,  # For monitoring
        }

    # ==================== End K=3 Methods ====================

    def meta_validate_step_k3(self, checkpoints: List[Dict]) -> Dict[str, Any]:
        """Validation step for K=3 inner loop (no gradient updates).

        Args:
            checkpoints: List of checkpoint dictionaries

        Returns:
            Dictionary with validation metrics
        """
        self.learned_loss.eval()  # Evaluation mode

        # Note: We don't use torch.no_grad() here because _vmapped_inner_loops
        # needs gradients to compute improvement metric (backward through inner loop).
        # Loss parameters won't be updated because we're in eval mode and never call optimizer.step().

        if self.config.get('use_vmap', False):
            # Vmap parallel processing
            improvements, final_bce_losses = self._vmapped_inner_loops(
                checkpoints,
                K=self.config.get('K_inner_steps', 3)
            )
            total_improvement = improvements.sum().item()
            total_oracle_loss = final_bce_losses.sum().item()
        else:
            # Sequential processing by dataset (datasets have unique input dimensions)
            print(f"  [Validation] Processing {len(checkpoints)} checkpoints...", end=" ", flush=True)
            total_improvement = 0.0
            total_oracle_loss = 0.0
            dataset_groups = self._group_by_dataset(checkpoints)

            for dataset_checkpoints in dataset_groups.values():
                # Process entire dataset group at once (all have same input dims)
                improvements, final_bce_losses = self._vmapped_inner_loops(
                    dataset_checkpoints,
                    K=self.config.get('K_inner_steps', 3)
                )

                total_improvement += improvements.sum().item()
                total_oracle_loss += final_bce_losses.sum().item()

            print("Done")

        avg_improvement = total_improvement / len(checkpoints)
        avg_oracle_loss = total_oracle_loss / len(checkpoints)

        self.learned_loss.train()  # Back to training mode

        return {
            'avg_improvement': avg_improvement,
            'total_improvement': total_improvement,
            'avg_oracle_loss': avg_oracle_loss,
            'total_oracle_loss': total_oracle_loss,
            'num_checkpoints': len(checkpoints),
        }

    def load_checkpoint(self, checkpoint_path: str):
        """Load learned loss from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.learned_loss.load_state_dict(checkpoint['state_dict'])
        self.optimizer_loss.load_state_dict(checkpoint['optimizer_state'])
        self.iteration = checkpoint['iteration']
        self.best_meta_loss = checkpoint['best_meta_loss']

        print(f"Loaded checkpoint from: {checkpoint_path}")
        print(f"  Iteration: {self.iteration}")
        print(f"  Best meta-loss: {self.best_meta_loss:.4f}")
