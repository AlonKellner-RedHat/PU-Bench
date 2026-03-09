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

        # Initialize learned loss
        self.learned_loss = MonotonicBasisLoss(
            num_repetitions=config.get('num_repetitions', 3),
            num_fourier=config.get('num_fourier', 16),
            use_prior=config.get('use_prior', True),
            l1_weight=config.get('l1_weight', 1e-4),
            l2_weight=config.get('l2_weight', 1e-3),
            oracle_mode=config.get('oracle_mode', False),
            init_scale=config.get('init_scale', 0.01)
        ).to(self.device)

        # Optimizer for loss parameters only
        self.optimizer_loss = torch.optim.Adam(
            self.learned_loss.parameters(),
            lr=config.get('meta_lr', 1e-4)
        )

        # Caches: treat models and loaders as reusable resources
        self.model_cache: Dict[str, nn.Module] = {}  # dataset -> model
        self.loader_cache: Dict[str, tuple] = {}  # task_id -> (train_loader, val_loader)

        # State
        self.iteration = 0
        self.best_meta_loss = float('inf')

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

        print(f"MetaTrainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Loss parameters: {sum(p.numel() for p in self.learned_loss.parameters())}")
        print(f"  Meta-objective: {self.meta_objective}")
        print(f"  Meta LR: {config.get('meta_lr', 1e-4)}")
        print(f"  L1 weight: {config.get('l1_weight', 1e-4)}")
        print(f"  L2 weight: {config.get('l2_weight', 1e-3)}")
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
                'optimizer': self.config.get('optimizer', 'adam'),
                'lr': self.config.get('lr', 3e-4),
                'weight_decay': self.config.get('weight_decay', 1e-4)
            }

            # Use a default prior (will be set in loss anyway)
            model = select_model('monotonic_basis', model_params, prior=0.5)
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
                'target_prevalence': prior
            }

            # Prepare loaders
            train_loader, val_loader, test_loader, actual_prior, _ = prepare_loaders(
                dataset_name=dataset,
                data_config=data_config,
                batch_size=self.config.get('batch_size', 128),
                method='monotonic_basis'
            )

            self.loader_cache[task_id] = (train_loader, val_loader)
            print(f"Created loaders for task: {task_id}")

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
                x, y = batch[0].to(self.device), batch[1].to(self.device)

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
                x_train, t_train = next(iter(train_loader))
                x_train = x_train.to(self.device)
                t_train = t_train.to(self.device)

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
                x_val, y_val = next(iter(val_loader))
                x_val = x_val.to(self.device)
                y_val = y_val.to(self.device)

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
                'target_prevalence': prior
            }

            # Prepare loaders
            train_loader, _, _, actual_prior, _ = prepare_loaders(
                dataset_name=dataset,
                data_config=data_config,
                batch_size=self.config.get('batch_size', 128),
                method='monotonic_basis'
            )

            # Create fresh model
            model_params = {
                'optimizer': self.config.get('optimizer', 'adam'),
                'lr': self.config.get('lr', 3e-4),
                'weight_decay': self.config.get('weight_decay', 1e-4)
            }
            model = select_model('monotonic_basis', model_params, prior=actual_prior)
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

    def _group_by_architecture(self, meta_batch: List[Dict]) -> Dict[str, List[Dict]]:
        """Group checkpoints by architecture for vmapping.

        Args:
            meta_batch: List of checkpoint dictionaries

        Returns:
            Dictionary mapping architecture name -> list of checkpoints
        """
        groups = {}
        for cp in meta_batch:
            dataset = cp.get('dataset', self.pool.get_dataset_from_task_id(cp['task_id']))
            arch = DATASET_TO_ARCH.get(dataset, 'unknown')

            if arch not in groups:
                groups[arch] = []
            groups[arch].append(cp)

        return groups

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
            x_train, t_train = next(iter(train_loader))
            x_train_list.append(x_train)
            t_train_list.append(t_train)

            # Sample batch from val loader (PN data)
            x_val, y_val = next(iter(val_loader))
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
        """Process multiple checkpoints with K=3 inner loop using vmap.

        This is the core K=3 inner loop meta-learning function. For each checkpoint:
        1. Load initial model state
        2. Run K gradient steps using learned loss (with create_graph=True)
        3. Measure improvement on validation data
        4. Return improvement (gradients flow through K steps to loss params)

        Args:
            checkpoints: List of checkpoint dicts (all same architecture)
            K: Number of inner loop gradient steps

        Returns:
            improvements: [num_ckpts] tensor with gradients to loss params
        """
        if not checkpoints:
            return torch.tensor(0.0, device=self.device)

        # Get architecture
        dataset = checkpoints[0].get('dataset',
                                     self.pool.get_dataset_from_task_id(checkpoints[0]['task_id']))
        arch = DATASET_TO_ARCH.get(dataset, 'unknown')

        # Stack initial states
        stacked_states = stack_state_dicts([cp['model_state'] for cp in checkpoints])

        # Prepare batched data
        x_train_batched, t_train_batched, x_val_batched, y_val_batched, priors = \
            self._prepare_batched_data(checkpoints)

        priors_tensor = torch.tensor(priors, device=self.device)

        # Get model template
        model_template = self._get_model(dataset)

        # Define single-checkpoint K-step inner loop
        def single_inner_loop(state_dict_values, x_train, t_train, x_val, y_val, prior):
            """K gradient steps for one checkpoint."""
            # Create fresh model
            model = deepcopy(model_template)

            # Load state dict (need to reconstruct dict from tensor values)
            state_dict = {}
            value_idx = 0
            for key in model.state_dict().keys():
                param_shape = model.state_dict()[key].shape
                param_size = model.state_dict()[key].numel()

                # Extract values for this parameter
                param_flat = state_dict_values[value_idx:value_idx + param_size]
                param = param_flat.view(param_shape)

                state_dict[key] = param
                value_idx += param_size

            model.load_state_dict(state_dict)
            model.train()  # Enable gradients

            # Fresh optimizer
            inner_lr = self.config.get('inner_lr', 1e-3)
            optimizer = torch.optim.Adam(model.parameters(), lr=inner_lr)

            # Baseline validation loss (before K steps)
            with torch.no_grad():
                val_out_before = model(x_val).view(-1)
                val_loss_before = F.binary_cross_entropy_with_logits(val_out_before, y_val.float())

            # K gradient steps with learned loss
            for k in range(K):
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
            improvement = val_loss_before - val_loss_after

            return improvement

        # Stack state dict values for vmap
        # Flatten all state dicts into vectors
        state_vectors = []
        for state_dict in [cp['model_state'] for cp in checkpoints]:
            values = torch.cat([param.flatten() for param in state_dict.values()])
            state_vectors.append(values)

        state_vectors_batched = torch.stack(state_vectors).to(self.device)

        # Process checkpoints sequentially (vmap over state dicts is complex)
        # For now, use sequential processing but with K=3 inner loop
        improvements = []
        for i in range(len(checkpoints)):
            imp = single_inner_loop(
                state_vectors_batched[i],
                x_train_batched[i],
                t_train_batched[i],
                x_val_batched[i],
                y_val_batched[i],
                priors_tensor[i]
            )
            improvements.append(imp)

        improvements = torch.stack(improvements)

        return improvements

    def meta_train_step_k3(self, meta_batch: List[Dict]) -> Dict[str, float]:
        """Single meta-training iteration with K=3 inner loop.

        This implements the external architecture cycle:
        LeNet → MLP_Tabular → MLP_Text → optimizer step

        Args:
            meta_batch: List of checkpoint dictionaries

        Returns:
            Dictionary with training metrics
        """
        # Group by architecture
        arch_groups = self._group_by_architecture(meta_batch)

        # Get cycle order from config
        cycle_order = self.config.get('architecture_cycle_order',
                                      ['LeNet', 'MLP_Tabular', 'MLP_Text'])

        # Zero gradients
        self.optimizer_loss.zero_grad()

        total_improvement = 0.0
        num_checkpoints = 0

        # External cycle through architectures
        for arch_name in cycle_order:
            checkpoints = arch_groups.get(arch_name, [])
            if not checkpoints:
                continue

            # Vmap K=3 inner loops for this architecture
            improvements = self._vmapped_inner_loops(
                checkpoints,
                K=self.config.get('K_inner_steps', 3)
            )

            # Accumulate (negative for minimization: maximize improvement = minimize -improvement)
            arch_improvement = improvements.sum()
            (-arch_improvement).backward()  # Accumulate gradients

            total_improvement += arch_improvement.item()
            num_checkpoints += len(checkpoints)

        # Add regularization
        reg_loss = self.learned_loss.compute_regularization()
        reg_loss.backward()

        # Single optimizer step with accumulated gradients
        self.optimizer_loss.step()

        # Compute average improvement
        avg_improvement = total_improvement / max(num_checkpoints, 1)

        return {
            'avg_improvement': avg_improvement,
            'total_improvement': total_improvement,
            'num_checkpoints': num_checkpoints,
            'reg_loss': reg_loss.item()
        }

    # ==================== End K=3 Methods ====================

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
