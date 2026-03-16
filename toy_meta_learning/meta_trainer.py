"""Simplified meta-trainer for toy meta-learning example."""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List
import copy

from models.simple_mlp import SimpleMLP
from loss.simple_basis_loss import SimpleBasisLoss
from tasks.task_pool import CheckpointPool


class ToyMetaTrainer:
    """Simplified meta-trainer for learning PN loss parameters.

    Meta-learning loop:
    1. Sample batch of checkpoints
    2. For each checkpoint:
       a. Load model weights
       b. Inner loop: train model with learnable PN loss for K steps
       c. Evaluate: compute BCE on validation
    3. Meta-update: update loss parameters to minimize val BCE
    """

    def __init__(self, config: Dict, device: str = 'cpu'):
        """Initialize meta-trainer.

        Args:
            config: Configuration dictionary
            device: Device to run on
        """
        self.config = config
        self.device = device

        # Create learnable loss
        self.loss_fn = SimpleBasisLoss(
            init_mode=config.get('loss_init_mode', 'random'),
            init_scale=config.get('loss_init_scale', 0.01),
        ).to(device)

        # Meta-optimizer (optimizes loss parameters)
        meta_lr = config.get('meta_lr', 0.001)
        meta_optimizer_type = config.get('meta_optimizer', 'adam')

        if meta_optimizer_type == 'adam':
            self.meta_optimizer = torch.optim.Adam(
                self.loss_fn.parameters(),
                lr=meta_lr,
            )
        elif meta_optimizer_type == 'sgd':
            self.meta_optimizer = torch.optim.SGD(
                self.loss_fn.parameters(),
                lr=meta_lr,
                momentum=config.get('meta_momentum', 0.9),
            )
        else:
            raise ValueError(f"Unknown meta_optimizer: {meta_optimizer_type}")

        # Create checkpoint pool
        self.checkpoint_pool = CheckpointPool(config)
        print("Creating checkpoint pool...")
        self.checkpoint_pool.create_checkpoint_pool(device=device)

        # Training mode (PU or PN)
        self.mode = config.get('mode', 'pu')

        # Log settings
        self.save_dir = Path(config.get('save_dir', './toy_meta_output'))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nToyMetaTrainer initialized:")
        print(f"  Mode: {self.mode}")
        print(f"  Device: {device}")
        print(f"  Loss parameters: {self.loss_fn.get_num_parameters()}")
        print(f"  Initial loss: {self.loss_fn}")
        print(f"  Checkpoint pool size: {len(self.checkpoint_pool)}")

    def inner_loop(
        self,
        model: nn.Module,
        train_loader,
        num_steps: int,
        inner_lr: float,
    ) -> nn.Module:
        """Run inner loop training.

        Args:
            model: Model to train
            train_loader: Training dataloader
            num_steps: Number of gradient steps
            inner_lr: Inner loop learning rate

        Returns:
            Updated model
        """
        # Create inner optimizer
        inner_optimizer_type = self.config.get('inner_optimizer', 'sgd')

        if inner_optimizer_type == 'sgd':
            inner_optimizer = torch.optim.SGD(
                model.parameters(),
                lr=inner_lr,
                momentum=self.config.get('inner_momentum', 0.0),
            )
        elif inner_optimizer_type == 'adam':
            inner_optimizer = torch.optim.Adam(
                model.parameters(),
                lr=inner_lr,
            )
        else:
            raise ValueError(f"Unknown inner_optimizer: {inner_optimizer_type}")

        model.train()

        # Training loop
        step = 0
        while step < num_steps:
            for batch_x, batch_y in train_loader:
                if step >= num_steps:
                    break

                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                inner_optimizer.zero_grad()

                # Forward pass
                outputs = model(batch_x)

                # Compute loss with learnable PN loss
                loss = self.loss_fn(outputs, batch_y, mode=self.mode)

                # Backward pass
                loss.backward(create_graph=True)  # create_graph for meta-gradients

                inner_optimizer.step()

                step += 1

        return model

    def evaluate_bce(self, model: nn.Module, val_loader) -> float:
        """Evaluate model with BCE on validation set.

        Args:
            model: Model to evaluate
            val_loader: Validation dataloader (with true labels)

        Returns:
            BCE loss value
        """
        model.eval()
        bce_fn = nn.BCEWithLogitsLoss()

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    # val_loader with PU labels and true labels
                    batch_x, batch_y_pu, batch_y_true = batch
                    batch_y = batch_y_true
                else:
                    # Standard val_loader with true labels
                    batch_x, batch_y = batch

                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = model(batch_x).squeeze()
                loss = bce_fn(outputs, batch_y)

                total_loss += loss.item() * len(batch_x)
                total_samples += len(batch_x)

        return total_loss / total_samples if total_samples > 0 else 0.0

    def meta_step(self, meta_batch_size: int) -> Dict:
        """Perform one meta-learning step.

        Args:
            meta_batch_size: Number of checkpoints to sample

        Returns:
            Dictionary with metrics
        """
        # Sample checkpoint batch
        checkpoint_indices = self.checkpoint_pool.sample_batch(meta_batch_size)

        # Meta-objective: average val BCE across checkpoints
        total_meta_loss = 0.0
        checkpoint_bces = []

        for ckpt_idx in checkpoint_indices:
            # Get checkpoint
            checkpoint, task, dataloaders = self.checkpoint_pool.get_checkpoint(ckpt_idx)

            # Create fresh model
            model = SimpleMLP(
                input_dim=checkpoint['task_config']['num_dimensions'],
                hidden_dims=self.config.get('model_hidden_dims', [32, 32]),
                activation=self.config.get('model_activation', 'relu'),
            ).to(self.device)

            # Load checkpoint weights
            model.load_state_dict(checkpoint['model_state_dict'])

            # Inner loop: train with learnable loss
            model = self.inner_loop(
                model=model,
                train_loader=dataloaders['train'],
                num_steps=self.config.get('inner_steps', 3),
                inner_lr=self.config.get('inner_lr', 0.01),
            )

            # Evaluate with BCE on validation
            val_bce = self.evaluate_bce(model, dataloaders['val'])

            # Create computational graph for meta-gradient
            # Re-evaluate in train mode to get gradients
            model.train()
            bce_fn = nn.BCEWithLogitsLoss()
            meta_loss = 0.0
            n_batches = 0

            for batch in dataloaders['val']:
                if len(batch) == 3:
                    batch_x, batch_y_pu, batch_y_true = batch
                    batch_y = batch_y_true
                else:
                    batch_x, batch_y = batch

                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = model(batch_x).squeeze()
                loss = bce_fn(outputs, batch_y)
                meta_loss += loss
                n_batches += 1

            meta_loss = meta_loss / n_batches if n_batches > 0 else torch.tensor(0.0)

            total_meta_loss += meta_loss
            checkpoint_bces.append(val_bce)

        # Average meta-loss across batch
        avg_meta_loss = total_meta_loss / len(checkpoint_indices)

        # Meta-optimization step
        self.meta_optimizer.zero_grad()
        avg_meta_loss.backward()
        self.meta_optimizer.step()

        # Return metrics
        return {
            'meta_loss': avg_meta_loss.item(),
            'avg_val_bce': np.mean(checkpoint_bces),
            'std_val_bce': np.std(checkpoint_bces),
            'num_checkpoints': len(checkpoint_indices),
        }

    def train(self, num_iterations: int):
        """Run meta-training.

        Args:
            num_iterations: Number of meta-iterations
        """
        print("\n" + "="*70)
        print("STARTING META-TRAINING")
        print("="*70)
        print(f"Meta-iterations: {num_iterations}")
        print(f"Meta-batch size: {self.config.get('meta_batch_size', 8)}")
        print(f"Inner steps: {self.config.get('inner_steps', 3)}")
        print(f"Inner LR: {self.config.get('inner_lr', 0.01)}")
        print(f"Meta LR: {self.config.get('meta_lr', 0.001)}")
        print()

        for iteration in range(num_iterations):
            # Meta-step
            metrics = self.meta_step(self.config.get('meta_batch_size', 8))

            # Logging
            if (iteration + 1) % self.config.get('log_freq', 10) == 0:
                print(f"Iteration {iteration + 1}/{num_iterations}")
                print(f"  Meta-loss (val BCE): {metrics['meta_loss']:.4f}")
                print(f"  Avg val BCE: {metrics['avg_val_bce']:.4f} ± {metrics['std_val_bce']:.4f}")
                print(f"  Current loss params: {self.loss_fn}")
                print()

            # Save checkpoint
            if (iteration + 1) % self.config.get('save_freq', 50) == 0:
                self.save_checkpoint(iteration + 1)

        print("="*70)
        print("META-TRAINING COMPLETED")
        print("="*70)
        print(f"Final loss parameters: {self.loss_fn}")
        print(f"Optimal (BCE): a1=0, a2=0, a3=-1")
        print()

    def save_checkpoint(self, iteration: int):
        """Save loss parameters checkpoint.

        Args:
            iteration: Current iteration number
        """
        checkpoint_path = self.save_dir / f"loss_params_iter{iteration:04d}.pt"
        torch.save({
            'iteration': iteration,
            'loss_state_dict': self.loss_fn.state_dict(),
            'meta_optimizer_state_dict': self.meta_optimizer.state_dict(),
            'config': self.config,
        }, checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")
