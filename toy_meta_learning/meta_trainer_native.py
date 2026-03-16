"""Meta-trainer using PyTorch's native differentiable=True parameter.

This is the PROPER solution for differentiable optimization in PyTorch.
No external libraries needed - just use the built-in `differentiable=True` parameter.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import copy

from models.simple_mlp import SimpleMLP
from loss.simple_basis_loss import SimpleBasisLoss
from tasks.task_pool import CheckpointPool


class ToyMetaTrainerNative:
    """Meta-trainer using PyTorch's native differentiable optimization."""

    def __init__(self, config: Dict, device: str = 'cpu'):
        self.config = config
        self.device = device

        # Create learnable loss
        self.loss_fn = SimpleBasisLoss(
            init_mode=config.get('loss_init_mode', 'random'),
            init_scale=config.get('loss_init_scale', 0.01),
        ).to(device)

        # Meta-optimizer
        meta_lr = config.get('meta_lr', 0.001)
        self.meta_optimizer = torch.optim.Adam(
            self.loss_fn.parameters(),
            lr=meta_lr,
        )

        # Create checkpoint pool
        self.checkpoint_pool = CheckpointPool(config)
        print("Creating checkpoint pool...")
        self.checkpoint_pool.create_checkpoint_pool(device=device)

        self.mode = config.get('mode', 'pu')

        self.save_dir = Path(config.get('save_dir', './toy_meta_output_native'))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nToyMetaTrainerNative initialized:")
        print(f"  Mode: {self.mode}")
        print(f"  Device: {device}")
        print(f"  Loss parameters: {self.loss_fn.get_num_parameters()}")
        print(f"  Initial loss: {self.loss_fn}")
        print(f"  Using PyTorch native differentiable=True")

    def inner_loop_with_gradients(
        self,
        model: nn.Module,
        train_loader,
        num_steps: int,
        inner_lr: float,
    ) -> nn.Module:
        """Run inner loop with differentiable optimizer (PROPER SOLUTION).

        Uses PyTorch's native `differentiable=True` parameter to preserve
        the computational graph through optimizer.step() calls.

        Args:
            model: Initial model
            train_loader: Training dataloader
            num_steps: Number of gradient steps
            inner_lr: Learning rate

        Returns:
            model: Updated model (gradients preserved!)
        """
        model.train()

        # Create differentiable optimizer (KEY: differentiable=True)
        inner_optimizer_type = self.config.get('inner_optimizer', 'sgd')

        if inner_optimizer_type == 'sgd':
            inner_optimizer = torch.optim.SGD(
                model.parameters(),
                lr=inner_lr,
                momentum=self.config.get('inner_momentum', 0.9),
                differentiable=True  # CRITICAL: Preserves computational graph!
            )
        elif inner_optimizer_type == 'adam':
            inner_optimizer = torch.optim.Adam(
                model.parameters(),
                lr=inner_lr,
                differentiable=True  # CRITICAL: Preserves computational graph!
            )
        else:
            raise ValueError(f"Unknown inner_optimizer: {inner_optimizer_type}")

        # Inner loop training
        step = 0
        for batch_x, batch_y in train_loader:
            if step >= num_steps:
                break

            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass
            outputs = model(batch_x)

            # Compute loss with learned loss function
            loss = self.loss_fn(outputs, batch_y, mode=self.mode)

            # Backward pass with create_graph=True to enable second-order gradients
            inner_optimizer.zero_grad()
            loss.backward(create_graph=True)

            # Update model parameters (graph preserved because differentiable=True!)
            inner_optimizer.step()

            step += 1

        return model

    def evaluate_bce(
        self,
        model: nn.Module,
        val_loader,
    ) -> torch.Tensor:
        """Evaluate model with BCE (meta-objective).

        Args:
            model: Model to evaluate
            val_loader: Validation dataloader

        Returns:
            BCE loss (scalar tensor with grad_fn)
        """
        bce_fn = nn.BCEWithLogitsLoss()

        total_loss = torch.tensor(0.0, device=self.device)
        total_samples = 0

        for batch in val_loader:
            if len(batch) == 3:
                batch_x, batch_y_pu, batch_y_true = batch
                batch_y = batch_y_true
            else:
                batch_x, batch_y = batch

            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass (gradients flow through model params)
            outputs = model(batch_x).squeeze()
            loss = bce_fn(outputs, batch_y)

            total_loss = total_loss + loss * len(batch_x)
            total_samples += len(batch_x)

        return total_loss / total_samples if total_samples > 0 else torch.tensor(0.0, device=self.device)

    def meta_step(self, meta_batch_size: int) -> Dict:
        """Perform one meta-learning step."""
        checkpoint_indices = self.checkpoint_pool.sample_batch(meta_batch_size)

        total_meta_loss = torch.tensor(0.0, device=self.device)
        checkpoint_bces = []

        for ckpt_idx in checkpoint_indices:
            checkpoint, task, dataloaders = self.checkpoint_pool.get_checkpoint(ckpt_idx)

            # Create model
            model = SimpleMLP(
                input_dim=checkpoint['task_config']['num_dimensions'],
                hidden_dims=self.config.get('model_hidden_dims', [32, 32]),
            ).to(self.device)
            model.load_state_dict(checkpoint['model_state_dict'])

            # Inner loop with differentiable optimizer
            model = self.inner_loop_with_gradients(
                model=model,
                train_loader=dataloaders['train'],
                num_steps=self.config.get('inner_steps', 3),
                inner_lr=self.config.get('inner_lr', 0.01),
            )

            # Evaluate with BCE (gradients preserved through inner loop!)
            val_bce = self.evaluate_bce(model, dataloaders['val'])

            total_meta_loss = total_meta_loss + val_bce
            checkpoint_bces.append(val_bce.item())

        # Average meta-loss
        avg_meta_loss = total_meta_loss / len(checkpoint_indices)

        # Meta-optimization
        self.meta_optimizer.zero_grad()
        avg_meta_loss.backward()

        # Log gradient magnitudes
        if self.loss_fn.a1.grad is not None:
            grad_norm = torch.sqrt(
                self.loss_fn.a1.grad**2 +
                self.loss_fn.a2.grad**2 +
                self.loss_fn.a3.grad**2
            ).item()
        else:
            grad_norm = 0.0

        self.meta_optimizer.step()

        return {
            'meta_loss': avg_meta_loss.item(),
            'avg_val_bce': np.mean(checkpoint_bces),
            'std_val_bce': np.std(checkpoint_bces),
            'grad_norm': grad_norm,
        }

    def train(self, num_iterations: int):
        """Run meta-training."""
        print("\n" + "="*70)
        print("STARTING META-TRAINING (NATIVE PyTorch Solution)")
        print("="*70)
        print(f"Using differentiable=True in inner optimizer")
        print()

        for iteration in range(num_iterations):
            metrics = self.meta_step(self.config.get('meta_batch_size', 8))

            if (iteration + 1) % self.config.get('log_freq', 10) == 0:
                print(f"Iteration {iteration + 1}/{num_iterations}")
                print(f"  Meta-loss: {metrics['meta_loss']:.4f}")
                print(f"  Val BCE: {metrics['avg_val_bce']:.4f} ± {metrics['std_val_bce']:.4f}")
                print(f"  Grad norm: {metrics['grad_norm']:.6f}")
                print(f"  Loss params: {self.loss_fn}")
                print()

            if (iteration + 1) % self.config.get('save_freq', 50) == 0:
                self.save_checkpoint(iteration + 1)

        print("="*70)
        print("META-TRAINING COMPLETED")
        print("="*70)
        print(f"Final: {self.loss_fn}")
        print(f"Optimal (BCE): a1=0, a2=0, a3=-1")

    def save_checkpoint(self, iteration: int):
        """Save checkpoint."""
        path = self.save_dir / f"loss_params_iter{iteration:04d}.pt"
        torch.save({
            'iteration': iteration,
            'loss_state_dict': self.loss_fn.state_dict(),
            'config': self.config,
        }, path)
        print(f"  Saved: {path}")
