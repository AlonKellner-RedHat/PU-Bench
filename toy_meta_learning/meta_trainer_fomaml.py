"""Meta-trainer using First-Order MAML (FOMAML).

Based on PyTorch Lightning tutorial:
https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/12-meta-learning.html

FOMAML is a simpler alternative to full second-order MAML that:
- Uses only first-order gradients (much cheaper)
- Performs almost as well as full MAML
- Allows using standard PyTorch optimizers with a detach trick
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict
import copy

from models.simple_mlp import SimpleMLP
from loss.simple_basis_loss import SimpleBasisLoss
from tasks.task_pool import CheckpointPool


class ToyMetaTrainerFOMAML:
    """Meta-trainer using First-Order MAML (FOMAML)."""

    def __init__(self, config: Dict, device: str = 'cpu'):
        self.config = config
        self.device = device

        # Create learnable loss
        self.loss_fn = SimpleBasisLoss(
            init_mode=config.get('loss_init_mode', 'random'),
            init_scale=config.get('loss_init_scale', 0.01),
        ).to(device)

        # Meta-optimizer (for loss parameters)
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

        self.save_dir = Path(config.get('save_dir', './toy_meta_output_fomaml'))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nToyMetaTrainerFOMAML initialized:")
        print(f"  Mode: {self.mode}")
        print(f"  Device: {device}")
        print(f"  Loss parameters: {self.loss_fn.get_num_parameters()}")
        print(f"  Initial loss: {self.loss_fn}")
        print(f"  Using First-Order MAML (FOMAML)")

    def inner_loop_fomaml(
        self,
        model: nn.Module,
        train_loader,
        num_steps: int,
        inner_lr: float,
    ) -> nn.Module:
        """Run inner loop with FOMAML.

        Key technique: Store initial parameters, run standard optimizer,
        then use detach-and-reattach to preserve first-order gradients.

        Args:
            model: Initial model
            train_loader: Training dataloader
            num_steps: Number of gradient steps
            inner_lr: Learning rate

        Returns:
            Updated model with first-order gradients preserved
        """
        model.train()

        # Store initial parameters (for gradient computation)
        initial_params = {name: param.clone() for name, param in model.named_parameters()}

        # Create standard optimizer (no differentiable=True needed!)
        inner_optimizer_type = self.config.get('inner_optimizer', 'sgd')

        if inner_optimizer_type == 'sgd':
            inner_optimizer = torch.optim.SGD(
                model.parameters(),
                lr=inner_lr,
                momentum=self.config.get('inner_momentum', 0.9),
            )
        elif inner_optimizer_type == 'adam':
            inner_optimizer = torch.optim.Adam(
                model.parameters(),
                lr=inner_lr,
            )
        else:
            raise ValueError(f"Unknown inner_optimizer: {inner_optimizer_type}")

        # Inner loop training (standard backprop)
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

            # Standard backward and update (NO create_graph needed!)
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()

            step += 1

        # FOMAML trick: Detach and reattach
        # This preserves gradients w.r.t. initial params but not inner optimization
        for name, param in model.named_parameters():
            init_param = initial_params[name]
            # Compute delta: final_param - init_param
            delta = param.detach() - init_param
            # Reattach: delta.detach() + init_param
            # Gradients flow through init_param but not through inner loop!
            param.data = delta.detach() + init_param

        return model

    def evaluate_bce(
        self,
        model: nn.Module,
        val_loader,
    ) -> torch.Tensor:
        """Evaluate model with BCE.

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

            # Inner loop with FOMAML
            model = self.inner_loop_fomaml(
                model=model,
                train_loader=dataloaders['train'],
                num_steps=self.config.get('inner_steps', 3),
                inner_lr=self.config.get('inner_lr', 0.01),
            )

            # Evaluate (gradients preserved via detach-and-reattach!)
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
        print("STARTING META-TRAINING (FOMAML)")
        print("="*70)
        print(f"Using First-Order MAML (simplified, fast)")
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
