"""Meta-trainer using PyTorch's NATIVE torch.func API (THE MODERN SOLUTION).

This is the PROPER way to do meta-learning in PyTorch 2.0+:
- torch.func.functional_call: Pass arbitrary params to a model without modifying state
- torch.func.grad: Functional gradient computation (like JAX)
- torch.func.vmap: Vectorize across tasks automatically

No external libraries needed. This is the actively maintained, official PyTorch solution.
Replaces higher, learn2learn, torchopt - all of which are now obsolete.

Reference: https://pytorch.org/docs/stable/func.html
"""

import torch
import torch.nn as nn
from torch.func import functional_call, grad
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import copy

from models.simple_mlp import SimpleMLP
from loss.simple_basis_loss import SimpleBasisLoss
from tasks.task_pool import CheckpointPool


class ToyMetaTrainerTorchFunc:
    """Meta-trainer using PyTorch's native torch.func API (Modern Solution)."""

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

        self.save_dir = Path(config.get('save_dir', './toy_meta_output_torch_func'))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nToyMetaTrainerTorchFunc initialized:")
        print(f"  Mode: {self.mode}")
        print(f"  Device: {device}")
        print(f"  Loss parameters: {self.loss_fn.get_num_parameters()}")
        print(f"  Initial loss: {self.loss_fn}")
        print(f"  Using torch.func (Modern PyTorch 2.0+ native solution)")

    def compute_task_loss(
        self,
        model: nn.Module,
        params: Dict[str, torch.Tensor],
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss using functional_call with given parameters.

        This is the key: functional_call lets us use arbitrary parameters
        without modifying the model's internal state.

        Args:
            model: Model (for structure, params not used)
            params: Dictionary of parameters to use
            x: Input tensor
            y: Labels

        Returns:
            Loss value
        """
        # functional_call uses 'params' instead of model's internal state
        outputs = functional_call(model, params, x)

        # Compute loss with learned loss function
        loss = self.loss_fn(outputs, y, mode=self.mode)

        return loss

    def inner_loop_step(
        self,
        model: nn.Module,
        params: Dict[str, torch.Tensor],
        x: torch.Tensor,
        y: torch.Tensor,
        inner_lr: float,
    ) -> Dict[str, torch.Tensor]:
        """Single inner loop adaptation step using torch.func.grad.

        This is the MODERN way: use functional gradients, no create_graph tricks.

        Args:
            model: Model structure
            params: Current parameters
            x: Training batch
            y: Training labels
            inner_lr: Learning rate

        Returns:
            Updated parameters (with computational graph intact!)
        """
        # Compute functional gradients w.r.t. parameters
        # This is like torch.autograd.grad but cleaner and more functional
        grads = grad(self.compute_task_loss, argnums=1)(model, params, x, y)

        # Stateless gradient descent
        adapted_params = {
            name: param - inner_lr * grads[name]
            for name, param in params.items()
        }

        return adapted_params

    def inner_loop(
        self,
        model: nn.Module,
        train_loader,
        num_steps: int,
        inner_lr: float,
    ) -> Dict[str, torch.Tensor]:
        """Run full inner loop with K adaptation steps.

        Args:
            model: Model structure
            train_loader: Training dataloader
            num_steps: Number of adaptation steps
            inner_lr: Learning rate

        Returns:
            Adapted parameters (with grad_fn preserved!)
        """
        # Start with model's current parameters
        params = dict(model.named_parameters())

        # Inner loop adaptation
        step = 0
        for batch_x, batch_y in train_loader:
            if step >= num_steps:
                break

            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Functional gradient step
            params = self.inner_loop_step(model, params, batch_x, batch_y, inner_lr)

            step += 1

        return params

    def evaluate_bce(
        self,
        model: nn.Module,
        params: Dict[str, torch.Tensor],
        val_loader,
    ) -> torch.Tensor:
        """Evaluate using BCE with given parameters.

        Args:
            model: Model structure
            params: Parameters to evaluate
            val_loader: Validation dataloader

        Returns:
            BCE loss (with grad_fn for meta-learning!)
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

            # Use functional_call with adapted parameters
            outputs = functional_call(model, params, batch_x).squeeze()
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

            # Inner loop adaptation using torch.func
            adapted_params = self.inner_loop(
                model=model,
                train_loader=dataloaders['train'],
                num_steps=self.config.get('inner_steps', 3),
                inner_lr=self.config.get('inner_lr', 0.01),
            )

            # Meta-objective: evaluate adapted parameters
            val_bce = self.evaluate_bce(model, adapted_params, dataloaders['val'])

            total_meta_loss = total_meta_loss + val_bce
            checkpoint_bces.append(val_bce.item())

        # Average meta-loss
        avg_meta_loss = total_meta_loss / len(checkpoint_indices)

        # Meta-optimization (standard)
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
        print("STARTING META-TRAINING (torch.func)")
        print("="*70)
        print(f"Using PyTorch's native torch.func API")
        print(f"This is the MODERN, official solution for meta-learning")
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
