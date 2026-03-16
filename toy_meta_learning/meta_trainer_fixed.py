"""Fixed meta-trainer with functional parameter updates.

KEY FIX: Replace optimizer.step() with manual parameter updates to preserve
         the computational graph for meta-gradients.
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


class ToyMetaTrainerFixed:
    """Fixed meta-trainer using functional optimization."""

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

        self.save_dir = Path(config.get('save_dir', './toy_meta_output_fixed'))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nToyMetaTrainerFixed initialized:")
        print(f"  Mode: {self.mode}")
        print(f"  Device: {device}")
        print(f"  Loss parameters: {self.loss_fn.get_num_parameters()}")
        print(f"  Initial loss: {self.loss_fn}")
        print(f"  Using FUNCTIONAL optimization (graph-preserving)")

    def functional_forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        params: List[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass using explicit parameters.

        Args:
            model: Model (for structure, params not used)
            x: Input tensor
            params: List of parameter tensors [weight1, bias1, weight2, bias2, ...]

        Returns:
            Model output
        """
        # For SimpleMLP: Linear -> ReLU -> Linear -> ReLU -> Linear
        # Manually construct forward pass
        hidden_dims = self.config.get('model_hidden_dims', [32, 32])
        num_layers = len(hidden_dims) + 1

        # Extract parameters
        param_idx = 0
        h = x

        for layer_idx in range(num_layers):
            weight = params[param_idx]
            bias = params[param_idx + 1]
            param_idx += 2

            # Linear transformation
            h = h @ weight.T + bias

            # ReLU activation (except last layer)
            if layer_idx < num_layers - 1:
                h = torch.relu(h)

        return h

    def inner_loop_functional(
        self,
        model: nn.Module,
        train_loader,
        num_steps: int,
        inner_lr: float,
    ) -> Tuple[nn.Module, List[torch.Tensor]]:
        """Run inner loop with FUNCTIONAL parameter updates (graph-preserving).

        Args:
            model: Initial model
            train_loader: Training dataloader
            num_steps: Number of gradient steps
            inner_lr: Learning rate

        Returns:
            (model, updated_params): Model and list of updated parameter tensors
        """
        # Get initial parameters as list
        params = [p.clone() for p in model.parameters()]

        model.train()

        step = 0
        for batch_x, batch_y in train_loader:
            if step >= num_steps:
                break

            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass with current parameters
            outputs = self.functional_forward(model, batch_x, params)

            # Compute loss
            loss = self.loss_fn(outputs, batch_y, mode=self.mode)

            # Compute gradients w.r.t. parameters (preserves graph!)
            grads = torch.autograd.grad(
                loss,
                params,
                create_graph=True,  # KEY: This preserves the computational graph
            )

            # Manual SGD update: param = param - lr * grad
            params = [p - inner_lr * g for p, g in zip(params, grads)]

            step += 1

        # Update model with final parameters (for evaluation)
        with torch.no_grad():
            for p_model, p_new in zip(model.parameters(), params):
                p_model.copy_(p_new)

        return model, params

    def evaluate_bce_functional(
        self,
        model: nn.Module,
        val_loader,
        params: List[torch.Tensor],
    ) -> torch.Tensor:
        """Evaluate model with BCE using functional forward (for meta-gradient).

        Args:
            model: Model structure
            val_loader: Validation dataloader
            params: Updated parameters (with computational graph)

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

            # Functional forward (maintains graph through params)
            outputs = self.functional_forward(model, batch_x, params).squeeze()
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

            # Inner loop with functional updates
            model, updated_params = self.inner_loop_functional(
                model=model,
                train_loader=dataloaders['train'],
                num_steps=self.config.get('inner_steps', 3),
                inner_lr=self.config.get('inner_lr', 0.01),
            )

            # Evaluate with functional forward (preserves graph!)
            val_bce = self.evaluate_bce_functional(model, dataloaders['val'], updated_params)

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
        print("STARTING META-TRAINING (FIXED VERSION)")
        print("="*70)
        print(f"Using functional optimization to preserve gradients")
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
