"""Meta-trainer using PyTorch's differentiable=True with non-leaf parameters.

Based on PyTorch issue #150183: https://github.com/pytorch/pytorch/issues/150183

The correct way to use differentiable=True:
1. Convert leaf parameters to non-leaf by doing: param.clone() * 1.0
2. Use optimizer with differentiable=True on these non-leaf parameters
3. The optimizer.step() preserves the computational graph!
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List
import copy

from models.simple_mlp import SimpleMLP
from loss.simple_basis_loss import SimpleBasisLoss
from tasks.task_pool import CheckpointPool


class ToyMetaTrainerDifferentiable:
    """Meta-trainer using differentiable=True (PROPER PyTorch-native solution)."""

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

        self.save_dir = Path(config.get('save_dir', './toy_meta_output_diff'))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nToyMetaTrainerDifferentiable initialized:")
        print(f"  Mode: {self.mode}")
        print(f"  Device: {device}")
        print(f"  Loss parameters: {self.loss_fn.get_num_parameters()}")
        print(f"  Initial loss: {self.loss_fn}")
        print(f"  Using differentiable=True with non-leaf parameters")

    def convert_to_nonleaf(self, model: nn.Module) -> List[torch.Tensor]:
        """Convert model parameters from leaf to non-leaf tensors.

        This is required for differentiable=True to work.

        Args:
            model: Model with leaf parameters

        Returns:
            List of non-leaf parameter tensors
        """
        nonleaf_params = []
        for p in model.parameters():
            # Multiply by 1.0 to create non-leaf tensor with grad_fn
            nonleaf = p.clone() * 1.0
            nonleaf.retain_grad()  # Keep gradients for debugging
            nonleaf_params.append(nonleaf)

        return nonleaf_params

    def functional_forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        params: List[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass using explicit parameter list.

        Args:
            model: Model (for structure)
            x: Input tensor
            params: List of parameter tensors

        Returns:
            Model output
        """
        # For SimpleMLP: Linear -> ReLU -> Linear -> ReLU -> Linear
        hidden_dims = self.config.get('model_hidden_dims', [32, 32])
        num_layers = len(hidden_dims) + 1

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

    def inner_loop_differentiable(
        self,
        model: nn.Module,
        train_loader,
        num_steps: int,
        inner_lr: float,
    ) -> List[torch.Tensor]:
        """Run inner loop with differentiable optimizer.

        Uses differentiable=True with non-leaf parameters.

        Args:
            model: Initial model
            train_loader: Training dataloader
            num_steps: Number of gradient steps
            inner_lr: Learning rate

        Returns:
            List of updated non-leaf parameters (with grad_fn!)
        """
        # Convert leaf parameters to non-leaf
        params = self.convert_to_nonleaf(model)

        # Create differentiable optimizer with non-leaf params
        inner_optimizer_type = self.config.get('inner_optimizer', 'sgd')

        if inner_optimizer_type == 'sgd':
            inner_optimizer = torch.optim.SGD(
                params,
                lr=inner_lr,
                momentum=self.config.get('inner_momentum', 0.9),
                differentiable=True  # Works with non-leaf params!
            )
        elif inner_optimizer_type == 'adam':
            inner_optimizer = torch.optim.Adam(
                params,
                lr=inner_lr,
                differentiable=True  # Works with non-leaf params!
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

            # Forward pass with current parameters
            outputs = self.functional_forward(model, batch_x, params)

            # Compute loss with learned loss function
            loss = self.loss_fn(outputs, batch_y, mode=self.mode)

            # Backward with create_graph=True
            inner_optimizer.zero_grad()
            loss.backward(create_graph=True)

            # Optimizer step - preserves graph with differentiable=True!
            inner_optimizer.step()

            step += 1

        return params  # Now contains updated values with grad_fn intact!

    def evaluate_bce(
        self,
        model: nn.Module,
        val_loader,
        params: List[torch.Tensor],
    ) -> torch.Tensor:
        """Evaluate model with BCE using given parameters.

        Args:
            model: Model structure
            val_loader: Validation dataloader
            params: Non-leaf parameters (with grad_fn)

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

            # Functional forward with parameters
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

            # Inner loop with differentiable optimizer
            updated_params = self.inner_loop_differentiable(
                model=model,
                train_loader=dataloaders['train'],
                num_steps=self.config.get('inner_steps', 3),
                inner_lr=self.config.get('inner_lr', 0.01),
            )

            # Evaluate with updated parameters (gradients preserved!)
            val_bce = self.evaluate_bce(model, dataloaders['val'], updated_params)

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
        print("STARTING META-TRAINING (differentiable=True)")
        print("="*70)
        print(f"Using differentiable=True with non-leaf parameters")
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
