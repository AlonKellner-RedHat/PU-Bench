#!/usr/bin/env python3
"""FAST train-from-scratch meta-learning with optimizations.

Key optimizations:
1. Single full-batch updates (no epoch/batch loops)
2. Fewer gradient steps (3 instead of 10 epochs)
3. Higher learning rate (0.3 instead of 0.1)
4. Reduced meta batch (8 tasks instead of 16)
5. Less frequent validation
"""

import torch
import yaml
from pathlib import Path
import numpy as np
from torch.func import functional_call, grad
import torch.nn as nn
import time
from tqdm import tqdm
import sys

from models.simple_mlp import SimpleMLP
from loss.hierarchical_pu_loss import HierarchicalPULoss
from loss.baseline_losses import PUDRaNaiveLoss, VPUNoMixUpLoss
from tasks.task_pool import CheckpointPool
from tasks.gaussian_task import GaussianBlobTask


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if len(config) == 1 and isinstance(list(config.values())[0], dict):
        config = list(config.values())[0]
    return config


def get_device(config: dict) -> str:
    device_config = config.get('device', 'auto')
    if device_config == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    return device_config


def compute_loss_step(model, params, x_train, y_pu_train, loss_fn):
    """Compute loss for one training step."""
    outputs = functional_call(model, params, x_train).squeeze(-1)
    return loss_fn(outputs, y_pu_train, mode='pu')


def fast_train_from_scratch(model, train_data, loss_fn, num_steps, lr, device):
    """Fast training with full-batch updates.

    Args:
        model: SimpleMLP model
        train_data: Tuple of (X, y_true, y_pu) tensors
        loss_fn: Learned PU loss
        num_steps: Number of gradient steps
        lr: Learning rate
        device: Device

    Returns:
        Final model parameters
    """
    # Start with fresh random parameters
    params = {name: param.clone().detach().requires_grad_(True)
              for name, param in model.named_parameters()}

    x_train, _, y_pu_train = train_data
    x_train = x_train.to(device)
    y_pu_train = y_pu_train.to(device)

    # Single-batch gradient descent
    for step in range(num_steps):
        # Forward pass
        loss = compute_loss_step(model, params, x_train, y_pu_train, loss_fn)

        # Compute gradients
        grads = torch.autograd.grad(
            loss,
            params.values(),
            create_graph=True,
            allow_unused=True
        )

        # Update parameters
        params = {
            name: param - lr * grad
            for (name, param), grad in zip(params.items(), grads)
        }

    return params


# Load config
config = load_config('config/toy_gaussian_meta_large_pool.yaml')
device = get_device(config)

# Optimized hyperparameters
config['inner_steps'] = 3  # Reduced from 10 epochs
config['inner_lr'] = 0.3  # Increased from 0.1
config['meta_batch_size'] = 4  # Reduced from 8
config['tasks_per_checkpoint'] = 2  # Keep at 2
config['meta_iterations'] = 200

print("="*70)
print("FAST TRAIN-FROM-SCRATCH META-LEARNING")
print("="*70)
print("Optimizations:")
print("  - Full-batch updates (no epoch/batch loops)")
print("  - 3 gradient steps (vs 10 epochs)")
print("  - Higher LR: 0.3 (vs 0.1)")
print("  - Smaller batch: 8 tasks (vs 16)")
print(f"Device: {device}")
print(f"PyTorch version: {torch.__version__}")
print("="*70)
print()

# Create learnable loss
loss_fn = HierarchicalPULoss(
    init_mode='pudra_inspired',
    init_scale=0.01,
    l1_lambda=0.001
).to(device)

print("Initial loss:")
print(loss_fn)
print()

# NOTE: Cannot use torch.compile() on loss function because:
# - Meta-learning uses higher-order gradients (create_graph=True)
# - torch.compile does not support double backwards
# - Error: "torch.compile with aot_autograd does not currently support double backward"
print("Note: torch.compile() disabled for meta-learning compatibility")
print("(Higher-order gradients required for differentiating through training)")
print()

# Meta-optimizer
meta_optimizer = torch.optim.AdamW(
    loss_fn.parameters(),
    lr=0.0001,
    weight_decay=1e-5
)

# Load checkpoint pool
pool = CheckpointPool(config)
if not pool.load_checkpoint_pool():
    pool.create_checkpoint_pool(device=device)

print(f"Loaded {len(pool.checkpoints)} task configurations")
print()

# Create validation tasks
val_tasks = []
for i in range(3):
    task = GaussianBlobTask(
        num_dimensions=2,
        mean_separation=2.5,
        std=1.0,
        prior=0.5,
        labeling_freq=0.3,
        num_samples=1000,
        seed=9000 + i,
        mode='pu',
        negative_labeling_freq=0.3,
    )
    val_tasks.append(task)

cached_baselines = None

# Training loop
print("Starting FAST meta-learning...")
print()
start_time = time.time()

for iteration in tqdm(range(config['meta_iterations']), desc="Meta-training"):
    # Sample checkpoints
    checkpoint_indices = pool.sample_batch(config['meta_batch_size'])

    total_meta_loss = torch.tensor(0.0, device=device)
    num_tasks = 0

    for ckpt_idx in checkpoint_indices:
        # Get task configuration
        checkpoint, _, _ = pool.get_checkpoint(ckpt_idx)
        task_config = checkpoint['task_config'].copy()
        task_config.pop('training_method', None)
        task_config.pop('seed', None)

        # Create fresh tasks
        for _ in range(config['tasks_per_checkpoint']):
            # Create task
            fresh_task = GaussianBlobTask(
                **task_config,
                seed=np.random.randint(0, 100000)
            )

            # Get dataloaders
            dataloaders = fresh_task.get_dataloaders(
                batch_size=1000,  # Single batch = full dataset
                num_train=1000,
                num_val=500,
                num_test=500,
            )

            # Get full training data
            train_batch = next(iter(dataloaders['train']))
            train_data = (train_batch[0], train_batch[1], train_batch[2])

            # Create model
            model = SimpleMLP(
                input_dim=2,
                hidden_dims=config.get('model_hidden_dims', [32, 32]),
            ).to(device)

            # Fast inner loop training
            final_params = fast_train_from_scratch(
                model,
                train_data,
                loss_fn,
                num_steps=config['inner_steps'],
                lr=config['inner_lr'],
                device=device,
            )

            # Meta-objective: Evaluate on validation
            val_batch = next(iter(dataloaders['val']))
            x_val = val_batch[0].to(device)
            y_true_val = val_batch[1].to(device)

            outputs_val = functional_call(model, final_params, x_val).squeeze(-1)
            bce_fn = nn.BCEWithLogitsLoss()
            val_loss = bce_fn(outputs_val, y_true_val)

            total_meta_loss = total_meta_loss + val_loss
            num_tasks += 1

    # Meta-update
    avg_meta_loss = total_meta_loss / num_tasks

    meta_optimizer.zero_grad()
    avg_meta_loss.backward()
    torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), max_norm=1.0)
    meta_optimizer.step()

    # Validation every 20 iterations
    if (iteration + 1) % 20 == 0:
        from validation_utils import train_from_scratch_validation

        val_results, cached_baselines = train_from_scratch_validation(
            val_tasks, loss_fn, config, device, cached_baselines
        )

        elapsed = time.time() - start_time
        iters_per_min = (iteration + 1) / (elapsed / 60)

        tqdm.write(f"\nIteration {iteration + 1}/{config['meta_iterations']}")
        tqdm.write(f"  Training meta-loss:             {avg_meta_loss.item():.6f}")
        tqdm.write(f"  Speed: {iters_per_min:.1f} iters/min")
        tqdm.write(f"  --- Validation ---")
        tqdm.write(f"  Learned:      {val_results['learned']:.6f}")
        tqdm.write(f"  PUDRa-naive:  {val_results['pudra_naive']:.6f}")
        tqdm.write(f"  VPU-NoMixUp:  {val_results['vpu_nomixup']:.6f}")

        params = loss_fn.get_parameters().detach().cpu().numpy()
        near_zero = np.sum(np.abs(params) < 0.01)
        tqdm.write(f"  Sparsity: {near_zero}/27 ({near_zero/27*100:.1f}%)")
        tqdm.write("")

elapsed_time = time.time() - start_time
print()
print("="*70)
print("FINAL RESULTS")
print("="*70)
print(f"Training time: {elapsed_time/60:.1f} minutes")
print(f"Speed: {config['meta_iterations']/(elapsed_time/60):.1f} iterations/min")
print()
print("Final learned loss:")
print(loss_fn)
print("="*70)
