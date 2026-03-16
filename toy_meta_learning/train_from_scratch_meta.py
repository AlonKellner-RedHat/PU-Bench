#!/usr/bin/env python3
"""Meta-learning PU loss with TRAIN-FROM-SCRATCH objective.

Key differences from previous approach:
1. Inner loop trains models from RANDOM initialization (not checkpoints)
2. Meta-objective = BCE after full training (not few-shot adaptation)
3. Checkpoints define TASK CONFIGURATIONS (not model states)
4. Higher inner LR with fewer steps for efficiency
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


def train_from_scratch_inner_loop(model, dataloaders, loss_fn, inner_epochs, inner_lr, device):
    """Train model from scratch using functional API for gradient tracking.

    Args:
        model: SimpleMLP model (used for structure, not weights)
        dataloaders: Dict with 'train' dataloader
        loss_fn: Learned PU loss
        inner_epochs: Number of training epochs
        inner_lr: Learning rate for inner loop
        device: Device to train on

    Returns:
        Final model parameters after training
    """
    # Start with fresh random parameters
    params = {name: param.clone().detach().requires_grad_(True)
              for name, param in model.named_parameters()}

    # Train for specified epochs
    for epoch in range(inner_epochs):
        for batch in dataloaders['train']:
            x = batch[0].to(device)
            y_pu = batch[2].to(device)

            # Compute loss with current params
            outputs = functional_call(model, params, x).squeeze(-1)
            loss = loss_fn(outputs, y_pu, mode='pu')

            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                params.values(),
                create_graph=True,  # Need graph for meta-gradient
                allow_unused=True
            )

            # Update parameters
            params = {
                name: param - inner_lr * grad
                for (name, param), grad in zip(params.items(), grads)
            }

    return params


# Load config
config_path = 'config/toy_gaussian_meta_large_pool.yaml'
config = load_config(config_path)
device = get_device(config)

# Override config for train-from-scratch
config['inner_epochs'] = 10  # Reduced from 50
config['inner_lr'] = 0.1  # Increased from 0.01 for faster convergence
config['meta_batch_size'] = 8  # Reduced from 48 for computational efficiency
config['tasks_per_checkpoint'] = 2  # Task parallelism
config['meta_iterations'] = 200  # Reduced from 500 (each iteration is more expensive)

print("="*70)
print("TRAIN-FROM-SCRATCH META-LEARNING")
print("="*70)
print("Meta-objective: BCE after training models from random initialization")
print(f"Device: {device}")
print(f"Inner epochs: {config['inner_epochs']}")
print(f"Inner LR: {config['inner_lr']}")
print(f"Meta batch size: {config['meta_batch_size']}")
print(f"Tasks per checkpoint: {config['tasks_per_checkpoint']}")
print(f"Meta iterations: {config['meta_iterations']}")
print("="*70)
print()

# Create learnable loss with PUDRa-naive initialization
loss_fn = HierarchicalPULoss(
    init_mode='pudra_inspired',
    init_scale=0.01,
    l1_lambda=0.001  # Lower L1 to not destroy the good initialization
).to(device)

print("Initial loss:")
print(loss_fn)
print()

# Meta-optimizer with lower LR (we start from good initialization)
meta_optimizer = torch.optim.AdamW(
    loss_fn.parameters(),
    lr=0.0001,  # Much lower than before
    weight_decay=1e-5
)

# Load checkpoint pool (we use it for task configurations only)
pool = CheckpointPool(config)
if not pool.load_checkpoint_pool():
    print("Creating checkpoint pool...")
    pool.create_checkpoint_pool(device=device)

print(f"Loaded {len(pool.checkpoints)} task configurations")
print()

# Create validation tasks
print("Creating validation tasks...")
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
print(f"Created {len(val_tasks)} validation tasks")
print()

# Baseline cache
cached_baselines = None

# Training loop
print("Starting train-from-scratch meta-learning...")
print()

start_time = time.time()

for iteration in tqdm(range(config['meta_iterations']), desc="Meta-training"):
    # Sample checkpoints (task configurations)
    checkpoint_indices = pool.sample_batch(config['meta_batch_size'])

    total_meta_loss = torch.tensor(0.0, device=device)
    num_tasks_processed = 0

    for ckpt_idx in checkpoint_indices:
        # Get task configuration from checkpoint
        checkpoint, _, _ = pool.get_checkpoint(ckpt_idx)
        task_config = checkpoint['task_config'].copy()

        # Remove training_method and seed (we'll use fresh seeds)
        task_config.pop('training_method', None)
        task_config.pop('seed', None)

        # Create multiple fresh tasks with this configuration
        for task_seed in range(config['tasks_per_checkpoint']):
            # Create fresh task
            fresh_task = GaussianBlobTask(
                **task_config,
                seed=np.random.randint(0, 100000)
            )

            # Get dataloaders
            dataloaders = fresh_task.get_dataloaders(
                batch_size=128,  # Larger batches for efficiency
                num_train=1000,
                num_val=500,
                num_test=500,
            )

            # Create fresh model (random initialization)
            model = SimpleMLP(
                input_dim=2,
                hidden_dims=config.get('model_hidden_dims', [32, 32]),
            ).to(device)

            # Inner loop: Train from scratch with learned loss
            final_params = train_from_scratch_inner_loop(
                model,
                dataloaders,
                loss_fn,
                inner_epochs=config['inner_epochs'],
                inner_lr=config['inner_lr'],
                device=device,
            )

            # Meta-objective: Evaluate on validation with BCE
            bce_fn = nn.BCEWithLogitsLoss()
            val_loss = torch.tensor(0.0, device=device)

            for val_batch in dataloaders['val']:
                x_val = val_batch[0].to(device)
                y_true_val = val_batch[1].to(device)

                outputs_val = functional_call(model, final_params, x_val).squeeze(-1)
                val_loss += bce_fn(outputs_val, y_true_val)

            val_loss /= len(dataloaders['val'])
            total_meta_loss = total_meta_loss + val_loss
            num_tasks_processed += 1

    # Average meta-loss
    avg_meta_loss = total_meta_loss / num_tasks_processed

    # Meta-update
    meta_optimizer.zero_grad()
    avg_meta_loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), max_norm=1.0)

    meta_optimizer.step()

    # Validation every 20 iterations
    if (iteration + 1) % 20 == 0:
        # Quick train-from-scratch validation
        from validation_utils import train_from_scratch_validation

        val_results, cached_baselines = train_from_scratch_validation(
            val_tasks, loss_fn, config, device, cached_baselines
        )

        tqdm.write(f"\nIteration {iteration + 1}/{config['meta_iterations']}")
        tqdm.write(f"  Training meta-loss:             {avg_meta_loss.item():.6f}")
        tqdm.write(f"  --- Train-from-Scratch Validation ---")
        tqdm.write(f"  Learned BCE:                    {val_results['learned']:.6f}")
        tqdm.write(f"  Oracle BCE (cached):            {val_results['oracle']:.6f}")
        tqdm.write(f"  Naive BCE (cached):             {val_results['naive']:.6f}")
        tqdm.write(f"  PUDRa-naive BCE (cached):       {val_results['pudra_naive']:.6f}")
        tqdm.write(f"  VPU-NoMixUp BCE (cached):       {val_results['vpu_nomixup']:.6f}")

        # Show current parameters
        params = loss_fn.get_parameters().detach().cpu().numpy()
        near_zero = np.sum(np.abs(params) < 0.01)
        tqdm.write(f"  Sparsity: {near_zero}/27 params near zero ({near_zero/27*100:.1f}%)")
        tqdm.write("")

print()
print("="*70)
print("FINAL RESULTS")
print("="*70)
print(f"\nFinal learned PU loss:")
print(loss_fn)

elapsed_time = time.time() - start_time
print(f"\nTraining time: {elapsed_time/60:.1f} minutes")
print("="*70)
