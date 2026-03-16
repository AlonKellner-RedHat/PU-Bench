#!/usr/bin/env python3
"""FAST train-from-scratch meta-learning with DYNAMIC task generation.

Key improvements:
1. No checkpoint pool - tasks generated dynamically at runtime
2. Single full-batch updates (no epoch/batch loops)
3. Fewer gradient steps (3 instead of 10 epochs)
4. Higher learning rate (0.3 instead of 0.1)
5. Reduced meta batch (8 tasks instead of 16)
"""

import torch
import yaml
from pathlib import Path
import numpy as np
from torch.func import functional_call
import torch.nn as nn
import time
from tqdm import tqdm

from models.simple_mlp import SimpleMLP
from loss.hierarchical_pu_loss import HierarchicalPULoss
from tasks.gaussian_task import GaussianBlobTask
from validation_utils import train_from_scratch_validation


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


def generate_random_task_config(config: dict) -> dict:
    """Generate a random task configuration."""
    # Sample from configured ranges
    mean_separations = config.get('mean_separations', [2.0, 2.5, 3.0, 3.5])
    stds = config.get('stds', [0.8, 1.0])
    labeling_freqs = config.get('labeling_freqs', [0.3])
    priors = config.get('priors', [0.5])

    return {
        'num_dimensions': config.get('num_dimensions', 2),
        'mean_separation': float(np.random.choice(mean_separations)),
        'std': float(np.random.choice(stds)),
        'prior': float(np.random.choice(priors)),
        'labeling_freq': float(np.random.choice(labeling_freqs)),
        'num_samples': config.get('num_samples_per_task', 1000),
        'mode': 'pu',
        'negative_labeling_freq': 0.3,
        'seed': np.random.randint(0, 1000000),
    }


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
config['meta_batch_size'] = 8  # Reduced from 16
config['tasks_per_batch'] = 1  # 1 task per sampled config
config['meta_iterations'] = 200

print("="*70)
print("FAST TRAIN-FROM-SCRATCH META-LEARNING (DYNAMIC TASKS)")
print("="*70)
print("Optimizations:")
print("  - Full-batch updates (no epoch/batch loops)")
print("  - 3 gradient steps (vs 10 epochs)")
print("  - Higher LR: 0.3 (vs 0.1)")
print("  - Meta batch: 8 tasks")
print("  - Dynamic task generation (no checkpoint pool)")
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

# Meta-optimizer
meta_optimizer = torch.optim.AdamW(
    loss_fn.parameters(),
    lr=0.0001,
    weight_decay=1e-5
)

# Create validation tasks (fixed for reproducibility)
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

# ITERATION 0 VALIDATION (before any meta-learning)
print("Running iteration 0 validation (before meta-learning)...")
val_results, cached_baselines = train_from_scratch_validation(
    val_tasks, loss_fn, config, device, cached_baselines
)

print()
print("="*70)
print("ITERATION 0 RESULTS (PUDRa initialization, before meta-learning)")
print("="*70)
print(f"  Learned:      {val_results['learned']:.6f}")
print(f"  PUDRa-naive:  {val_results['pudra_naive']:.6f}")
print(f"  VPU-NoMixUp:  {val_results['vpu_nomixup']:.6f}")
print()
diff_from_pudra = abs(val_results['learned'] - val_results['pudra_naive'])
print(f"  Difference from PUDRa baseline: {diff_from_pudra:.6f}")
if diff_from_pudra < 0.02:
    print("  ✓ PUDRa initialization verified!")
else:
    print("  ✗ PUDRa initialization differs from baseline")
print("="*70)
print()

# Training loop
print("Starting FAST meta-learning with dynamic task generation...")
print()
start_time = time.time()

for iteration in tqdm(range(config['meta_iterations']), desc="Meta-training"):
    total_meta_loss = torch.tensor(0.0, device=device)
    num_tasks = 0

    # Generate random tasks on the fly
    for _ in range(config['meta_batch_size']):
        # Generate random task configuration
        task_config = generate_random_task_config(config)

        # Create fresh task
        fresh_task = GaussianBlobTask(**task_config)

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
