#!/usr/bin/env python3
"""Meta-learning with Neural PU Loss (random initialization).

Testing the neural network-based loss with random initialization
to see what structures it learns through meta-learning.
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
from loss.neural_pu_loss import NeuralPULoss
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
    """Fast training with full-batch updates."""
    params = {name: param.clone().detach().requires_grad_(True)
              for name, param in model.named_parameters()}

    x_train, _, y_pu_train = train_data
    x_train = x_train.to(device)
    y_pu_train = y_pu_train.to(device)

    for step in range(num_steps):
        loss = compute_loss_step(model, params, x_train, y_pu_train, loss_fn)

        grads = torch.autograd.grad(
            loss,
            params.values(),
            create_graph=True,
            allow_unused=True
        )

        params = {
            name: param - lr * grad
            for (name, param), grad in zip(params.items(), grads)
            if grad is not None
        }

    return params


# Load config
config = load_config('config/toy_gaussian_meta_large_pool.yaml')
device = get_device(config)

# Optimized hyperparameters
config['inner_steps'] = 3
config['inner_lr'] = 0.3
config['meta_batch_size'] = 8
config['meta_iterations'] = 200

print("="*70)
print("NEURAL PU LOSS META-LEARNING (RANDOM INITIALIZATION)")
print("="*70)
print("Configuration:")
print("  - Loss: NeuralPULoss")
print("  - Hidden dim: 64 (768 parameters)")
print("  - Initialization: random_normal")
print("  - L0.5 regularization: 0.001")
print("  - Inner steps: 3")
print("  - Inner LR: 0.3")
print("  - Meta batch: 8 tasks")
print("  - Meta iterations: 200")
print(f"  - Device: {device}")
print("="*70)
print()

# Create Neural PU Loss with RANDOM initialization
loss_fn = NeuralPULoss(
    hidden_dim=64,
    eps=1e-7,
    l05_lambda=0.001,  # Encourage sparsity
    init_mode='random_normal',
    init_scale=0.1,  # Small random initialization
).to(device)

print("Initial loss:")
print(loss_fn)
print()

# Meta-optimizer
meta_optimizer = torch.optim.AdamW(
    loss_fn.parameters(),
    lr=1e-4,
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
print("ITERATION 0 RESULTS (random initialization, before meta-learning)")
print("="*70)
print(f"  Learned:      {val_results['learned']:.6f}")
print(f"  PUDRa-naive:  {val_results['pudra_naive']:.6f}")
print(f"  VPU-NoMixUp:  {val_results['vpu_nomixup']:.6f}")
print()
print(f"  Starting point is random - expect poor performance initially")
print("="*70)
print()

# Training loop
print("Starting meta-learning with random initialization...")
print()
start_time = time.time()

for iteration in tqdm(range(config['meta_iterations']), desc="Meta-training"):
    total_meta_loss = torch.tensor(0.0, device=device)
    num_tasks = 0

    # Generate random tasks on the fly
    for _ in range(config['meta_batch_size']):
        task_config = generate_random_task_config(config)
        fresh_task = GaussianBlobTask(**task_config)

        dataloaders = fresh_task.get_dataloaders(
            batch_size=1000,
            num_train=1000,
            num_val=500,
            num_test=500,
        )

        train_batch = next(iter(dataloaders['train']))
        train_data = (train_batch[0], train_batch[1], train_batch[2])

        model = SimpleMLP(
            input_dim=2,
            hidden_dims=config.get('model_hidden_dims', [32, 32]),
        ).to(device)

        # Inner loop training
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

        # Compute weight statistics
        weights = loss_fn.linear.weight.detach().cpu().numpy()
        bias = loss_fn.linear.bias.detach().cpu().numpy()
        all_params = np.concatenate([weights.flatten(), bias])

        near_zero = np.sum(np.abs(all_params) < 0.01)
        sparsity_pct = near_zero / len(all_params) * 100

        tqdm.write(f"\nIteration {iteration + 1}/{config['meta_iterations']}")
        tqdm.write(f"  Training meta-loss:             {avg_meta_loss.item():.6f}")
        tqdm.write(f"  Speed: {iters_per_min:.1f} iters/min")
        tqdm.write(f"  --- Validation ---")
        tqdm.write(f"  Learned:      {val_results['learned']:.6f}")
        tqdm.write(f"  PUDRa-naive:  {val_results['pudra_naive']:.6f}")
        tqdm.write(f"  VPU-NoMixUp:  {val_results['vpu_nomixup']:.6f}")
        tqdm.write(f"  --- Loss Parameters ---")
        tqdm.write(f"  Sparsity: {near_zero}/{len(all_params)} ({sparsity_pct:.1f}%)")
        tqdm.write(f"  Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
        tqdm.write(f"  Bias range: [{bias.min():.4f}, {bias.max():.4f}]")
        tqdm.write("")

elapsed_time = time.time() - start_time
print()
print("="*70)
print("FINAL RESULTS")
print("="*70)
print(f"Training time: {elapsed_time/60:.1f} minutes")
print(f"Speed: {config['meta_iterations']/(elapsed_time/60):.1f} iterations/min")
print()

# Final statistics
print("Final learned loss:")
print(loss_fn)
print()

weights = loss_fn.linear.weight.detach().cpu().numpy()
bias = loss_fn.linear.bias.detach().cpu().numpy()
all_params = np.concatenate([weights.flatten(), bias])

print("Parameter statistics:")
print(f"  Sparsity: {np.sum(np.abs(all_params) < 0.01)}/{len(all_params)} ({np.sum(np.abs(all_params) < 0.01)/len(all_params)*100:.1f}%)")
print(f"  Mean absolute value: {np.abs(all_params).mean():.6f}")
print(f"  Median absolute value: {np.median(np.abs(all_params)):.6f}")
print(f"  Max absolute value: {np.abs(all_params).max():.6f}")
print()

# Performance summary
init_learned = 0.0  # Will be filled from logs
final_learned = val_results['learned']
pudra_baseline = val_results['pudra_naive']
vpu_baseline = val_results['vpu_nomixup']

print("Performance summary:")
print(f"  Final (learned):      {final_learned:.6f}")
print(f"  PUDRa baseline:       {pudra_baseline:.6f}")
print(f"  VPU baseline:         {vpu_baseline:.6f}")
print()

if final_learned < pudra_baseline:
    improvement = (pudra_baseline - final_learned) / pudra_baseline * 100
    print(f"  ✓ BEATS PUDRa by {improvement:.1f}%")
elif final_learned < pudra_baseline + 0.02:
    print(f"  ~ MATCHES PUDRa (within 2%)")
else:
    gap = (final_learned - pudra_baseline) / pudra_baseline * 100
    print(f"  ✗ Below PUDRa by {gap:.1f}%")

print("="*70)
