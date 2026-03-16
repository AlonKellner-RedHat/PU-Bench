#!/usr/bin/env python3
"""Quick test of VPU stabilization - 40 iterations to verify iter 20 behavior."""

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
    outputs = functional_call(model, params, x_train).squeeze(-1)
    return loss_fn(outputs, y_pu_train, mode='pu')


def fast_train_from_scratch_fomaml(model, train_data, loss_fn, num_steps, lr, device):
    """FOMAML: First-order approximation (no second-order gradients)."""
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
            create_graph=False,  # FOMAML: No second-order gradients
            allow_unused=True
        )

        params = {
            name: param - lr * grad.detach()
            for (name, param), grad in zip(params.items(), grads)
            if grad is not None
        }

    params = {name: param.requires_grad_(True) for name, param in params.items()}
    return params


# Load config
config = load_config('config/toy_gaussian_meta_large_pool.yaml')
device = get_device(config)

config['inner_steps'] = 3
config['inner_lr'] = 0.3
config['meta_batch_size'] = 8
config['meta_iterations'] = 40  # SHORT TEST

print("="*70)
print("VPU STABILIZATION TEST (40 iterations)")
print("="*70)
print("Checking if iteration 20 divergence is prevented")
print(f"Device: {device}")
print("="*70)
print()

# VPU with stabilization
loss_fn = HierarchicalPULoss(
    init_mode='vpu_inspired',
    init_scale=0.01,
    l1_lambda=0.001,
    eps=1e-6  # QUICK WIN 1: Increased epsilon
).to(device)

print("Initial loss (VPU-inspired):")
print(loss_fn)
print()

# Meta-optimizer with lower LR
meta_optimizer = torch.optim.AdamW(
    loss_fn.parameters(),
    lr=5e-5,  # Reduced for stability
    weight_decay=1e-5
)

# Validation tasks
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

# ITERATION 0 VALIDATION
print("Running iteration 0 validation...")
val_results, cached_baselines = train_from_scratch_validation(
    val_tasks, loss_fn, config, device, cached_baselines
)

print()
print("="*70)
print("ITERATION 0 (before meta-learning)")
print("="*70)
print(f"  Learned:      {val_results['learned']:.6f}")
print(f"  VPU-NoMixUp:  {val_results['vpu_nomixup']:.6f}")
diff = abs(val_results['learned'] - val_results['vpu_nomixup'])
print(f"  Difference:   {diff:.6f}")
if diff < 0.02:
    print("  ✓ VPU initialization verified!")
print("="*70)
print()

# Training loop
print("Starting stabilized meta-learning...")
print()
start_time = time.time()

for iteration in tqdm(range(config['meta_iterations']), desc="Meta-training"):
    total_meta_loss = torch.tensor(0.0, device=device)
    num_tasks = 0

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

        # FOMAML inner loop
        final_params = fast_train_from_scratch_fomaml(
            model,
            train_data,
            loss_fn,
            num_steps=config['inner_steps'],
            lr=config['inner_lr'],
            device=device,
        )

        # Meta-objective
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

    # QUICK WIN 3: Stronger gradient clipping
    global_norm = torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), max_norm=0.1)

    for param in loss_fn.parameters():
        if param.grad is not None:
            param.grad.data.clamp_(-0.1, 0.1)

    meta_optimizer.step()

    # Validation every 10 iterations
    if (iteration + 1) % 10 == 0:
        val_results, cached_baselines = train_from_scratch_validation(
            val_tasks, loss_fn, config, device, cached_baselines
        )

        elapsed = time.time() - start_time
        iters_per_min = (iteration + 1) / (elapsed / 60)

        tqdm.write(f"\nIteration {iteration + 1}/{config['meta_iterations']}")
        tqdm.write(f"  Meta-loss:    {avg_meta_loss.item():.6f}")
        tqdm.write(f"  Grad norm:    {global_norm:.6f}")
        tqdm.write(f"  Learned BCE:  {val_results['learned']:.6f}")
        tqdm.write(f"  VPU baseline: {val_results['vpu_nomixup']:.6f}")

        if (iteration + 1) == 20:
            tqdm.write("\n  *** CRITICAL ITERATION 20 ***")
            if val_results['learned'] > 1.0:
                tqdm.write("  ❌ DIVERGENCE DETECTED (like unstabilized version)")
            else:
                tqdm.write("  ✓ STABLE (stabilization working!)")

        tqdm.write("")

elapsed_time = time.time() - start_time
print()
print("="*70)
print("RESULTS")
print("="*70)
print(f"Training time: {elapsed_time:.1f} seconds")
print()
print("Final validation:")
print(f"  Learned:      {val_results['learned']:.6f}")
print(f"  VPU baseline: {val_results['vpu_nomixup']:.6f}")
print()

if val_results['learned'] < 0.35:
    print("✓ STABILIZATION SUCCESSFUL - VPU remained stable!")
elif val_results['learned'] < 0.6:
    print("~ PARTIAL SUCCESS - Some instability but better than before")
else:
    print("❌ STABILIZATION FAILED - Still diverging like unstabilized version")

print("="*70)
