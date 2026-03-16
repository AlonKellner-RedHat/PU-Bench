#!/usr/bin/env python3
"""Train asymmetric 6-parameter PU loss (f_p for positives, f_u for unlabeled).

This is more natural than symmetric PU because:
- Labeled positives are CLEAN (reliable labels)
- Unlabeled is NOISY (mixture of hidden positives + negatives)

Meta-learning should discover whether to treat them differently!
"""

import torch
import yaml
from pathlib import Path
import numpy as np
from torch.func import functional_call, grad
import torch.nn as nn

from models.simple_mlp import SimpleMLP
from loss.asymmetric_pu_loss import AsymmetricPULoss
from loss.simple_basis_loss import SimpleBasisLoss
from tasks.task_pool import CheckpointPool


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


# Load config
config = load_config('config/toy_gaussian_meta.yaml')
config['mean_separations'] = [2.0, 3.0]
config['checkpoint_seeds'] = [42, 123]
config['checkpoint_epochs'] = [1, 5, 10]
config['meta_iterations'] = 1000
config['meta_batch_size'] = 8
config['log_freq'] = 100
config['mode'] = 'pu'
config['labeling_freq'] = 0.3

device = get_device(config)

# Create asymmetric PU loss
loss_fn = AsymmetricPULoss(init_mode='random', init_scale=0.01).to(device)

# Meta-optimizer
meta_lr = config.get('meta_lr', 0.001)
meta_optimizer = torch.optim.Adam(loss_fn.parameters(), lr=meta_lr)

# Create checkpoint pool
pool = CheckpointPool(config)
print("Creating checkpoint pool for asymmetric PU learning...")
pool.create_checkpoint_pool(device=device)

print()
print("="*70)
print("ASYMMETRIC PU META-LEARNING (6 PARAMETERS)")
print("="*70)
print(f"Device: {device}")
print(f"Mode: PU (30% labeling)")
print(f"Initial loss:\n{loss_fn}")
print()
print("Separate functions:")
print("  f_p(p) for labeled positives (clean labels)")
print("  f_u(1-p) for unlabeled (noisy mixture)")
print()


def compute_task_loss(model, params, x, y_pu, loss_fn):
    """Compute PU loss."""
    outputs = functional_call(model, params, x)
    return loss_fn(outputs, y_pu, mode='pu')


def inner_loop_step(model, params, x, y_pu, loss_fn, lr):
    """Single PU adaptation step."""
    grads = grad(lambda m, p, x, y: compute_task_loss(m, p, x, y, loss_fn), argnums=1)(
        model, params, x, y_pu
    )
    return {name: param - lr * grads[name] for name, param in params.items()}


def inner_loop(model, train_loader, loss_fn, num_steps, lr, device):
    """Inner loop: Adapt on PU data."""
    params = dict(model.named_parameters())

    step = 0
    for batch in train_loader:
        if step >= num_steps:
            break

        x = batch[0].to(device)
        y_pu = batch[2].to(device)

        params = inner_loop_step(model, params, x, y_pu, loss_fn, lr)
        step += 1

    return params


def evaluate_bce_on_gt(model, params, val_loader, device):
    """Evaluate on ground truth with BCE."""
    bce_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_samples = 0

    for batch in val_loader:
        x = batch[0].to(device)
        y_true = batch[1].to(device)

        outputs = functional_call(model, params, x).squeeze()
        loss = bce_fn(outputs, y_true)

        total_loss += loss.item() * len(x)
        total_samples += len(x)

    return total_loss / total_samples if total_samples > 0 else 0.0


# Training loop
print("Starting asymmetric PU meta-training...")
print()

for iteration in range(config['meta_iterations']):
    checkpoint_indices = [i % len(pool.checkpoints) for i in range(config['meta_batch_size'])]

    total_meta_loss = torch.tensor(0.0, device=device)

    for ckpt_idx in checkpoint_indices:
        checkpoint, task, dataloaders = pool.get_checkpoint(ckpt_idx)

        model = SimpleMLP(
            input_dim=checkpoint['task_config']['num_dimensions'],
            hidden_dims=config.get('model_hidden_dims', [32, 32]),
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Inner loop: Train on PU
        adapted_params = inner_loop(
            model=model,
            train_loader=dataloaders['train'],
            loss_fn=loss_fn,
            num_steps=config.get('inner_steps', 3),
            lr=config.get('inner_lr', 0.01),
            device=device,
        )

        # Meta-objective: BCE on ground truth
        val_batch = next(iter(dataloaders['val']))
        x_val = val_batch[0].to(device)
        y_true_val = val_batch[1].to(device)

        outputs_val = functional_call(model, adapted_params, x_val).squeeze()
        bce_fn = nn.BCEWithLogitsLoss()
        val_bce = bce_fn(outputs_val, y_true_val)

        total_meta_loss = total_meta_loss + val_bce

    # Optimize
    avg_meta_loss = total_meta_loss / len(checkpoint_indices)

    meta_optimizer.zero_grad()
    avg_meta_loss.backward()
    meta_optimizer.step()

    # Log
    if (iteration + 1) % config['log_freq'] == 0:
        # Compute average BCE
        total_bce = 0.0
        for ckpt_idx in range(len(pool.checkpoints)):
            checkpoint, task, dataloaders = pool.get_checkpoint(ckpt_idx)
            model = SimpleMLP(
                checkpoint['task_config']['num_dimensions'],
                config.get('model_hidden_dims', [32, 32])
            ).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])

            adapted_params = inner_loop(model, dataloaders['train'], loss_fn, 3, 0.01, device)
            bce = evaluate_bce_on_gt(model, adapted_params, dataloaders['val'], device)
            total_bce += bce

        avg_bce = total_bce / len(pool.checkpoints)
        symmetry = loss_fn.get_symmetry_measure()

        print(f"Iteration {iteration + 1}/{config['meta_iterations']}")
        print(f"  Meta-loss: {avg_meta_loss.item():.6f}")
        print(f"  Avg Val BCE (GT): {avg_bce:.6f}")
        print(f"  Symmetry: {symmetry:.6f}")
        print(f"  {loss_fn}")
        print()

print()
print("="*70)
print("FINAL RESULTS: ASYMMETRIC PU")
print("="*70)
print()
print(f"Final learned asymmetric PU loss:\n{loss_fn}")
print()

symmetry = loss_fn.get_symmetry_measure()
print(f"Symmetry measure: {symmetry:.6f}")
print()

if symmetry < 0.1:
    print("✅ SYMMETRIC: f_p ≈ f_u")
    print("   Meta-learning discovered that the same function works for both.")
else:
    print("🔍 ASYMMETRIC: f_p ≠ f_u")
    print("   Meta-learning found distinct functions for labeled pos and unlabeled!")

print()
print("Comparing to baselines:")
print()

configs_to_test = [
    ("Asymmetric PU (6-param)", None),
    ("Symmetric PU (3-param)", (0, -0.97, -0.95)),
    ("Pure BCE", (0, 0, -1)),
]

results = []

for name, params in configs_to_test:
    if params is None:
        test_loss = loss_fn
    else:
        test_loss = SimpleBasisLoss(init_mode='random').to(device)
        test_loss.a1.data = torch.tensor([params[0]], dtype=torch.float32, device=device)
        test_loss.a2.data = torch.tensor([params[1]], dtype=torch.float32, device=device)
        test_loss.a3.data = torch.tensor([params[2]], dtype=torch.float32, device=device)

    total_bce = 0.0
    for ckpt_idx in range(len(pool.checkpoints)):
        checkpoint, task, dataloaders = pool.get_checkpoint(ckpt_idx)
        model = SimpleMLP(
            checkpoint['task_config']['num_dimensions'],
            config.get('model_hidden_dims', [32, 32])
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        adapted_params = inner_loop(model, dataloaders['train'], test_loss, 3, 0.01, device)
        bce = evaluate_bce_on_gt(model, adapted_params, dataloaders['val'], device)
        total_bce += bce

    avg_bce = total_bce / len(pool.checkpoints)
    results.append((name, avg_bce))
    print(f"{name:30s}: Val BCE (GT) = {avg_bce:.6f}")

print()
print("="*70)
print("INTERPRETATION")
print("="*70)
print()

asym_result = results[0][1]
sym_result = results[1][1]
bce_result = results[2][1]

print(f"Asymmetric PU (6-param): {asym_result:.6f}")
print(f"Symmetric PU (3-param):  {sym_result:.6f}")
print(f"Pure BCE:                {bce_result:.6f}")
print()

if asym_result < sym_result:
    improvement = (sym_result - asym_result) / sym_result * 100
    print(f"✅ Asymmetric BEATS symmetric by {improvement:.2f}%!")
    print()
    print("The 6 parameters discovered useful asymmetric structure:")
    print("  → Labeled positives and unlabeled require different treatment")
    print("  → The extra capacity captures the noise characteristics of PU data")
elif asym_result > sym_result:
    degradation = (asym_result - sym_result) / sym_result * 100
    print(f"⚠️  Asymmetric is {degradation:.2f}% worse than symmetric")
    print()
    print("Possible reasons:")
    print("  1. The labeled pos vs unlabeled distinction isn't beneficial")
    print("  2. Overfitting with 6 parameters on 12 checkpoints")
    print("  3. Need more diverse tasks to benefit from asymmetry")
else:
    print("≈ Asymmetric ≈ Symmetric")
    print("  The extra 3 parameters didn't help or hurt")

print()
print("Key insight: This tests if PU needs DIFFERENT functions for")
print("             clean labeled positives vs noisy unlabeled mixture!")
