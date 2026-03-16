#!/usr/bin/env python3
"""Train with asymmetric 6-parameter basis loss.

Tests whether meta-learning discovers asymmetry between positive and negative classes,
or converges to symmetric solution (f_p ≈ f_n).
"""

import torch
import yaml
from pathlib import Path
import numpy as np
from torch.func import functional_call, grad

from models.simple_mlp import SimpleMLP
from loss.asymmetric_basis_loss import AsymmetricBasisLoss
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

device = get_device(config)
mode = 'pn'

# Create asymmetric loss
loss_fn = AsymmetricBasisLoss(init_mode='random', init_scale=0.01).to(device)

# Meta-optimizer
meta_lr = config.get('meta_lr', 0.001)
meta_optimizer = torch.optim.Adam(loss_fn.parameters(), lr=meta_lr)

# Create checkpoint pool
pool = CheckpointPool(config)
print("Creating checkpoint pool...")
pool.create_checkpoint_pool(device=device)

print("="*70)
print("TRAINING ASYMMETRIC 6-PARAMETER LOSS")
print("="*70)
print(f"Device: {device}")
print(f"Initial loss:\n{loss_fn}")
print()


def compute_task_loss(model, params, x, y, loss_fn, mode):
    outputs = functional_call(model, params, x)
    return loss_fn(outputs, y, mode=mode)


def inner_loop_step(model, params, x, y, loss_fn, mode, lr):
    grads = grad(lambda m, p, x, y: compute_task_loss(m, p, x, y, loss_fn, mode), argnums=1)(
        model, params, x, y
    )
    return {name: param - lr * grads[name] for name, param in params.items()}


def inner_loop(model, train_loader, loss_fn, mode, num_steps, lr, device):
    params = dict(model.named_parameters())
    step = 0
    for batch_x, batch_y in train_loader:
        if step >= num_steps:
            break
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        params = inner_loop_step(model, params, batch_x, batch_y, loss_fn, mode, lr)
        step += 1
    return params


def evaluate_bce(model, params, val_loader, device):
    import torch.nn as nn
    bce_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_samples = 0

    for batch in val_loader:
        if len(batch) == 3:
            batch_x, batch_y_pu, batch_y_true = batch
            batch_y = batch_y_true
        else:
            batch_x, batch_y = batch

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        outputs = functional_call(model, params, batch_x).squeeze()
        loss = bce_fn(outputs, batch_y)

        total_loss += loss.item() * len(batch_x)
        total_samples += len(batch_x)

    return total_loss / total_samples if total_samples > 0 else 0.0


# Training loop
print("Starting meta-training...")
print()

for iteration in range(config['meta_iterations']):
    # Sample checkpoints
    checkpoint_indices = [i % len(pool.checkpoints) for i in range(config['meta_batch_size'])]

    total_meta_loss = torch.tensor(0.0, device=device)

    for ckpt_idx in checkpoint_indices:
        checkpoint, task, dataloaders = pool.get_checkpoint(ckpt_idx)

        # Create model
        model = SimpleMLP(
            input_dim=checkpoint['task_config']['num_dimensions'],
            hidden_dims=config.get('model_hidden_dims', [32, 32]),
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Inner loop
        adapted_params = inner_loop(
            model=model,
            train_loader=dataloaders['train'],
            loss_fn=loss_fn,
            mode=mode,
            num_steps=config.get('inner_steps', 3),
            lr=config.get('inner_lr', 0.01),
            device=device,
        )

        # Evaluate - need to keep gradients!
        import torch.nn as nn
        bce_fn = nn.BCEWithLogitsLoss()

        # Get validation batch and compute BCE with gradients
        val_batch = next(iter(dataloaders['val']))
        if len(val_batch) == 3:
            batch_x, batch_y_pu, batch_y_true = val_batch
            batch_y = batch_y_true
        else:
            batch_x, batch_y = val_batch

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        outputs = functional_call(model, adapted_params, batch_x).squeeze()
        val_bce = bce_fn(outputs, batch_y)

        total_meta_loss = total_meta_loss + val_bce

    # Average and optimize
    avg_meta_loss = total_meta_loss / len(checkpoint_indices)

    meta_optimizer.zero_grad()
    avg_meta_loss.backward()
    meta_optimizer.step()

    # Log
    if (iteration + 1) % config['log_freq'] == 0:
        symmetry = loss_fn.get_symmetry_measure()
        print(f"Iteration {iteration + 1}/{config['meta_iterations']}")
        print(f"  Meta-loss: {avg_meta_loss.item():.6f}")
        print(f"  Symmetry measure: {symmetry:.6f}")
        print(f"  {loss_fn}")
        print()

print("="*70)
print("FINAL RESULTS")
print("="*70)
print()
print(f"Final loss:\n{loss_fn}")
print()

# Analyze symmetry
symmetry = loss_fn.get_symmetry_measure()
print(f"Symmetry measure: {symmetry:.6f}")
print()

if symmetry < 0.1:
    print("✅ SYMMETRIC: f_p ≈ f_n")
    print("   Meta-learning discovered that the same function works for both.")
else:
    print("🔍 ASYMMETRIC: f_p ≠ f_n")
    print("   Meta-learning found distinct functions for positives and negatives!")
print()

# Compare to symmetric baseline
print("Comparing to optimal configurations:")
print()

# Test configurations
from loss.simple_basis_loss import SimpleBasisLoss

configs = [
    ("Asymmetric (learned)", None),  # Current asymmetric loss
    ("Symmetric BCE", (0, 0, -1)),
    ("Symmetric Learned (3-param)", (0, -0.95, -0.97)),
]

results = []

for name, params in configs:
    if params is None:
        # Use current asymmetric loss
        test_loss = loss_fn
    else:
        # Create symmetric loss
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

        adapted_params = inner_loop(model, dataloaders['train'], test_loss, mode, 3, 0.01, device)
        total_bce += evaluate_bce(model, adapted_params, dataloaders['val'], device)

    avg_bce = total_bce / len(pool.checkpoints)
    results.append((name, avg_bce))
    print(f"{name:30s}: Val BCE = {avg_bce:.6f}")

print()
print("="*70)
print("INTERPRETATION")
print("="*70)
print()

asym_result = results[0][1]
sym_bce_result = results[1][1]
sym_learned_result = results[2][1]

print(f"Asymmetric (6-param):       {asym_result:.6f}")
print(f"Symmetric Learned (3-param): {sym_learned_result:.6f}")
print(f"Symmetric BCE:              {sym_bce_result:.6f}")
print()

if asym_result < sym_learned_result:
    improvement = (sym_learned_result - asym_result) / sym_learned_result * 100
    print(f"✅ Asymmetric BEATS symmetric by {improvement:.2f}%!")
    print("   The extra 3 parameters discovered useful asymmetric structure.")
elif asym_result > sym_learned_result:
    print("⚠️  Asymmetric doesn't improve over symmetric.")
    print("   Either:")
    print("   1. The problem is truly symmetric")
    print("   2. Need more iterations to find asymmetric optimum")
    print("   3. The 3 extra parameters led to overfitting")
else:
    print("≈ Asymmetric ≈ Symmetric")
    print("   The learned functions converged to symmetric solution.")
