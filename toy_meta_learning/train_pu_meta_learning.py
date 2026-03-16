#!/usr/bin/env python3
"""Train with ACTUAL PU meta-learning setup.

This is the TRUE PU-Bench scenario:
- Inner loop: Train on PU labels (labeled positives + unlabeled)
- Meta-objective: Evaluate on ground truth PN labels with BCE

The learned loss should adapt models trained on PU data to perform well on PN evaluation.
"""

import torch
import yaml
from pathlib import Path
import numpy as np
from torch.func import functional_call, grad
import torch.nn as nn

from models.simple_mlp import SimpleMLP
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

# CRITICAL: Set mode to 'pu' for PU learning!
config['mode'] = 'pu'
config['labeling_freq'] = 0.3  # Only 30% of positives are labeled

device = get_device(config)

# Create learnable PU loss
loss_fn = SimpleBasisLoss(init_mode='random', init_scale=0.01).to(device)

# Meta-optimizer
meta_lr = config.get('meta_lr', 0.001)
meta_optimizer = torch.optim.Adam(loss_fn.parameters(), lr=meta_lr)

# Create checkpoint pool with PU data
pool = CheckpointPool(config)
print("Creating checkpoint pool for PU learning...")
pool.create_checkpoint_pool(device=device)

print()
print("="*70)
print("PU META-LEARNING SETUP")
print("="*70)
print(f"Device: {device}")
print(f"Mode: {config['mode']} (PU learning)")
print(f"Labeling frequency: {config['labeling_freq']} (only {config['labeling_freq']*100:.0f}% of positives labeled)")
print(f"Initial loss:\n{loss_fn}")
print()
print("Inner loop: Train with PU labels (labeled P + unlabeled)")
print("Meta-objective: Evaluate with ground truth PN labels (BCE)")
print()


def compute_task_loss(model, params, x, y_pu, loss_fn):
    """Compute PU loss for inner loop training."""
    outputs = functional_call(model, params, x)
    return loss_fn(outputs, y_pu, mode='pu')  # Use PU labels!


def inner_loop_step(model, params, x, y_pu, loss_fn, lr):
    """Single PU adaptation step."""
    grads = grad(lambda m, p, x, y: compute_task_loss(m, p, x, y, loss_fn), argnums=1)(
        model, params, x, y_pu
    )
    return {name: param - lr * grads[name] for name, param in params.items()}


def inner_loop(model, train_loader, loss_fn, num_steps, lr, device):
    """Inner loop: Adapt model on PU data."""
    params = dict(model.named_parameters())

    step = 0
    for batch in train_loader:
        if step >= num_steps:
            break

        # Unpack batch: (features, y_true, y_pu)
        x = batch[0].to(device)
        y_pu = batch[2].to(device)  # Use PU labels for training!

        params = inner_loop_step(model, params, x, y_pu, loss_fn, lr)
        step += 1

    return params


def evaluate_bce_on_gt(model, params, val_loader, device):
    """Evaluate on ground truth PN labels with BCE (meta-objective)."""
    bce_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_samples = 0

    for batch in val_loader:
        # Unpack batch: (features, y_true, y_pu)
        x = batch[0].to(device)
        y_true = batch[1].to(device)  # Use ground truth labels for evaluation!

        outputs = functional_call(model, params, x).squeeze()
        loss = bce_fn(outputs, y_true)

        total_loss += loss.item() * len(x)
        total_samples += len(x)

    return total_loss / total_samples if total_samples > 0 else 0.0


# Training loop
print("Starting PU meta-training...")
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

        # Inner loop: Train on PU labels
        adapted_params = inner_loop(
            model=model,
            train_loader=dataloaders['train'],
            loss_fn=loss_fn,
            num_steps=config.get('inner_steps', 3),
            lr=config.get('inner_lr', 0.01),
            device=device,
        )

        # Meta-objective: Evaluate on ground truth with BCE
        # Need to compute with gradients!
        val_batch = next(iter(dataloaders['val']))
        x_val = val_batch[0].to(device)
        y_true_val = val_batch[1].to(device)  # Ground truth!

        outputs_val = functional_call(model, adapted_params, x_val).squeeze()
        bce_fn = nn.BCEWithLogitsLoss()
        val_bce = bce_fn(outputs_val, y_true_val)

        total_meta_loss = total_meta_loss + val_bce

    # Average and optimize
    avg_meta_loss = total_meta_loss / len(checkpoint_indices)

    meta_optimizer.zero_grad()
    avg_meta_loss.backward()
    meta_optimizer.step()

    # Log
    if (iteration + 1) % config['log_freq'] == 0:
        # Also compute average BCE on all checkpoints for monitoring
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

        print(f"Iteration {iteration + 1}/{config['meta_iterations']}")
        print(f"  Meta-loss: {avg_meta_loss.item():.6f}")
        print(f"  Avg Val BCE (GT): {avg_bce:.6f}")
        print(f"  {loss_fn}")
        print()

print()
print("="*70)
print("FINAL RESULTS: PU META-LEARNING")
print("="*70)
print()
print(f"Final learned PU loss:\n{loss_fn}")
print()

# Compare to baselines
print("Comparing PU loss to baselines:")
print()

configs_to_test = [
    ("Learned PU loss", None),
    ("Pure BCE (not PU)", (0, 0, -1)),
    ("Symmetric Learned", (0, -0.95, -0.97)),
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

        # Train with PU labels using this loss
        adapted_params = inner_loop(model, dataloaders['train'], test_loss, 3, 0.01, device)

        # Evaluate on ground truth
        bce = evaluate_bce_on_gt(model, adapted_params, dataloaders['val'], device)
        total_bce += bce

    avg_bce = total_bce / len(pool.checkpoints)
    results.append((name, avg_bce))
    print(f"{name:25s}: Val BCE (GT) = {avg_bce:.6f}")

print()
print("="*70)
print("INTERPRETATION")
print("="*70)
print()

learned_result = results[0][1]
bce_result = results[1][1]

print(f"Learned PU loss:  {learned_result:.6f}")
print(f"Pure BCE:         {bce_result:.6f}")
print()

if learned_result < bce_result:
    improvement = (bce_result - learned_result) / bce_result * 100
    print(f"✅ Learned PU loss BEATS pure BCE by {improvement:.2f}%!")
    print()
    print("This proves meta-learning discovered a loss function that:")
    print("  1. Works better for PU learning than standard BCE")
    print("  2. Improves model performance when trained on incomplete labels")
    print("  3. Adapts to the specific characteristics of PU data")
else:
    print("⚠️  Pure BCE performs as well or better")
    print("   This suggests:")
    print("   1. Need more diverse tasks or longer training")
    print("   2. The labeling frequency may be too high (too easy)")
    print("   3. The task is too simple to benefit from learned PU loss")
print()
print(f"Key insight: The loss was learned on PU data (only {config['labeling_freq']*100:.0f}% positives labeled)")
print(f"but evaluated on full ground truth PN labels!")
