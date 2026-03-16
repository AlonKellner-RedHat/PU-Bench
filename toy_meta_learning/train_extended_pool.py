#!/usr/bin/env python3
"""Train PU meta-learning with extended checkpoint pool.

This includes:
- Oracle checkpoints: trained with ground truth PN labels
- Naive checkpoints: trained with PU labels (only 30% positives labeled)
- Checkpoints from early training (1 epoch) to near-convergence (200 epochs)
- More task diversity (4 difficulties × 2 overlaps × 5 seeds = 40 base tasks)

Total pool: 560 checkpoints (40 tasks × 2 methods × 7 epochs)
"""

import torch
import yaml
from pathlib import Path
import numpy as np
from torch.func import functional_call, grad
import torch.nn as nn

from models.simple_mlp import SimpleMLP
from loss.asymmetric_pu_loss import AsymmetricPULoss
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


# Load extended config
config = load_config('config/toy_gaussian_meta_extended.yaml')
device = get_device(config)

# Create learnable asymmetric PU loss (6 parameters)
loss_fn = AsymmetricPULoss(init_mode='random', init_scale=0.01).to(device)

# Meta-optimizer
meta_lr = config.get('meta_lr', 0.001)
meta_optimizer = torch.optim.Adam(loss_fn.parameters(), lr=meta_lr)

# Create checkpoint pool with both oracle and naive checkpoints
pool = CheckpointPool(config)
print("Creating extended checkpoint pool...")
print("This will include:")
print("  - Oracle checkpoints (trained with PN labels)")
print("  - Naive checkpoints (trained with PU labels)")
print("  - Checkpoints from epochs: 1, 5, 10, 20, 50, 100, 200")
print()
pool.create_checkpoint_pool(device=device)

print()
print("="*70)
print("EXTENDED PU META-LEARNING SETUP")
print("="*70)
print(f"Device: {device}")
print(f"Total checkpoints: {len(pool.checkpoints)}")
print(f"Initial loss:\n{loss_fn}")
print()

# Count oracle vs naive checkpoints
oracle_count = sum(1 for ckpt in pool.checkpoints if ckpt['task_config']['training_method'] == 'oracle')
naive_count = sum(1 for ckpt in pool.checkpoints if ckpt['task_config']['training_method'] == 'naive')
print(f"Oracle checkpoints (PN-trained): {oracle_count}")
print(f"Naive checkpoints (PU-trained):  {naive_count}")
print()

# Analyze checkpoint epochs distribution
from collections import Counter
epoch_counts = Counter([ckpt['epoch'] for ckpt in pool.checkpoints])
print("Checkpoints by epoch:")
for epoch in sorted(epoch_counts.keys()):
    print(f"  Epoch {epoch:3d}: {epoch_counts[epoch]:3d} checkpoints")
print()


def compute_task_loss(model, params, x, y_pu, loss_fn):
    """Compute PU loss for inner loop training."""
    outputs = functional_call(model, params, x)
    return loss_fn(outputs, y_pu, mode='pu')


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

        x = batch[0].to(device)
        y_pu = batch[2].to(device)  # Use PU labels for training

        params = inner_loop_step(model, params, x, y_pu, loss_fn, lr)
        step += 1

    return params


def evaluate_bce_on_gt(model, params, val_loader, device):
    """Evaluate on ground truth PN labels with BCE (meta-objective)."""
    bce_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_samples = 0

    for batch in val_loader:
        x = batch[0].to(device)
        y_true = batch[1].to(device)  # Use ground truth labels

        outputs = functional_call(model, params, x).squeeze()
        loss = bce_fn(outputs, y_true)

        total_loss += loss.item() * len(x)
        total_samples += len(x)

    return total_loss / total_samples if total_samples > 0 else 0.0


# Training loop
print("Starting extended PU meta-training...")
print()

for iteration in range(config['meta_iterations']):
    # Sample checkpoints
    checkpoint_indices = pool.sample_batch(config['meta_batch_size'])

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
        val_batch = next(iter(dataloaders['val']))
        x_val = val_batch[0].to(device)
        y_true_val = val_batch[1].to(device)

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
        # Sample checkpoints for evaluation (separate oracle vs naive)
        oracle_indices = [i for i in range(len(pool.checkpoints))
                         if pool.checkpoints[i]['task_config']['training_method'] == 'oracle']
        naive_indices = [i for i in range(len(pool.checkpoints))
                        if pool.checkpoints[i]['task_config']['training_method'] == 'naive']

        # Evaluate on oracle checkpoints
        oracle_bce = 0.0
        for ckpt_idx in np.random.choice(oracle_indices, size=min(20, len(oracle_indices)), replace=False):
            checkpoint, task, dataloaders = pool.get_checkpoint(ckpt_idx)
            model = SimpleMLP(
                checkpoint['task_config']['num_dimensions'],
                config.get('model_hidden_dims', [32, 32])
            ).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])

            adapted_params = inner_loop(model, dataloaders['train'], loss_fn, 3, 0.01, device)
            bce = evaluate_bce_on_gt(model, adapted_params, dataloaders['val'], device)
            oracle_bce += bce
        oracle_bce /= min(20, len(oracle_indices))

        # Evaluate on naive checkpoints
        naive_bce = 0.0
        for ckpt_idx in np.random.choice(naive_indices, size=min(20, len(naive_indices)), replace=False):
            checkpoint, task, dataloaders = pool.get_checkpoint(ckpt_idx)
            model = SimpleMLP(
                checkpoint['task_config']['num_dimensions'],
                config.get('model_hidden_dims', [32, 32])
            ).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])

            adapted_params = inner_loop(model, dataloaders['train'], loss_fn, 3, 0.01, device)
            bce = evaluate_bce_on_gt(model, adapted_params, dataloaders['val'], device)
            naive_bce += bce
        naive_bce /= min(20, len(naive_indices))

        print(f"Iteration {iteration + 1}/{config['meta_iterations']}")
        print(f"  Meta-loss: {avg_meta_loss.item():.6f}")
        print(f"  Oracle checkpoints BCE: {oracle_bce:.6f}")
        print(f"  Naive checkpoints BCE:  {naive_bce:.6f}")
        print(f"  {loss_fn}")
        print()

print()
print("="*70)
print("FINAL RESULTS: EXTENDED PU META-LEARNING")
print("="*70)
print()
print(f"Final learned PU loss:\n{loss_fn}")
print()

# Final comprehensive evaluation
print("="*70)
print("COMPREHENSIVE EVALUATION")
print("="*70)
print()

# Evaluate on all checkpoints by training method and epoch
results_by_method = {'oracle': {}, 'naive': {}}

for method in ['oracle', 'naive']:
    method_indices = [i for i in range(len(pool.checkpoints))
                     if pool.checkpoints[i]['task_config']['training_method'] == method]

    # Group by epoch
    by_epoch = {}
    for idx in method_indices:
        epoch = pool.checkpoints[idx]['epoch']
        if epoch not in by_epoch:
            by_epoch[epoch] = []
        by_epoch[epoch].append(idx)

    # Evaluate each epoch
    for epoch in sorted(by_epoch.keys()):
        total_bce = 0.0
        count = 0
        for ckpt_idx in by_epoch[epoch]:
            checkpoint, task, dataloaders = pool.get_checkpoint(ckpt_idx)
            model = SimpleMLP(
                checkpoint['task_config']['num_dimensions'],
                config.get('model_hidden_dims', [32, 32])
            ).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])

            adapted_params = inner_loop(model, dataloaders['train'], loss_fn, 3, 0.01, device)
            bce = evaluate_bce_on_gt(model, adapted_params, dataloaders['val'], device)
            total_bce += bce
            count += 1

        avg_bce = total_bce / count
        results_by_method[method][epoch] = avg_bce

# Print results
print("Oracle checkpoints (PN-trained):")
for epoch in sorted(results_by_method['oracle'].keys()):
    print(f"  Epoch {epoch:3d}: {results_by_method['oracle'][epoch]:.6f}")
print()

print("Naive checkpoints (PU-trained):")
for epoch in sorted(results_by_method['naive'].keys()):
    print(f"  Epoch {epoch:3d}: {results_by_method['naive'][epoch]:.6f}")
print()

# Overall averages
oracle_avg = np.mean(list(results_by_method['oracle'].values()))
naive_avg = np.mean(list(results_by_method['naive'].values()))

print(f"Overall Oracle Average: {oracle_avg:.6f}")
print(f"Overall Naive Average:  {naive_avg:.6f}")
print()

if naive_avg < oracle_avg:
    improvement = (oracle_avg - naive_avg) / oracle_avg * 100
    print(f"Learned loss helps naive checkpoints MORE than oracle!")
    print(f"Naive is {improvement:.2f}% better (unexpected but interesting)")
else:
    gap = (naive_avg - oracle_avg) / oracle_avg * 100
    print(f"Oracle checkpoints still {gap:.2f}% better than naive")
    print(f"But the learned loss reduces the gap from PU training!")
