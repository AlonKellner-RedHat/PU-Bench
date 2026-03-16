#!/usr/bin/env python3
"""Train PU meta-learning with HIGH L1 REGULARIZATION configuration.

Quality optimizations:
- inner_steps: 8 (maximum task adaptation, strongest meta-gradient)
- meta_lr: 0.003 (3× higher for faster convergence)
- l1_lambda: 1e-2 (10× higher than sparse config for aggressive sparsity)
- AdamW with weight_decay (better regularization)
- Weight normalization (stable parameter scale)

Expected: 50-70% parameter sparsity, clearer loss structure
Training time: ~25-30 minutes (same as sparse config)
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


# Load high L1 config
config = load_config('config/toy_gaussian_meta_high_l1.yaml')
device = get_device(config)

# Create learnable hierarchical PU loss (27 parameters) with HIGH L1 regularization
loss_l1_lambda = float(config.get('loss_l1_lambda', 0.0))
loss_fn = HierarchicalPULoss(
    init_mode=config.get('loss_init_mode', 'random'),
    init_scale=float(config.get('loss_init_scale', 0.01)),
    l1_lambda=loss_l1_lambda
).to(device)

# Meta-optimizer: AdamW with weight decay
meta_lr = float(config.get('meta_lr', 0.001))
meta_weight_decay = float(config.get('meta_weight_decay', 1e-4))
meta_optimizer = torch.optim.AdamW(
    loss_fn.parameters(),
    lr=meta_lr,
    weight_decay=meta_weight_decay
)

# Create checkpoint pool with both oracle and naive checkpoints
pool = CheckpointPool(config)

# Try to load cached checkpoint pool first
if not pool.load_checkpoint_pool():
    # Cache miss - create new checkpoint pool
    print("Creating optimized checkpoint pool...")
    print("This will include:")
    print("  - Oracle checkpoints (trained with PN labels)")
    print("  - Naive checkpoints (trained with PU labels)")
    print("  - Checkpoints from epochs: 1, 50, 100")
    print()
    pool.create_checkpoint_pool(device=device)

print()
print("="*70)
print("HIGH L1 REGULARIZATION PU META-LEARNING SETUP")
print("="*70)
print(f"Device: {device}")
print(f"Total checkpoints: {len(pool.checkpoints)}")
print(f"Initial loss:\n{loss_fn}")
print(f"L1 regularization: lambda={loss_l1_lambda} (10× higher than sparse config)")
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

# Print optimization details
print("="*70)
print("HIGH L1 REGULARIZATION CONFIGURATION")
print("="*70)
print(f"Inner steps: {config.get('inner_steps', 3)} (maximum task adaptation)")
print(f"Inner batch size: {config.get('inner_batch_size', 64)} (full dataset)")
print(f"Meta optimizer: AdamW (lr={meta_lr}, weight_decay={meta_weight_decay})")
print(f"L1 regularization: lambda={loss_l1_lambda} (very aggressive sparsity)")
print(f"Meta batch size: {config.get('meta_batch_size', 12)} (max parallelism)")
print(f"Meta iterations: {config.get('meta_iterations', 300)}")
print(f"Log frequency: {config.get('log_freq', 20)}")
print(f"Weight normalization: Enabled (max |param| = 1 after each step)")
print()
print("Expected benefits:")
print("  - Higher meta LR (3× faster convergence)")
print("  - More inner steps (stronger meta-gradient)")
print("  - HIGH L1 regularization (50-70% sparsity, ~8-13 active params)")
print()


def normalize_loss_parameters(loss_fn):
    """Normalize loss parameters so max absolute value = 1.

    This helps prevent parameter scale drift and improves optimization stability.
    """
    with torch.no_grad():
        params = loss_fn.get_parameters()
        max_abs_val = torch.abs(params).max()

        if max_abs_val > 1e-8:  # Avoid division by zero
            scale_factor = 1.0 / max_abs_val

            # Scale all parameters
            for param in loss_fn.parameters():
                param.data *= scale_factor


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
    """Inner loop: Adapt model on PU task using learned loss."""
    params = dict(model.named_parameters())

    for step in range(num_steps):
        for batch in train_loader:
            x, y_pu = batch[0].to(device), batch[2].to(device)
            params = inner_loop_step(model, params, x, y_pu, loss_fn, lr)
            break  # Only one step per batch

    return params


def evaluate_bce_on_gt(model, params, val_loader, device):
    """Evaluate adapted model on ground truth labels using BCE."""
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


# Create fixed validation sets for stable evaluation
oracle_indices = [i for i in range(len(pool.checkpoints))
                 if pool.checkpoints[i]['task_config']['training_method'] == 'oracle']
naive_indices = [i for i in range(len(pool.checkpoints))
                if pool.checkpoints[i]['task_config']['training_method'] == 'naive']

# Use ALL checkpoints for validation (more stable than sampling)
val_oracle_indices = oracle_indices
val_naive_indices = naive_indices
val_all_indices = oracle_indices + naive_indices

print(f"Validation sets (fixed for all iterations):")
print(f"  Oracle checkpoints: {len(val_oracle_indices)}")
print(f"  Naive checkpoints:  {len(val_naive_indices)}")
print(f"  Total validation:   {len(val_all_indices)}")
print()

# Training loop with quality focus
print("Starting high L1 regularization PU meta-training...")
print(f"  - 8 inner steps for maximum adaptation")
print(f"  - Meta LR: {meta_lr} (3× higher)")
print(f"  - L1 regularization: λ={loss_l1_lambda} (10× higher)")
print()
sys.stdout.flush()

start_time = time.time()
total_samples_processed = 0

# Progress bar for meta-iterations
pbar = tqdm(range(config['meta_iterations']), desc="Meta-training",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

for iteration in pbar:
    iter_start = time.time()

    # Sample checkpoints
    checkpoint_indices = pool.sample_batch(config['meta_batch_size'])

    total_meta_loss = torch.tensor(0.0, device=device)
    iter_samples = 0

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

        # Count samples processed
        iter_samples += config.get('num_samples_per_task', 1000) * config.get('inner_steps', 1)

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

    # Normalize loss parameters to prevent scale drift
    normalize_loss_parameters(loss_fn)

    # Track throughput
    total_samples_processed += iter_samples
    iter_time = time.time() - iter_start
    iter_throughput = iter_samples / iter_time if iter_time > 0 else 0

    # Update progress bar with current metrics
    pbar.set_postfix({
        'meta_loss': f'{avg_meta_loss.item():.4f}',
        'samples/s': f'{int(iter_throughput)}'
    })

    # Log with fixed validation sets
    if (iteration + 1) % config['log_freq'] == 0:
        elapsed_time = time.time() - start_time
        avg_throughput = total_samples_processed / elapsed_time if elapsed_time > 0 else 0

        # Evaluate on FIXED validation sets (same checkpoints every iteration)

        # 1. Meta-loss on ALL validation checkpoints (what we're optimizing)
        meta_val_loss = 0.0
        for ckpt_idx in val_all_indices:
            checkpoint, task, dataloaders = pool.get_checkpoint(ckpt_idx)
            model = SimpleMLP(
                checkpoint['task_config']['num_dimensions'],
                config.get('model_hidden_dims', [32, 32])
            ).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])

            adapted_params = inner_loop(model, dataloaders['train'], loss_fn,
                                       config.get('inner_steps', 3),
                                       config.get('inner_lr', 0.01), device)
            bce = evaluate_bce_on_gt(model, adapted_params, dataloaders['val'], device)
            meta_val_loss += bce
        meta_val_loss /= len(val_all_indices)

        # 2. Oracle-only BCE (independent measurement)
        oracle_bce = 0.0
        for ckpt_idx in val_oracle_indices:
            checkpoint, task, dataloaders = pool.get_checkpoint(ckpt_idx)
            model = SimpleMLP(
                checkpoint['task_config']['num_dimensions'],
                config.get('model_hidden_dims', [32, 32])
            ).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])

            adapted_params = inner_loop(model, dataloaders['train'], loss_fn,
                                       config.get('inner_steps', 3),
                                       config.get('inner_lr', 0.01), device)
            bce = evaluate_bce_on_gt(model, adapted_params, dataloaders['val'], device)
            oracle_bce += bce
        oracle_bce /= len(val_oracle_indices)

        # 3. Naive-only BCE (independent measurement)
        naive_bce = 0.0
        for ckpt_idx in val_naive_indices:
            checkpoint, task, dataloaders = pool.get_checkpoint(ckpt_idx)
            model = SimpleMLP(
                checkpoint['task_config']['num_dimensions'],
                config.get('model_hidden_dims', [32, 32])
            ).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])

            adapted_params = inner_loop(model, dataloaders['train'], loss_fn,
                                       config.get('inner_steps', 3),
                                       config.get('inner_lr', 0.01), device)
            bce = evaluate_bce_on_gt(model, adapted_params, dataloaders['val'], device)
            naive_bce += bce
        naive_bce /= len(val_naive_indices)

        tqdm.write(f"\nIteration {iteration + 1}/{config['meta_iterations']}")
        tqdm.write(f"  Meta-loss (val):        {meta_val_loss:.6f}")
        tqdm.write(f"  Oracle-only BCE (val):  {oracle_bce:.6f}")
        tqdm.write(f"  Naive-only BCE (val):   {naive_bce:.6f}")
        tqdm.write(f"  Oracle vs Naive gap:    {(naive_bce / oracle_bce - 1) * 100:+.1f}%")
        tqdm.write(f"  Throughput: {avg_throughput:,.0f} samples/min (current iter: {iter_throughput*60:,.0f})")
        tqdm.write(f"  {loss_fn}")
        tqdm.write("")
        sys.stdout.flush()

print()
print("="*70)
print("FINAL RESULTS: HIGH L1 REGULARIZATION PU META-LEARNING")
print("="*70)
print()
print(f"Final learned PU loss (with HIGH L1 regularization λ={loss_l1_lambda}):")
print(loss_fn)
print()
print(f"Total samples processed: {total_samples_processed:,}")
print(f"Total time: {time.time() - start_time:.1f} seconds")
print(f"Average throughput: {total_samples_processed / (time.time() - start_time):,.0f} samples/min")
print()

# Analyze parameter sparsity from L1 regularization
params = loss_fn.get_parameters().detach().cpu().numpy()
near_zero = np.sum(np.abs(params) < 0.01)
print(f"Parameter sparsity analysis:")
print(f"  Near-zero parameters (|param| < 0.01): {near_zero}/27 ({near_zero/27*100:.1f}%)")
print(f"  Max |parameter|: {np.abs(params).max():.4f}")
print(f"  Mean |parameter|: {np.abs(params).mean():.4f}")
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

            adapted_params = inner_loop(model, dataloaders['train'], loss_fn,
                                       config.get('inner_steps', 3),
                                       config.get('inner_lr', 0.01), device)
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
print(f"Oracle-Naive Gap: {(naive_avg / oracle_avg - 1) * 100:.1f}%")
print()

# ======================================================================
# TRAIN FROM SCRATCH EVALUATION
# ======================================================================
print("="*70)
print("TRAIN FROM SCRATCH EVALUATION")
print("="*70)
print()
print("Training three models from scratch on fresh tasks:")
print("  1. Oracle: BCE loss on ground truth PN labels")
print("  2. Naive: BCE loss on PU labels (unlabeled = negative)")
print("  3. Learned: Learned hierarchical loss on PU labels")
print()

from tasks.gaussian_task import GaussianBlobTask

# Train on multiple fresh tasks and average results
num_test_tasks = 5
train_epochs = 100
test_results = {'oracle': [], 'naive': [], 'learned': []}
sys.stdout.flush()

for task_idx in tqdm(range(num_test_tasks), desc="Testing on fresh tasks", leave=True):
    tqdm.write(f"\nTest task {task_idx + 1}/{num_test_tasks}")

    # Create a fresh test task (different seed)
    test_task = GaussianBlobTask(
        num_dimensions=2,
        mean_separation=2.5,
        std=1.0,
        prior=0.5,
        labeling_freq=0.3,
        num_samples=1000,
        seed=10000 + task_idx,  # Different seeds
        mode='pu',
        negative_labeling_freq=0.3,
    )

    # Get dataloaders
    test_dataloaders = test_task.get_dataloaders(
        batch_size=64,
        num_train=1000,
        num_val=500,
        num_test=500,
    )

    # --- 1. Oracle: Train with BCE on ground truth ---
    model_oracle = SimpleMLP(2, [32, 32]).to(device)
    optimizer_oracle = torch.optim.Adam(model_oracle.parameters(), lr=0.001)
    bce_fn = nn.BCEWithLogitsLoss()

    for epoch in range(train_epochs):
        model_oracle.train()
        for batch in test_dataloaders['train']:
            x, y_true, y_pu = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            optimizer_oracle.zero_grad()
            outputs = model_oracle(x).squeeze(-1)
            loss = bce_fn(outputs, y_true)  # Ground truth
            loss.backward()
            optimizer_oracle.step()

    # Evaluate oracle
    model_oracle.eval()
    with torch.no_grad():
        test_loss = 0.0
        for batch in test_dataloaders['test']:
            x, y_true = batch[0].to(device), batch[1].to(device)
            outputs = model_oracle(x).squeeze(-1)
            loss = bce_fn(outputs, y_true)
            test_loss += loss.item()
        test_loss /= len(test_dataloaders['test'])
    test_results['oracle'].append(test_loss)

    # --- 2. Naive: Train with BCE on PU labels (unlabeled = negative) ---
    model_naive = SimpleMLP(2, [32, 32]).to(device)
    optimizer_naive = torch.optim.Adam(model_naive.parameters(), lr=0.001)

    for epoch in range(train_epochs):
        model_naive.train()
        for batch in test_dataloaders['train']:
            x, y_true, y_pu = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            # Convert PU labels: labeled positive (1) and unlabeled (-1) → treat unlabeled as negative (0)
            y_naive = torch.where(y_pu == 1, torch.ones_like(y_pu), torch.zeros_like(y_pu))

            optimizer_naive.zero_grad()
            outputs = model_naive(x).squeeze(-1)
            loss = bce_fn(outputs, y_naive)  # Naive PU
            loss.backward()
            optimizer_naive.step()

    # Evaluate naive
    model_naive.eval()
    with torch.no_grad():
        test_loss = 0.0
        for batch in test_dataloaders['test']:
            x, y_true = batch[0].to(device), batch[1].to(device)
            outputs = model_naive(x).squeeze(-1)
            loss = bce_fn(outputs, y_true)
            test_loss += loss.item()
        test_loss /= len(test_dataloaders['test'])
    test_results['naive'].append(test_loss)

    # --- 3. Learned: Train with learned hierarchical loss on PU labels ---
    model_learned = SimpleMLP(2, [32, 32]).to(device)
    optimizer_learned = torch.optim.Adam(model_learned.parameters(), lr=0.001)

    for epoch in range(train_epochs):
        model_learned.train()
        for batch in test_dataloaders['train']:
            x, y_true, y_pu = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            optimizer_learned.zero_grad()
            outputs = model_learned(x).squeeze(-1)
            loss = loss_fn(outputs, y_pu, mode='pu')  # Learned PU loss
            loss.backward()
            optimizer_learned.step()

    # Evaluate learned
    model_learned.eval()
    with torch.no_grad():
        test_loss = 0.0
        for batch in test_dataloaders['test']:
            x, y_true = batch[0].to(device), batch[1].to(device)
            outputs = model_learned(x).squeeze(-1)
            loss = bce_fn(outputs, y_true)
            test_loss += loss.item()
        test_loss /= len(test_dataloaders['test'])
    test_results['learned'].append(test_loss)

    tqdm.write(f"  Oracle: {test_results['oracle'][-1]:.6f}")
    tqdm.write(f"  Naive:  {test_results['naive'][-1]:.6f}")
    tqdm.write(f"  Learned: {test_results['learned'][-1]:.6f}")
    sys.stdout.flush()

# Print final averages
print()
print("="*70)
print("TRAIN FROM SCRATCH RESULTS (averaged over {} tasks)".format(num_test_tasks))
print("="*70)
oracle_avg_scratch = np.mean(test_results['oracle'])
naive_avg_scratch = np.mean(test_results['naive'])
learned_avg_scratch = np.mean(test_results['learned'])

print(f"Oracle BCE (ground truth):  {oracle_avg_scratch:.6f} ± {np.std(test_results['oracle']):.6f}")
print(f"Naive BCE (PU as PN):       {naive_avg_scratch:.6f} ± {np.std(test_results['naive']):.6f}")
print(f"Learned Loss (PU):          {learned_avg_scratch:.6f} ± {np.std(test_results['learned']):.6f}")
print()

# Calculate improvements
naive_vs_oracle = (naive_avg_scratch / oracle_avg_scratch - 1) * 100
learned_vs_oracle = (learned_avg_scratch / oracle_avg_scratch - 1) * 100
learned_vs_naive = (learned_avg_scratch / naive_avg_scratch - 1) * 100

print("Performance gaps:")
print(f"  Naive vs Oracle:  {naive_vs_oracle:+.1f}% (naive is worse)" if naive_vs_oracle > 0 else f"  Naive vs Oracle:  {naive_vs_oracle:+.1f}% (naive is better)")
print(f"  Learned vs Oracle: {learned_vs_oracle:+.1f}% (learned is worse)" if learned_vs_oracle > 0 else f"  Learned vs Oracle: {learned_vs_oracle:+.1f}% (learned is better)")
print(f"  Learned vs Naive:  {learned_vs_naive:+.1f}% (learned is worse)" if learned_vs_naive > 0 else f"  Learned vs Naive:  {learned_vs_naive:+.1f}% (learned is better)")
print()

if learned_avg_scratch < naive_avg_scratch:
    print("✓ Learned loss OUTPERFORMS naive BCE!")
else:
    print("✗ Learned loss does NOT outperform naive BCE")

if learned_avg_scratch < oracle_avg_scratch:
    print("✓✓ Learned loss MATCHES/BEATS oracle BCE!")
else:
    print("  Learned loss still worse than oracle (expected, as oracle has ground truth)")
print("="*70)
