#!/usr/bin/env python3
"""Meta-learning with ALIGNED meta-objective and soft regularization.

Key improvements over aggressive stabilization:
1. Multi-timescale meta-objective (3-step + 10-step evaluation)
2. Soft regularization toward initialization (not hard clipping)
3. Trust region for safety
4. Normal gradient clipping (not aggressive)

This should enable actual learning while maintaining stability.
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


def train_n_steps(model, train_data, loss_fn, num_steps, lr, device):
    """Train model for N steps and return final parameters."""
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
            create_graph=True,  # Full MAML (not FOMAML)
            allow_unused=True
        )

        params = {
            name: param - lr * grad
            for (name, param), grad in zip(params.items(), grads)
            if grad is not None
        }

    return params


def compute_multi_timescale_meta_loss(model, dataloaders, loss_fn, device, config):
    """Meta-objective: weighted average at 3 steps and 10 steps.

    This aligns meta-objective with final performance better than single 3-step eval.
    """
    train_batch = next(iter(dataloaders['train']))
    val_batch = next(iter(dataloaders['val']))

    train_data = (train_batch[0], train_batch[1], train_batch[2])
    x_val = val_batch[0].to(device)
    y_true_val = val_batch[1].to(device)

    bce_fn = nn.BCEWithLogitsLoss()
    inner_lr = config['inner_lr']

    # Evaluation at 3 steps (quick adaptation)
    params_3step = train_n_steps(model, train_data, loss_fn, num_steps=3, lr=inner_lr, device=device)
    outputs_3step = functional_call(model, params_3step, x_val).squeeze(-1)
    loss_3step = bce_fn(outputs_3step, y_true_val)

    # Evaluation at 10 steps (better final performance proxy)
    params_10step = train_n_steps(model, train_data, loss_fn, num_steps=10, lr=inner_lr, device=device)
    outputs_10step = functional_call(model, params_10step, x_val).squeeze(-1)
    loss_10step = bce_fn(outputs_10step, y_true_val)

    # Weighted combination (more weight on longer adaptation)
    meta_loss = 0.3 * loss_3step + 0.7 * loss_10step

    return meta_loss, loss_3step.item(), loss_10step.item()


# Load config
config = load_config('config/toy_gaussian_meta_large_pool.yaml')
device = get_device(config)

config['inner_steps'] = 3  # Used for display only
config['inner_lr'] = 0.3
config['meta_batch_size'] = 8
config['meta_iterations'] = 100  # Test with 100 first

print("="*70)
print("META-LEARNING WITH ALIGNED OBJECTIVE (Multi-timescale + Soft Regularization)")
print("="*70)
print("Improvements:")
print("  1. Multi-timescale meta-objective (30% @3steps + 70% @10steps)")
print("  2. Soft regularization toward initialization (drift penalty)")
print("  3. Trust region for bounded updates")
print("  4. Normal gradient clipping (max_norm=1.0, not 0.1)")
print()
print("Testing with VPU initialization to see if it can learn stably")
print(f"Device: {device}")
print("="*70)
print()

# VPU initialization
loss_fn = HierarchicalPULoss(
    init_mode='vpu_inspired',
    init_scale=0.01,
    l1_lambda=0.001,
    eps=1e-6
).to(device)

# Store initial parameters for drift penalty
params_init = loss_fn.get_parameters().clone().detach()

print("Initial loss:")
print(loss_fn)
print()

# Meta-optimizer with normal LR
meta_optimizer = torch.optim.AdamW(
    loss_fn.parameters(),
    lr=1e-4,  # Normal LR (not reduced)
    weight_decay=1e-5
)

# Regularization and trust region settings
DRIFT_LAMBDA = 0.01  # Soft penalty for parameter drift
MAX_PARAM_CHANGE = 0.05  # Trust region bound

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
print(f"  VPU baseline: {val_results['vpu_nomixup']:.6f}")
diff = abs(val_results['learned'] - val_results['vpu_nomixup'])
print(f"  Difference:   {diff:.6f}")
if diff < 0.02:
    print("  ✓ VPU initialization verified!")
print("="*70)
print()

# Training loop
print("Starting aligned meta-learning...")
print()
start_time = time.time()

trust_region_activations = 0
param_changes = []

for iteration in tqdm(range(config['meta_iterations']), desc="Meta-training"):
    total_meta_loss = torch.tensor(0.0, device=device)
    total_loss_3step = 0.0
    total_loss_10step = 0.0
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

        model = SimpleMLP(
            input_dim=2,
            hidden_dims=config.get('model_hidden_dims', [32, 32]),
        ).to(device)

        # Multi-timescale meta-loss
        meta_loss, loss_3s, loss_10s = compute_multi_timescale_meta_loss(
            model, dataloaders, loss_fn, device, config
        )

        total_meta_loss = total_meta_loss + meta_loss
        total_loss_3step += loss_3s
        total_loss_10step += loss_10s
        num_tasks += 1

    # Average meta-loss
    avg_meta_loss = total_meta_loss / num_tasks
    avg_3step = total_loss_3step / num_tasks
    avg_10step = total_loss_10step / num_tasks

    # SOFT REGULARIZATION: Drift penalty
    params_current = loss_fn.get_parameters()
    drift_penalty = DRIFT_LAMBDA * torch.norm(params_current - params_init)

    # Total loss with regularization
    total_loss = avg_meta_loss + drift_penalty

    # Get parameters before update
    params_before = loss_fn.get_parameters().clone().detach()

    # Meta-update
    meta_optimizer.zero_grad()
    total_loss.backward()

    # Normal gradient clipping (not aggressive)
    global_norm = torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), max_norm=1.0)

    meta_optimizer.step()

    # TRUST REGION: Check parameter change
    params_after = loss_fn.get_parameters()
    param_change = torch.norm(params_after - params_before).item()
    param_changes.append(param_change)

    if param_change > MAX_PARAM_CHANGE:
        # Revert and scale to trust region boundary
        direction = (params_after - params_before) / param_change
        params_safe = params_before + MAX_PARAM_CHANGE * direction

        with torch.no_grad():
            idx = 0
            for param in loss_fn.parameters():
                param.copy_(params_safe[idx:idx+1].view(1))
                idx += 1

        trust_region_activations += 1
        param_change = MAX_PARAM_CHANGE

    # Validation every 20 iterations
    if (iteration + 1) % 20 == 0:
        val_results, cached_baselines = train_from_scratch_validation(
            val_tasks, loss_fn, config, device, cached_baselines
        )

        elapsed = time.time() - start_time
        iters_per_min = (iteration + 1) / (elapsed / 60)

        # Compute current drift
        current_drift = torch.norm(loss_fn.get_parameters() - params_init).item()
        avg_param_change = np.mean(param_changes[-20:]) if len(param_changes) >= 20 else np.mean(param_changes)

        tqdm.write(f"\nIteration {iteration + 1}/{config['meta_iterations']}")
        tqdm.write(f"  Meta-loss (weighted):           {avg_meta_loss.item():.6f}")
        tqdm.write(f"    - 3-step component:           {avg_3step:.6f}")
        tqdm.write(f"    - 10-step component:          {avg_10step:.6f}")
        tqdm.write(f"  Drift penalty:                  {drift_penalty.item():.6f}")
        tqdm.write(f"  Total loss:                     {total_loss.item():.6f}")
        tqdm.write(f"  Gradient norm:                  {global_norm:.6f}")
        tqdm.write(f"  Param change (current):         {param_change:.6f}")
        tqdm.write(f"  Param change (avg last 20):     {avg_param_change:.6f}")
        tqdm.write(f"  Total drift from init:          {current_drift:.6f}")
        tqdm.write(f"  Trust region activations:       {trust_region_activations}/{iteration+1}")
        tqdm.write(f"  Speed: {iters_per_min:.1f} iters/min")
        tqdm.write(f"  --- Validation ---")
        tqdm.write(f"  Learned:      {val_results['learned']:.6f}")
        tqdm.write(f"  VPU baseline: {val_results['vpu_nomixup']:.6f}")

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
print(f"Trust region activated: {trust_region_activations}/{config['meta_iterations']} times")
print(f"Final drift from initialization: {torch.norm(loss_fn.get_parameters() - params_init).item():.6f}")
print()
print("Final learned loss:")
print(loss_fn)
print()
print("="*70)
print("ANALYSIS")
print("="*70)

final_learned = val_results['learned']
final_baseline = val_results['vpu_nomixup']
init_learned = 0.309  # From iter 0

improvement_from_init = ((init_learned - final_learned) / init_learned) * 100
diff_from_baseline = abs(final_learned - final_baseline)

print(f"Initial (iter 0):     {init_learned:.6f}")
print(f"Final (iter 100):     {final_learned:.6f}")
print(f"VPU baseline:         {final_baseline:.6f}")
print()
print(f"Improvement from init: {improvement_from_init:+.1f}%")
print(f"Diff from baseline:    {diff_from_baseline:.6f}")
print()

if final_learned < init_learned - 0.01:
    print("✓ IMPROVEMENT - Meta-learning made loss better!")
elif final_learned < init_learned + 0.01:
    print("~ STABLE - Negligible change (as expected if optimal init)")
else:
    print("✗ DEGRADATION - Need to tune hyperparameters")

print()
print("Parameter changes:")
params_final = loss_fn.get_parameters().detach().cpu().numpy()
params_init_np = params_init.detach().cpu().numpy()
changes = np.abs(params_final - params_init_np)
print(f"  Max change:  {changes.max():.6f}")
print(f"  Mean change: {changes.mean():.6f}")
print(f"  Num changed >0.01: {(changes > 0.01).sum()}/27")
print("="*70)
