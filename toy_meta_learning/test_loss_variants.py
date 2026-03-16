#!/usr/bin/env python3
"""Test different loss configurations to see which performs best.

This properly tests each loss by using it in the inner loop adaptation.
"""

import torch
import torch.nn as nn
from torch.func import functional_call, grad
import yaml

from models.simple_mlp import SimpleMLP
from loss.simple_basis_loss import SimpleBasisLoss
from tasks.task_pool import CheckpointPool


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if len(config) == 1:
        config = list(config.values())[0]
    return config


def compute_task_loss(model, params, x, y, loss_fn, mode='pn'):
    """Compute loss using functional_call."""
    outputs = functional_call(model, params, x)
    return loss_fn(outputs, y, mode=mode)


def inner_loop_step(model, params, x, y, loss_fn, mode, lr):
    """Single adaptation step."""
    grads = grad(lambda m, p, x, y: compute_task_loss(m, p, x, y, loss_fn, mode), argnums=1)(
        model, params, x, y
    )
    return {name: param - lr * grads[name] for name, param in params.items()}


def inner_loop(model, train_loader, loss_fn, mode, num_steps, lr, device):
    """Full inner loop adaptation."""
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
    """Evaluate with BCE."""
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


# Load config
config = load_config('config/toy_gaussian_meta.yaml')
config['mean_separations'] = [2.0, 3.0]
config['checkpoint_seeds'] = [42, 123]
config['checkpoint_epochs'] = [1, 5, 10]

device = 'mps'
mode = 'pn'

# Create checkpoint pool
pool = CheckpointPool(config)
print("Creating checkpoint pool...")
pool.create_checkpoint_pool(device=device)

print("\n" + "="*70)
print("TESTING LOSS VARIANTS")
print("="*70)
print()

# Test different loss configurations
loss_configs = [
    ("Pure BCE", {"a1": 0.0, "a2": 0.0, "a3": -1.0}),
    ("Learned (1000 iter)", {"a1": 0.0218, "a2": -0.9525, "a3": -0.9674}),
    ("Only log term (a3=-0.97)", {"a1": 0.0, "a2": 0.0, "a3": -0.9674}),
    ("Log + Linear", {"a1": 0.0, "a2": -0.9525, "a3": -1.0}),
    ("Only linear", {"a1": 0.0, "a2": -0.9525, "a3": 0.0}),
]

checkpoint_indices = list(range(len(pool.checkpoints)))  # All checkpoints

results = []

for loss_name, params in loss_configs:
    # Create loss with these parameters
    loss_fn = SimpleBasisLoss(init_mode='random').to(device)
    loss_fn.a1.data = torch.tensor([params["a1"]], device=device)
    loss_fn.a2.data = torch.tensor([params["a2"]], device=device)
    loss_fn.a3.data = torch.tensor([params["a3"]], device=device)

    total_bce = 0.0

    for ckpt_idx in checkpoint_indices:
        checkpoint, task, dataloaders = pool.get_checkpoint(ckpt_idx)

        # Load model
        model = SimpleMLP(
            input_dim=checkpoint['task_config']['num_dimensions'],
            hidden_dims=config.get('model_hidden_dims', [32, 32]),
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Adapt with THIS loss (not a different one!)
        adapted_params = inner_loop(
            model=model,
            train_loader=dataloaders['train'],
            loss_fn=loss_fn,  # Use the test loss!
            mode=mode,
            num_steps=3,
            lr=config.get('inner_lr', 0.01),
            device=device,
        )

        # Evaluate on validation with BCE
        val_bce = evaluate_bce(model, adapted_params, dataloaders['val'], device)
        total_bce += val_bce

    avg_bce = total_bce / len(checkpoint_indices)
    results.append((loss_name, avg_bce, params))

    print(f"{loss_name:25s}: Val BCE = {avg_bce:.6f}")
    print(f"  Params: a1={params['a1']:+.4f}, a2={params['a2']:+.4f}, a3={params['a3']:+.4f}")
    print()

# Find best
results.sort(key=lambda x: x[1])
best_name, best_bce, best_params = results[0]

print("="*70)
print("RESULTS")
print("="*70)
print(f"\nBest configuration: {best_name}")
print(f"  BCE: {best_bce:.6f}")
print(f"  Params: a1={best_params['a1']:+.4f}, a2={best_params['a2']:+.4f}, a3={best_params['a3']:+.4f}")
print()

pure_bce_result = [r for r in results if r[0] == "Pure BCE"][0]
learned_result = [r for r in results if "Learned" in r[0]][0]

print(f"Pure BCE:        {pure_bce_result[1]:.6f}")
print(f"Learned (1000):  {learned_result[1]:.6f}")
print(f"Difference:      {abs(pure_bce_result[1] - learned_result[1]):.6f}")
print()

if learned_result[1] < pure_bce_result[1]:
    print("✅ Learned loss BEATS pure BCE!")
    print("   The linear term (a2) provides a useful inductive bias.")
elif learned_result[1] > pure_bce_result[1]:
    print("❌ Pure BCE BEATS learned loss")
    print("   Meta-learning got stuck in a suboptimal solution.")
else:
    print("⚠️  Nearly identical performance")
    print("   The linear term has minimal effect.")
