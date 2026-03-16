#!/usr/bin/env python3
"""Debug script to investigate why meta-learning parameters don't update.

Tests 5 hypotheses:
1. Meta-gradients are zero/vanishingly small
2. Computational graph is broken (create_graph issue)
3. Meta-optimizer not applying updates
4. Learning rate too small
5. Parameters detached from graph
"""

import torch
import torch.nn as nn
import yaml
import numpy as np
from pathlib import Path

from models.simple_mlp import SimpleMLP
from loss.simple_basis_loss import SimpleBasisLoss
from tasks.task_pool import CheckpointPool


def load_config(config_path: str = 'config/toy_gaussian_meta.yaml') -> dict:
    """Load config."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if len(config) == 1 and isinstance(list(config.values())[0], dict):
        config = list(config.values())[0]
    return config


def get_device(config: dict) -> str:
    """Get device."""
    device_config = config.get('device', 'auto')
    if device_config == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    return device_config


print("="*80)
print("META-GRADIENT DEBUGGING")
print("="*80)

# Load config
config = load_config()
device = get_device(config)
print(f"Device: {device}\n")

# Create minimal checkpoint pool (just 2 checkpoints for speed)
config['mean_separations'] = [2.0]
config['checkpoint_seeds'] = [42]
config['checkpoint_epochs'] = [1, 5]

checkpoint_pool = CheckpointPool(config)
print("Creating minimal checkpoint pool (2 checkpoints)...")
checkpoint_pool.create_checkpoint_pool(device=device)

# Create loss
loss_fn = SimpleBasisLoss(
    init_mode='random',
    init_scale=0.01,
).to(device)

print(f"\nInitial loss parameters: {loss_fn}")
print(f"  a1: {loss_fn.a1.item():.6f}")
print(f"  a2: {loss_fn.a2.item():.6f}")
print(f"  a3: {loss_fn.a3.item():.6f}")

# Create meta-optimizer
meta_lr = 0.001
meta_optimizer = torch.optim.Adam(loss_fn.parameters(), lr=meta_lr)

mode = config.get('mode', 'pn')

print("\n" + "="*80)
print("HYPOTHESIS 1: Meta-gradients are zero/vanishingly small")
print("="*80)

# Get first checkpoint
checkpoint, task, dataloaders = checkpoint_pool.get_checkpoint(0)

# Create model
model = SimpleMLP(
    input_dim=checkpoint['task_config']['num_dimensions'],
    hidden_dims=config.get('model_hidden_dims', [32, 32]),
).to(device)
model.load_state_dict(checkpoint['model_state_dict'])

# Inner loop (1 step)
inner_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
model.train()

for batch_x, batch_y in dataloaders['train']:
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    inner_optimizer.zero_grad()
    outputs = model(batch_x)
    loss = loss_fn(outputs, batch_y, mode=mode)

    print(f"\nInner loop loss value: {loss.item():.6f}")
    print(f"Inner loop loss requires_grad: {loss.requires_grad}")
    print(f"Inner loop loss grad_fn: {loss.grad_fn}")

    loss.backward(create_graph=True)
    inner_optimizer.step()
    break  # Just one batch

# Meta-objective: evaluate on validation
model.train()
bce_fn = nn.BCEWithLogitsLoss()
meta_loss = 0.0

for batch in dataloaders['val']:
    if len(batch) == 3:
        batch_x, batch_y_pu, batch_y_true = batch
        batch_y = batch_y_true
    else:
        batch_x, batch_y = batch

    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    outputs = model(batch_x).squeeze()
    loss = bce_fn(outputs, batch_y)
    meta_loss += loss
    break  # Just one batch

print(f"\nMeta-loss value: {meta_loss.item():.6f}")
print(f"Meta-loss requires_grad: {meta_loss.requires_grad}")
print(f"Meta-loss grad_fn: {meta_loss.grad_fn}")

# Check if loss parameters have grad before backward
print(f"\nBefore backward:")
print(f"  a1.grad: {loss_fn.a1.grad}")
print(f"  a2.grad: {loss_fn.a2.grad}")
print(f"  a3.grad: {loss_fn.a3.grad}")

# Backward to compute meta-gradients
meta_optimizer.zero_grad()
meta_loss.backward()

# Check gradients
print(f"\nAfter backward:")
print(f"  a1.grad: {loss_fn.a1.grad}")
print(f"  a2.grad: {loss_fn.a2.grad}")
print(f"  a3.grad: {loss_fn.a3.grad}")

if loss_fn.a1.grad is not None:
    print(f"\nGradient magnitudes:")
    print(f"  |∇a1|: {loss_fn.a1.grad.abs().item():.10f}")
    print(f"  |∇a2|: {loss_fn.a2.grad.abs().item():.10f}")
    print(f"  |∇a3|: {loss_fn.a3.grad.abs().item():.10f}")

    grad_a1 = loss_fn.a1.grad.item()
    grad_a2 = loss_fn.a2.grad.item()
    grad_a3 = loss_fn.a3.grad.item()

    if abs(grad_a1) < 1e-10 and abs(grad_a2) < 1e-10 and abs(grad_a3) < 1e-10:
        print("\n❌ HYPOTHESIS 1 CONFIRMED: Gradients are vanishingly small!")
    else:
        print("\n✅ HYPOTHESIS 1 REJECTED: Gradients exist and are non-zero")
else:
    print("\n❌ HYPOTHESIS 1: Gradients are None! Graph may be broken.")

print("\n" + "="*80)
print("HYPOTHESIS 2: Computational graph is broken")
print("="*80)

# Check if removing create_graph breaks things
print("\nTrying inner loop WITHOUT create_graph=True...")
model2 = SimpleMLP(
    input_dim=checkpoint['task_config']['num_dimensions'],
    hidden_dims=config.get('model_hidden_dims', [32, 32]),
).to(device)
model2.load_state_dict(checkpoint['model_state_dict'])

inner_optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
model2.train()

for batch_x, batch_y in dataloaders['train']:
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    inner_optimizer2.zero_grad()
    outputs = model2(batch_x)
    loss = loss_fn(outputs, batch_y, mode=mode)
    loss.backward(create_graph=False)  # No create_graph
    inner_optimizer2.step()
    break

# Try to compute meta-loss
model2.train()
meta_loss2 = 0.0
for batch in dataloaders['val']:
    if len(batch) == 3:
        batch_x, batch_y_pu, batch_y_true = batch
        batch_y = batch_y_true
    else:
        batch_x, batch_y = batch

    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    outputs = model2(batch_x).squeeze()
    loss = bce_fn(outputs, batch_y)
    meta_loss2 += loss
    break

print(f"Meta-loss2 requires_grad: {meta_loss2.requires_grad}")
print(f"Meta-loss2 grad_fn: {meta_loss2.grad_fn}")

try:
    meta_optimizer.zero_grad()
    meta_loss2.backward()
    print("✅ Backward succeeded without create_graph")
    print(f"  a1.grad: {loss_fn.a1.grad}")
    if loss_fn.a1.grad is not None:
        print(f"  |∇a1|: {loss_fn.a1.grad.abs().item():.10f}")
        print("\n✅ HYPOTHESIS 2 REJECTED: Graph works even without create_graph!")
        print("   (This suggests create_graph is not the issue)")
    else:
        print("\n❌ HYPOTHESIS 2 CONFIRMED: Gradients are None without create_graph")
except Exception as e:
    print(f"\n❌ Error during backward without create_graph: {e}")
    print("   HYPOTHESIS 2 CONFIRMED: create_graph is necessary")

print("\n" + "="*80)
print("HYPOTHESIS 3: Meta-optimizer not applying updates")
print("="*80)

# Save parameters before update
a1_before = loss_fn.a1.item()
a2_before = loss_fn.a2.item()
a3_before = loss_fn.a3.item()

print(f"Parameters BEFORE optimizer.step():")
print(f"  a1: {a1_before:.10f}")
print(f"  a2: {a2_before:.10f}")
print(f"  a3: {a3_before:.10f}")

if loss_fn.a1.grad is not None:
    print(f"\nGradients:")
    print(f"  ∇a1: {loss_fn.a1.grad.item():.10f}")
    print(f"  ∇a2: {loss_fn.a2.grad.item():.10f}")
    print(f"  ∇a3: {loss_fn.a3.grad.item():.10f}")

# Apply optimizer step
meta_optimizer.step()

# Check parameters after update
a1_after = loss_fn.a1.item()
a2_after = loss_fn.a2.item()
a3_after = loss_fn.a3.item()

print(f"\nParameters AFTER optimizer.step():")
print(f"  a1: {a1_after:.10f}")
print(f"  a2: {a2_after:.10f}")
print(f"  a3: {a3_after:.10f}")

# Compute change
delta_a1 = a1_after - a1_before
delta_a2 = a2_after - a2_before
delta_a3 = a3_after - a3_before

print(f"\nParameter changes:")
print(f"  Δa1: {delta_a1:.10f}")
print(f"  Δa2: {delta_a2:.10f}")
print(f"  Δa3: {delta_a3:.10f}")

if abs(delta_a1) < 1e-10 and abs(delta_a2) < 1e-10 and abs(delta_a3) < 1e-10:
    print("\n❌ HYPOTHESIS 3 CONFIRMED: Parameters did not change!")
else:
    print("\n✅ HYPOTHESIS 3 REJECTED: Parameters did change")

print("\n" + "="*80)
print("HYPOTHESIS 4: Learning rate too small")
print("="*80)

if loss_fn.a1.grad is not None:
    grad_a1 = loss_fn.a1.grad.item()
    grad_a2 = loss_fn.a2.grad.item()
    grad_a3 = loss_fn.a3.grad.item()

    # Expected update size (for SGD would be lr * grad, for Adam it's more complex)
    print(f"Meta-LR: {meta_lr}")
    print(f"\nSimple SGD update would be:")
    print(f"  Δa1 = -{meta_lr} * {grad_a1:.10f} = {-meta_lr * grad_a1:.10f}")
    print(f"  Δa2 = -{meta_lr} * {grad_a2:.10f} = {-meta_lr * grad_a2:.10f}")
    print(f"  Δa3 = -{meta_lr} * {grad_a3:.10f} = {-meta_lr * grad_a3:.10f}")

    # Check if gradients are non-zero but tiny
    if max(abs(grad_a1), abs(grad_a2), abs(grad_a3)) < 1e-6:
        print("\n❌ HYPOTHESIS 4 CONFIRMED: Gradients are tiny, LR may need to be huge")
    else:
        print("\n✅ HYPOTHESIS 4 REJECTED: Gradients are reasonable, LR should work")

print("\n" + "="*80)
print("HYPOTHESIS 5: Parameters detached from graph")
print("="*80)

print(f"Loss parameters require_grad:")
print(f"  a1.requires_grad: {loss_fn.a1.requires_grad}")
print(f"  a2.requires_grad: {loss_fn.a2.requires_grad}")
print(f"  a3.requires_grad: {loss_fn.a3.requires_grad}")

print(f"\nLoss parameters is_leaf:")
print(f"  a1.is_leaf: {loss_fn.a1.is_leaf}")
print(f"  a2.is_leaf: {loss_fn.a2.is_leaf}")
print(f"  a3.is_leaf: {loss_fn.a3.is_leaf}")

if loss_fn.a1.requires_grad and loss_fn.a2.requires_grad and loss_fn.a3.requires_grad:
    print("\n✅ HYPOTHESIS 5 REJECTED: Parameters have requires_grad=True")
else:
    print("\n❌ HYPOTHESIS 5 CONFIRMED: Some parameters don't require gradients!")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\nRun this script to see which hypothesis explains the frozen parameters.")
