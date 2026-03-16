#!/usr/bin/env python3
"""Test second-order gradients through L0.5 and L1 regularization.

This replicates the exact gradient matching scenario where NaN appears.
"""

import torch
import torch.nn as nn
from torch.func import functional_call
import sys
sys.path.insert(0, '/Users/akellner/MyDir/Code/Other/PU-Bench/toy_meta_learning')

from loss.neural_pu_loss import NeuralPULoss
from models.simple_mlp import SimpleMLP

print("="*70)
print("TESTING SECOND-ORDER GRADIENTS WITH L0.5 + L1 REGULARIZATION")
print("="*70)

# Create model and loss exactly as in meta-training
model = SimpleMLP(2, [32, 32])
learned_loss = NeuralPULoss(
    hidden_dim=128,
    eps=1e-7,
    l1_lambda=0.01,
    l05_lambda=0.005,
    init_mode='xavier_uniform',
    init_scale=1.0,
    max_weight_norm=10.0,
)

# Create params dict for functional API
params = {name: param.clone().requires_grad_(True) for name, param in model.named_parameters()}

# Generate dummy data
x = torch.randn(64, 2)
y_pu = torch.where(torch.rand(64) > 0.7, torch.tensor(1.0), torch.tensor(-1.0))
y_true = torch.rand(64)  # Binary labels

print("\nTest 1: Forward pass with regularization")
print("-" * 70)
outputs = model(x).squeeze(-1)
loss = learned_loss(outputs, y_pu, mode='pu')
print(f"Loss: {loss.item():.6f}")
print(f"Has NaN: {torch.isnan(loss).item()}")

print("\nTest 2: First-order gradients (standard backward)")
print("-" * 70)
learned_loss.zero_grad()
loss.backward()
print(f"Weight grad norm: {learned_loss.linear.weight.grad.norm().item():.6f}")
print(f"Has NaN in weight grads: {torch.isnan(learned_loss.linear.weight.grad).any().item()}")

print("\nTest 3: Second-order gradients via torch.func.grad (GRADIENT MATCHING)")
print("-" * 70)

# This is exactly what happens in gradient matching
def pu_loss_fn(param_dict):
    """Loss function for torch.func.grad."""
    model_outputs = functional_call(model, param_dict, x).squeeze(-1)
    return learned_loss(model_outputs, y_pu, mode='pu')

print("Computing gradients via torch.func.grad...")
try:
    pu_grads = torch.func.grad(pu_loss_fn)(params)

    # Flatten gradients
    pu_grad_vec = torch.cat([g.flatten() for g in pu_grads.values()])

    print(f"✓ Gradients computed successfully")
    print(f"  Gradient vector norm: {pu_grad_vec.norm().item():.6f}")
    print(f"  Has NaN: {torch.isnan(pu_grad_vec).any().item()}")
    print(f"  Has Inf: {torch.isinf(pu_grad_vec).any().item()}")

except Exception as e:
    print(f"✗ Error computing gradients: {e}")

print("\nTest 4: Backward through the torch.func.grad output (META-GRADIENT)")
print("-" * 70)

# This is the key: can we backprop through the gradients?
pu_grads = torch.func.grad(pu_loss_fn)(params)
pu_grad_vec = torch.cat([g.flatten() for g in pu_grads.values()])

# Create dummy target gradients
bce_grad_vec = torch.randn_like(pu_grad_vec).detach()

# Compute meta-loss (this requires second-order derivatives)
meta_loss = torch.mean((pu_grad_vec - bce_grad_vec) ** 2)

print(f"Meta-loss (gradient MSE): {meta_loss.item():.6f}")
print(f"Has NaN: {torch.isnan(meta_loss).item()}")

# Backward through meta-loss (this computes gradients of loss params)
print("\nComputing meta-gradients...")
try:
    meta_loss.backward()

    print(f"✓ Meta-gradients computed successfully")
    print(f"  Loss weight grad norm: {learned_loss.linear.weight.grad.norm().item():.6f}")
    print(f"  Loss weight grad has NaN: {torch.isnan(learned_loss.linear.weight.grad).any().item()}")
    print(f"  Loss weight grad has Inf: {torch.isinf(learned_loss.linear.weight.grad).any().item()}")

except Exception as e:
    print(f"✗ Error computing meta-gradients: {e}")

print("\nTest 5: Check L0.5 gradient at near-zero weights")
print("-" * 70)

# Manually set some weights very close to zero
with torch.no_grad():
    learned_loss.linear.weight[0, 0] = 1e-10
    learned_loss.linear.weight[0, 1] = -1e-10
    learned_loss.linear.weight[0, 2] = 0.0

print(f"Weight values: {learned_loss.linear.weight[0, :5]}")

# Compute L0.5 reg manually
l05_val_no_eps = (torch.abs(learned_loss.linear.weight[0, 0]) ** 0.5)
l05_val_with_eps = ((torch.abs(learned_loss.linear.weight[0, 0]) + learned_loss.eps) ** 0.5)

print(f"L0.5 (no eps) for 1e-10: {l05_val_no_eps.item():.15f}")
print(f"L0.5 (with eps) for 1e-10: {l05_val_with_eps.item():.15f}")

# Compute gradient of L0.5
l05_val_no_eps.backward()
print(f"Gradient of L0.5 (no eps): {learned_loss.linear.weight.grad[0, 0].item()}")

learned_loss.zero_grad()
l05_val_with_eps.backward()
print(f"Gradient of L0.5 (with eps): {learned_loss.linear.weight.grad[0, 0].item()}")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
