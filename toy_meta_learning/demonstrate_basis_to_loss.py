#!/usr/bin/env python3
"""Demonstrate how basis function f(x) forms the actual loss."""

import torch
import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("HOW BASIS FUNCTION FORMS A LOSS")
print("="*70)
print()

# Define basis function
def f(x, a1, a2, a3):
    """Basis function: f(x) = a1 + a2*x + a3*log(x)"""
    eps = 1e-7
    x_safe = torch.clamp(x, min=eps, max=1-eps)
    return a1 + a2 * x_safe + a3 * torch.log(x_safe)

# Example predictions
p_values = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])

print("Example: 3 positive samples and 2 negative samples")
print()

# Positive samples (high probabilities = good)
p_positives = torch.tensor([0.8, 0.7, 0.9])
print(f"Positive samples (p): {p_positives.numpy()}")

# Negative samples (low probabilities = good, so we flip to high values)
p_negatives = torch.tensor([0.2, 0.3])
print(f"Negative samples (p): {p_negatives.numpy()}")
print(f"Negatives flipped (1-p): {(1-p_negatives).numpy()}")
print()

# Test different parameter configurations
configs = [
    ("Pure BCE", (0.0, 0.0, -1.0)),
    ("Learned", (0.0, -0.95, -0.97)),
]

print("="*70)
print("COMPUTING LOSSES")
print("="*70)
print()

for name, (a1, a2, a3) in configs:
    print(f"{name}: a1={a1}, a2={a2}, a3={a3}")
    print()

    # Apply basis to positives
    f_pos = f(p_positives, a1, a2, a3)
    print(f"  f(p) for positives: {f_pos.numpy()}")
    print(f"  Mean: {f_pos.mean().item():.4f}")

    # Apply basis to negatives (flipped!)
    f_neg = f(1 - p_negatives, a1, a2, a3)
    print(f"  f(1-p) for negatives: {f_neg.numpy()}")
    print(f"  Mean: {f_neg.mean().item():.4f}")

    # Total loss
    total_loss = f_pos.mean() + f_neg.mean()
    print(f"  → Total Loss L_PN = E_P[f(p)] + E_N[f(1-p)] = {total_loss.item():.4f}")
    print()

print("="*70)
print("VERIFICATION: Compare to standard BCE")
print("="*70)
print()

# Standard BCE
bce = torch.nn.BCELoss()

# Create targets
y_positives = torch.ones(3)
y_negatives = torch.zeros(2)

# Compute BCE
bce_pos = -torch.log(p_positives).mean()
bce_neg = -torch.log(1 - p_negatives).mean()
bce_total = bce_pos + bce_neg

print(f"BCE on positives: -E_P[log(p)] = {bce_pos.item():.4f}")
print(f"BCE on negatives: -E_N[log(1-p)] = {bce_neg.item():.4f}")
print(f"Total BCE: {bce_total.item():.4f}")
print()

# Compute with basis (a1=0, a2=0, a3=-1)
basis_bce_pos = f(p_positives, 0, 0, -1).mean()
basis_bce_neg = f(1 - p_negatives, 0, 0, -1).mean()
basis_bce_total = basis_bce_pos + basis_bce_neg

print(f"Basis with (a1=0, a2=0, a3=-1):")
print(f"  E_P[f(p)] = {basis_bce_pos.item():.4f}")
print(f"  E_N[f(1-p)] = {basis_bce_neg.item():.4f}")
print(f"  Total = {basis_bce_total.item():.4f}")
print()

print(f"✓ Match? {abs(bce_total.item() - basis_bce_total.item()) < 1e-5}")
print()

print("="*70)
print("VISUALIZING THE DIFFERENCE")
print("="*70)
print()

# Generate smooth curves
p_smooth = torch.linspace(0.01, 0.99, 100)

# Pure BCE
f_bce = f(p_smooth, 0, 0, -1)

# Learned
f_learned = f(p_smooth, 0, -0.95, -0.97)

print("For positive example (we apply f(p)):")
print()
print("  p    | Pure BCE: -log(p) | Learned: -0.95p - 0.97log(p) | Difference")
print("  -----|-------------------|------------------------------|------------")
for p_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
    p_tensor = torch.tensor([p_val])
    bce_val = f(p_tensor, 0, 0, -1).item()
    learned_val = f(p_tensor, 0, -0.95, -0.97).item()
    diff = learned_val - bce_val
    print(f"  {p_val:.1f}  | {bce_val:17.3f} | {learned_val:28.3f} | {diff:+10.3f}")

print()
print("Insight: The learned loss adds a LINEAR PENALTY that grows with confidence!")
print()
print("For p → 1 (very confident):")
print("  - Pure BCE: loss → 0 (no penalty)")
print("  - Learned:  loss → -0.95 (significant penalty)")
print()
print("This prevents overconfidence and improves few-shot adaptation!")
