#!/usr/bin/env python3
"""Test that VPU-inspired initialization matches VPU-NoMixUp baseline."""

import torch
import numpy as np
from loss.hierarchical_pu_loss import HierarchicalPULoss
from loss.baseline_losses import VPUNoMixUpLoss

# Create both losses
hierarchical = HierarchicalPULoss(
    init_mode='vpu_inspired',
    l1_lambda=0.0
)

vpu_baseline = VPUNoMixUpLoss()

print("="*70)
print("VPU INITIALIZATION VERIFICATION")
print("="*70)
print()
print("Hierarchical loss with VPU initialization:")
print(hierarchical)
print()

# Test on random data
torch.manual_seed(42)
batch_size = 100
outputs = torch.randn(batch_size) * 2  # Random logits

# Create PU labels (50 positive, 50 unlabeled)
labels = torch.cat([
    torch.ones(50),
    -torch.ones(50)
])

# Compute losses
loss_hierarchical = hierarchical(outputs, labels, mode='pu')
loss_vpu = vpu_baseline(outputs, labels, mode='pu')

print("Test on random batch (batch_size=100):")
print(f"  Hierarchical (VPU init): {loss_hierarchical.item():.8f}")
print(f"  VPU-NoMixUp baseline:    {loss_vpu.item():.8f}")
print(f"  Difference:              {abs(loss_hierarchical.item() - loss_vpu.item()):.8f}")
print()

if abs(loss_hierarchical.item() - loss_vpu.item()) < 1e-4:
    print("✓ VPU initialization matches VPU-NoMixUp baseline!")
else:
    print("✗ VPU initialization does NOT match VPU-NoMixUp baseline")
    print("  This might be due to numerical differences in the implementation")

print("="*70)
