#!/usr/bin/env python3
"""Test that pudra_inspired initialization matches PUDRa-naive baseline."""

import torch
import torch.nn as nn
from loss.hierarchical_pu_loss import HierarchicalPULoss
from loss.baseline_losses import PUDRaNaiveLoss

# Create both losses
hierarchical = HierarchicalPULoss(init_mode='pudra_inspired', l1_lambda=0.0)
pudra_baseline = PUDRaNaiveLoss()

# Test on random data
torch.manual_seed(42)
batch_size = 100
outputs = torch.randn(batch_size)
pu_labels = torch.where(torch.rand(batch_size) < 0.3,
                        torch.ones(batch_size),
                        -torch.ones(batch_size))

# Compute losses
loss_hierarchical = hierarchical(outputs, pu_labels, mode='pu')
loss_pudra = pudra_baseline(outputs, pu_labels, mode='pu')

print("="*70)
print("PUDRA-INSPIRED INITIALIZATION TEST")
print("="*70)
print(f"Hierarchical loss (pudra_inspired): {loss_hierarchical.item():.8f}")
print(f"PUDRa-naive baseline:                {loss_pudra.item():.8f}")
print(f"Difference:                          {abs(loss_hierarchical.item() - loss_pudra.item()):.8f}")
print()

if abs(loss_hierarchical.item() - loss_pudra.item()) < 1e-5:
    print("✓ Initialization MATCHES PUDRa-naive baseline!")
else:
    print("✗ Initialization does NOT match PUDRa-naive baseline")
    print()
    print("Hierarchical parameters:")
    print(hierarchical)

print("="*70)
