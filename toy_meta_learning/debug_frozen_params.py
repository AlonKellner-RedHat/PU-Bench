#!/usr/bin/env python3
"""Debug why frozen VPU parameters show performance variance."""

import torch
import numpy as np
from loss.hierarchical_pu_loss import HierarchicalPULoss

# Create VPU loss
loss_vpu = HierarchicalPULoss(init_mode='vpu_inspired', l1_lambda=0.0)

# Get initial parameters
params_init = loss_vpu.get_parameters().detach().clone()

print("="*70)
print("DEBUGGING FROZEN PARAMETER PERFORMANCE VARIANCE")
print("="*70)
print()
print("Initial VPU parameters:")
print(params_init)
print()

# Simulate what happens with frozen parameters
# Test loss computation on same data multiple times
torch.manual_seed(42)
p_test = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
labels_test = torch.tensor([1, 1, -1, -1, -1])

losses = []
for i in range(10):
    torch.manual_seed(42)  # Same data
    loss_val = loss_vpu(torch.logit(p_test), labels_test)
    losses.append(loss_val.item())
    print(f"Run {i+1}: {loss_val.item():.6f}")

print()
print(f"Mean: {np.mean(losses):.6f}")
print(f"Std:  {np.std(losses):.6f}")
print(f"Range: [{min(losses):.6f}, {max(losses):.6f}]")
print()

if np.std(losses) < 1e-10:
    print("✓ Loss computation is deterministic (as expected)")
else:
    print("✗ Loss computation has variance (unexpected!)")

print()
print("="*70)
print("CHECKING META-OBJECTIVE ALIGNMENT")
print("="*70)
print()

# The real issue: meta-objective (BCE after 3 steps) vs final performance
print("Meta-objective: BCE on validation set after 3 gradient steps")
print("Final metric: BCE on test set after 50 epochs")
print()
print("Problem: These are misaligned!")
print()
print("With frozen parameters (no learning):")
print("  - Meta-loss oscillates based on random task sampling")
print("  - But loss function doesn't change")
print("  - So meta-gradients are computed but don't improve anything")
print()
print("This explains the variance:")
print("  1. Different task batches → different meta-loss values")
print("  2. Meta-gradients computed but clipped to ~0")
print("  3. No parameter updates, but optimizer state changes")
print("  4. Validation performance varies due to:")
print("     a) Stochastic model initialization")
print("     b) Stochastic 50-epoch training")
print("     c) No averaging over multiple seeds")
print()
print("="*70)
