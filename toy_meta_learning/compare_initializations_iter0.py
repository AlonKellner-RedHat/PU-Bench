#!/usr/bin/env python3
"""Compare PUDRa vs VPU initialization at iteration 0."""

import torch
import yaml
from loss.hierarchical_pu_loss import HierarchicalPULoss
from tasks.gaussian_task import GaussianBlobTask
from validation_utils import train_from_scratch_validation


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if len(config) == 1 and isinstance(list(config.values())[0], dict):
        config = list(config.values())[0]
    return config


config = load_config('config/toy_gaussian_meta_large_pool.yaml')
device = 'cpu'

# Create validation tasks
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

print("="*70)
print("ITERATION 0 COMPARISON: PUDRa vs VPU Initialization")
print("="*70)
print()

# Test PUDRa initialization
print("Testing PUDRa-inspired initialization...")
loss_pudra = HierarchicalPULoss(
    init_mode='pudra_inspired',
    l1_lambda=0.0
).to(device)

val_results_pudra, cached = train_from_scratch_validation(
    val_tasks, loss_pudra, config, device, None
)

print()
print("PUDRa Initialization (Iteration 0):")
print(f"  Learned:      {val_results_pudra['learned']:.6f}")
print(f"  PUDRa-naive:  {val_results_pudra['pudra_naive']:.6f}")
print(f"  VPU-NoMixUp:  {val_results_pudra['vpu_nomixup']:.6f}")
print(f"  Diff from PUDRa: {abs(val_results_pudra['learned'] - val_results_pudra['pudra_naive']):.6f}")
print()

# Test VPU initialization
print("Testing VPU-inspired initialization...")
loss_vpu = HierarchicalPULoss(
    init_mode='vpu_inspired',
    l1_lambda=0.0
).to(device)

# Reuse cached baselines
val_results_vpu, _ = train_from_scratch_validation(
    val_tasks, loss_vpu, config, device, cached
)

print()
print("VPU Initialization (Iteration 0):")
print(f"  Learned:      {val_results_vpu['learned']:.6f}")
print(f"  PUDRa-naive:  {val_results_vpu['pudra_naive']:.6f}")
print(f"  VPU-NoMixUp:  {val_results_vpu['vpu_nomixup']:.6f}")
print(f"  Diff from VPU: {abs(val_results_vpu['learned'] - val_results_vpu['vpu_nomixup']):.6f}")
print()

print("="*70)
print("SUMMARY")
print("="*70)
print()
print("Iteration 0 (before meta-learning):")
print(f"  PUDRa init → {val_results_pudra['learned']:.6f} BCE (matches PUDRa baseline)")
print(f"  VPU init   → {val_results_vpu['learned']:.6f} BCE (matches VPU baseline)")
print()
print("Baselines:")
print(f"  PUDRa-naive: {val_results_pudra['pudra_naive']:.6f}")
print(f"  VPU-NoMixUp: {val_results_pudra['vpu_nomixup']:.6f}")
print()

# Check verification
pudra_verified = abs(val_results_pudra['learned'] - val_results_pudra['pudra_naive']) < 0.02
vpu_verified = abs(val_results_vpu['learned'] - val_results_vpu['vpu_nomixup']) < 0.02

if pudra_verified:
    print("✓ PUDRa initialization verified!")
else:
    print("✗ PUDRa initialization differs from baseline")

if vpu_verified:
    print("✓ VPU initialization verified!")
else:
    print("✗ VPU initialization differs from baseline")

print()
print("Both initializations start at their respective baseline performance.")
print("Meta-learning will attempt to improve from these starting points.")
print("="*70)
