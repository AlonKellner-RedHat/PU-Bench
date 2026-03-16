#!/usr/bin/env python3
"""Test that VPU-initialized loss produces same performance as VPU baseline at iteration 0."""

import torch
import torch.nn as nn
import numpy as np
import yaml
from loss.hierarchical_pu_loss import HierarchicalPULoss
from loss.baseline_losses import VPUNoMixUpLoss
from tasks.gaussian_task import GaussianBlobTask
from models.simple_mlp import SimpleMLP


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if len(config) == 1 and isinstance(list(config.values())[0], dict):
        config = list(config.values())[0]
    return config


# Load config
config = load_config('config/toy_gaussian_meta_large_pool.yaml')
device = 'cpu'  # Use CPU for reproducibility

# Create both losses
hierarchical_vpu = HierarchicalPULoss(
    init_mode='vpu_inspired',
    l1_lambda=0.0  # No L1 for this test
).to(device)

vpu_baseline = VPUNoMixUpLoss().to(device)

print("="*70)
print("ITERATION 0 VALIDATION TEST (VPU INITIALIZATION)")
print("="*70)
print("Testing that VPU-initialized hierarchical loss matches VPU-NoMixUp baseline")
print("BEFORE any meta-learning (iteration 0)")
print()
print("Hierarchical loss parameters:")
print(hierarchical_vpu)
print()

# Create same validation tasks as training script
print("Creating validation tasks (same seeds as training)...")
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

# Train from scratch with each method
bce_fn = nn.BCEWithLogitsLoss()
train_epochs = 50

results = {
    'hierarchical_vpu': [],
    'vpu_baseline': []
}

print("Training models from scratch on validation tasks...")
print()

for task_idx, task in enumerate(val_tasks):
    print(f"Task {task_idx + 1}/3:")
    dataloaders = task.get_dataloaders(batch_size=64, num_train=1000, num_val=500, num_test=500)

    # === Train with HIERARCHICAL VPU (iteration 0) ===
    model_hierarchical = SimpleMLP(2, [32, 32]).to(device)
    optimizer_hierarchical = torch.optim.Adam(model_hierarchical.parameters(), lr=0.001)

    for epoch in range(train_epochs):
        model_hierarchical.train()
        for batch in dataloaders['train']:
            x, y_pu = batch[0].to(device), batch[2].to(device)
            optimizer_hierarchical.zero_grad()
            outputs = model_hierarchical(x).squeeze(-1)
            loss = hierarchical_vpu(outputs, y_pu, mode='pu')
            loss.backward()
            optimizer_hierarchical.step()

    model_hierarchical.eval()
    with torch.no_grad():
        test_loss = 0.0
        for batch in dataloaders['test']:
            x, y_true = batch[0].to(device), batch[1].to(device)
            outputs = model_hierarchical(x).squeeze(-1)
            loss = bce_fn(outputs, y_true)
            test_loss += loss.item()
        test_loss /= len(dataloaders['test'])
    results['hierarchical_vpu'].append(test_loss)
    print(f"  Hierarchical VPU (iter 0): {test_loss:.6f}")

    # === Train with VPU-NoMixUp baseline ===
    model_vpu = SimpleMLP(2, [32, 32]).to(device)
    optimizer_vpu = torch.optim.Adam(model_vpu.parameters(), lr=0.001)

    for epoch in range(train_epochs):
        model_vpu.train()
        for batch in dataloaders['train']:
            x, y_pu = batch[0].to(device), batch[2].to(device)
            optimizer_vpu.zero_grad()
            outputs = model_vpu(x).squeeze(-1)
            loss = vpu_baseline(outputs, y_pu, mode='pu')
            loss.backward()
            optimizer_vpu.step()

    model_vpu.eval()
    with torch.no_grad():
        test_loss = 0.0
        for batch in dataloaders['test']:
            x, y_true = batch[0].to(device), batch[1].to(device)
            outputs = model_vpu(x).squeeze(-1)
            loss = bce_fn(outputs, y_true)
            test_loss += loss.item()
        test_loss /= len(dataloaders['test'])
    results['vpu_baseline'].append(test_loss)
    print(f"  VPU-NoMixUp baseline:      {test_loss:.6f}")
    print()

# Average across tasks
print("="*70)
print("AVERAGED RESULTS (iteration 0, before meta-learning)")
print("="*70)
hierarchical_avg = np.mean(results['hierarchical_vpu'])
vpu_avg = np.mean(results['vpu_baseline'])

print(f"Hierarchical VPU (iter 0): {hierarchical_avg:.6f}")
print(f"VPU-NoMixUp baseline:      {vpu_avg:.6f}")
print(f"Difference:                {abs(hierarchical_avg - vpu_avg):.6f}")
print()

if abs(hierarchical_avg - vpu_avg) < 0.02:
    print("✓ Hierarchical VPU at iteration 0 matches VPU-NoMixUp baseline!")
    print("  The initialization is correct.")
else:
    print("✗ Hierarchical VPU at iteration 0 differs from VPU-NoMixUp baseline")
    print("  There may be an issue with the initialization.")

print("="*70)
print()
print("Individual task results:")
for i in range(3):
    print(f"Task {i+1}: Hierarchical={results['hierarchical_vpu'][i]:.6f}, "
          f"Baseline={results['vpu_baseline'][i]:.6f}, "
          f"Diff={abs(results['hierarchical_vpu'][i] - results['vpu_baseline'][i]):.6f}")
print("="*70)
