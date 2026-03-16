#!/usr/bin/env python3
"""Test validation at iteration 0 (before any meta-learning)."""

import torch
import torch.nn as nn
import numpy as np
import yaml
from loss.hierarchical_pu_loss import HierarchicalPULoss
from loss.baseline_losses import PUDRaNaiveLoss, VPUNoMixUpLoss
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
device = 'cpu'

# Create learned loss with PUDRa initialization (BEFORE any meta-learning)
loss_fn = HierarchicalPULoss(
    init_mode='pudra_inspired',
    init_scale=0.01,
    l1_lambda=0.0  # No L1 for this test
).to(device)

print("="*70)
print("ITERATION 0 VALIDATION TEST")
print("="*70)
print("Testing learned loss at initialization (before meta-learning)")
print()
print("Initial loss parameters:")
print(loss_fn)
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
    'learned': [],
    'pudra_naive': [],
    'vpu_nomixup': []
}

print("Training models from scratch on validation tasks...")
print()

for task_idx, task in enumerate(val_tasks):
    print(f"Task {task_idx + 1}/3:")
    dataloaders = task.get_dataloaders(batch_size=64, num_train=1000, num_val=500, num_test=500)

    # === Learned (pudra_inspired init) ===
    model_learned = SimpleMLP(2, [32, 32]).to(device)
    optimizer_learned = torch.optim.Adam(model_learned.parameters(), lr=0.001)

    for epoch in range(train_epochs):
        model_learned.train()
        for batch in dataloaders['train']:
            x, y_pu = batch[0].to(device), batch[2].to(device)
            optimizer_learned.zero_grad()
            outputs = model_learned(x).squeeze(-1)
            loss = loss_fn(outputs, y_pu, mode='pu')
            loss.backward()
            optimizer_learned.step()

    model_learned.eval()
    with torch.no_grad():
        test_loss = 0.0
        for batch in dataloaders['test']:
            x, y_true = batch[0].to(device), batch[1].to(device)
            outputs = model_learned(x).squeeze(-1)
            loss = bce_fn(outputs, y_true)
            test_loss += loss.item()
        test_loss /= len(dataloaders['test'])
    results['learned'].append(test_loss)
    print(f"  Learned:      {test_loss:.6f}")

    # === PUDRa-naive ===
    model_pudra = SimpleMLP(2, [32, 32]).to(device)
    optimizer_pudra = torch.optim.Adam(model_pudra.parameters(), lr=0.001)
    pudra_loss = PUDRaNaiveLoss().to(device)

    for epoch in range(train_epochs):
        model_pudra.train()
        for batch in dataloaders['train']:
            x, y_pu = batch[0].to(device), batch[2].to(device)
            optimizer_pudra.zero_grad()
            outputs = model_pudra(x).squeeze(-1)
            loss = pudra_loss(outputs, y_pu, mode='pu')
            loss.backward()
            optimizer_pudra.step()

    model_pudra.eval()
    with torch.no_grad():
        test_loss = 0.0
        for batch in dataloaders['test']:
            x, y_true = batch[0].to(device), batch[1].to(device)
            outputs = model_pudra(x).squeeze(-1)
            loss = bce_fn(outputs, y_true)
            test_loss += loss.item()
        test_loss /= len(dataloaders['test'])
    results['pudra_naive'].append(test_loss)
    print(f"  PUDRa-naive:  {test_loss:.6f}")

    # === VPU-NoMixUp ===
    model_vpu = SimpleMLP(2, [32, 32]).to(device)
    optimizer_vpu = torch.optim.Adam(model_vpu.parameters(), lr=0.001)
    vpu_loss = VPUNoMixUpLoss().to(device)

    for epoch in range(train_epochs):
        model_vpu.train()
        for batch in dataloaders['train']:
            x, y_pu = batch[0].to(device), batch[2].to(device)
            optimizer_vpu.zero_grad()
            outputs = model_vpu(x).squeeze(-1)
            loss = vpu_loss(outputs, y_pu, mode='pu')
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
    results['vpu_nomixup'].append(test_loss)
    print(f"  VPU-NoMixUp:  {test_loss:.6f}")
    print()

# Average across tasks
print("="*70)
print("AVERAGED RESULTS (iteration 0, before meta-learning)")
print("="*70)
learned_avg = np.mean(results['learned'])
pudra_avg = np.mean(results['pudra_naive'])
vpu_avg = np.mean(results['vpu_nomixup'])

print(f"Learned (pudra_inspired):  {learned_avg:.6f}")
print(f"PUDRa-naive:                {pudra_avg:.6f}")
print(f"VPU-NoMixUp:                {vpu_avg:.6f}")
print()

diff_pudra = abs(learned_avg - pudra_avg)
print(f"Difference from PUDRa-naive: {diff_pudra:.6f}")

if diff_pudra < 0.02:
    print("✓ Learned loss at iteration 0 matches PUDRa-naive!")
else:
    print("✗ Learned loss at iteration 0 differs from PUDRa-naive")

print("="*70)
