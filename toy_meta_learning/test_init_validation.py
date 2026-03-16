#!/usr/bin/env python3
"""Test that pudra_inspired initialization matches PUDRa-naive on validation tasks."""

import torch
import torch.nn as nn
import numpy as np
from loss.hierarchical_pu_loss import HierarchicalPULoss
from loss.baseline_losses import PUDRaNaiveLoss
from tasks.gaussian_task import GaussianBlobTask
from models.simple_mlp import SimpleMLP

device = 'cpu'  # Use CPU for simplicity

# Create both losses
hierarchical = HierarchicalPULoss(init_mode='pudra_inspired', l1_lambda=0.0).to(device)
pudra_baseline = PUDRaNaiveLoss().to(device)

print("="*70)
print("VALIDATION TEST: PUDRa-inspired vs PUDRa-naive baseline")
print("="*70)
print()

# Create a single validation task
task = GaussianBlobTask(
    num_dimensions=2,
    mean_separation=2.5,
    std=1.0,
    prior=0.5,
    labeling_freq=0.3,
    num_samples=1000,
    seed=9000,
    mode='pu',
    negative_labeling_freq=0.3,
)

dataloaders = task.get_dataloaders(batch_size=64, num_train=1000, num_val=500, num_test=500)

# Train with hierarchical loss (pudra_inspired init)
print("Training with hierarchical loss (pudra_inspired)...")
model_hierarchical = SimpleMLP(2, [32, 32]).to(device)
optimizer_hierarchical = torch.optim.Adam(model_hierarchical.parameters(), lr=0.001)
train_epochs = 50

for epoch in range(train_epochs):
    model_hierarchical.train()
    for batch in dataloaders['train']:
        x, y_true, y_pu = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        optimizer_hierarchical.zero_grad()
        outputs = model_hierarchical(x).squeeze(-1)
        loss = hierarchical(outputs, y_pu, mode='pu')
        loss.backward()
        optimizer_hierarchical.step()

# Evaluate with BCE
bce_fn = nn.BCEWithLogitsLoss()
model_hierarchical.eval()
with torch.no_grad():
    test_loss_hierarchical = 0.0
    for batch in dataloaders['test']:
        x, y_true = batch[0].to(device), batch[1].to(device)
        outputs = model_hierarchical(x).squeeze(-1)
        loss = bce_fn(outputs, y_true)
        test_loss_hierarchical += loss.item()
    test_loss_hierarchical /= len(dataloaders['test'])

print(f"Hierarchical BCE: {test_loss_hierarchical:.6f}")

# Train with PUDRa-naive baseline
print("\nTraining with PUDRa-naive baseline...")
model_pudra = SimpleMLP(2, [32, 32]).to(device)
optimizer_pudra = torch.optim.Adam(model_pudra.parameters(), lr=0.001)

for epoch in range(train_epochs):
    model_pudra.train()
    for batch in dataloaders['train']:
        x, y_pu = batch[0].to(device), batch[2].to(device)
        optimizer_pudra.zero_grad()
        outputs = model_pudra(x).squeeze(-1)
        loss = pudra_baseline(outputs, y_pu, mode='pu')
        loss.backward()
        optimizer_pudra.step()

# Evaluate with BCE
model_pudra.eval()
with torch.no_grad():
    test_loss_pudra = 0.0
    for batch in dataloaders['test']:
        x, y_true = batch[0].to(device), batch[1].to(device)
        outputs = model_pudra(x).squeeze(-1)
        loss = bce_fn(outputs, y_true)
        test_loss_pudra += loss.item()
    test_loss_pudra /= len(dataloaders['test'])

print(f"PUDRa-naive BCE:  {test_loss_pudra:.6f}")

print()
print("="*70)
print(f"Difference: {abs(test_loss_hierarchical - test_loss_pudra):.6f}")

if abs(test_loss_hierarchical - test_loss_pudra) < 0.01:
    print("✓ Hierarchical (pudra_inspired) MATCHES PUDRa-naive on validation!")
else:
    print("✗ Hierarchical (pudra_inspired) does NOT match PUDRa-naive")
    print()
    print("This suggests the initialization or training process differs.")

print("="*70)
