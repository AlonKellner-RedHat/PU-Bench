#!/usr/bin/env python3
"""Analyze why meta-learning doesn't converge to pure BCE.

After 1000 iterations:
  a1 = 0.0218  (should be 0)
  a2 = -0.9525 (should be 0)  ← Why is this non-zero?
  a3 = -0.9674 (should be -1)
"""

import torch
import yaml
import numpy as np

from meta_trainer_torch_func import ToyMetaTrainerTorchFunc
from loss.simple_basis_loss import SimpleBasisLoss


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if len(config) == 1:
        config = list(config.values())[0]
    return config


# Load config
config = load_config('config/toy_gaussian_meta.yaml')
config['device'] = 'mps'
config['mean_separations'] = [2.0, 3.0]
config['checkpoint_seeds'] = [42, 123]
config['checkpoint_epochs'] = [1, 5, 10]

device = 'mps'

# Create trainer (just to get checkpoint pool)
trainer = ToyMetaTrainerTorchFunc(config, device=device)

print("="*70)
print("CONVERGENCE ANALYSIS")
print("="*70)
print()

# Test different loss configurations
loss_configs = [
    ("Learned (1000 iter)", {"a1": 0.0218, "a2": -0.9525, "a3": -0.9674}),
    ("Pure BCE", {"a1": 0.0, "a2": 0.0, "a3": -1.0}),
    ("No linear term", {"a1": 0.0, "a2": 0.0, "a3": -0.9674}),
    ("With linear term", {"a1": 0.0, "a2": -0.9525, "a3": -1.0}),
]

# Sample some checkpoints and evaluate
checkpoint_indices = trainer.checkpoint_pool.sample_batch(12)  # All checkpoints

for loss_name, params in loss_configs:
    # Create loss with these parameters
    loss_fn = SimpleBasisLoss(init_mode='random').to(device)
    loss_fn.a1.data = torch.tensor([params["a1"]], device=device)
    loss_fn.a2.data = torch.tensor([params["a2"]], device=device)
    loss_fn.a3.data = torch.tensor([params["a3"]], device=device)

    total_bce = 0.0
    num_checkpoints = 0

    for ckpt_idx in checkpoint_indices:
        checkpoint, task, dataloaders = trainer.checkpoint_pool.get_checkpoint(ckpt_idx)

        # Load model
        from models.simple_mlp import SimpleMLP
        model = SimpleMLP(
            input_dim=checkpoint['task_config']['num_dimensions'],
            hidden_dims=config.get('model_hidden_dims', [32, 32]),
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Adapt with K=3 steps using this loss
        adapted_params = trainer.inner_loop(
            model=model,
            train_loader=dataloaders['train'],
            num_steps=3,
            inner_lr=config.get('inner_lr', 0.01),
        )

        # Evaluate on validation with BCE
        val_bce = trainer.evaluate_bce(model, adapted_params, dataloaders['val'])
        total_bce += val_bce.item()
        num_checkpoints += 1

    avg_bce = total_bce / num_checkpoints

    print(f"{loss_name:20s}: Val BCE = {avg_bce:.6f}")
    print(f"  Params: a1={params['a1']:.4f}, a2={params['a2']:.4f}, a3={params['a3']:.4f}")
    print()

print("="*70)
print("ANALYSIS")
print("="*70)
print()
print("If the learned loss performs BETTER than pure BCE, it means:")
print("  1. The linear term (a2) helps with the specific task distribution")
print("  2. Meta-learning found a better inductive bias than pure BCE")
print("  3. This is actually CORRECT behavior!")
print()
print("If pure BCE performs BETTER, it means:")
print("  1. Meta-learning got stuck in a local minimum")
print("  2. Learning rate or regularization needs tuning")
print("  3. Need more iterations or better initialization")
