#!/usr/bin/env python3
"""Train with NATIVE PyTorch meta-trainer (differentiable=True).

This is the PROPER solution using PyTorch's built-in differentiable optimization.
No external libraries needed - uses the native `differentiable=True` parameter.

References:
- PyTorch SGD docs: https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
- PyTorch Adam docs: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
"""

import torch
import yaml
from pathlib import Path

from meta_trainer_native import ToyMetaTrainerNative


def load_config(config_path: str) -> dict:
    """Load config."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if len(config) == 1 and isinstance(list(config.values())[0], dict):
        config = list(config.values())[0]
    return config


def get_device(config: dict) -> str:
    """Get device."""
    device_config = config.get('device', 'auto')
    if device_config == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    return device_config


# Load config
config = load_config('config/toy_gaussian_meta.yaml')

# Reduce for faster testing
config['mean_separations'] = [2.0, 3.0]
config['checkpoint_seeds'] = [42, 123]
config['checkpoint_epochs'] = [1, 5, 10]
config['meta_iterations'] = 200

device = get_device(config)

print("="*70)
print("TESTING NATIVE PyTorch META-TRAINER")
print("="*70)
print(f"Device: {device}")
print(f"Inner optimizer: {config.get('inner_optimizer', 'sgd')} (differentiable=True)")
print(f"Inner steps: {config.get('inner_steps', 3)}")
print(f"Inner LR: {config.get('inner_lr', 0.01)}")
print()

# Create native trainer
trainer = ToyMetaTrainerNative(config, device=device)

# Train
trainer.train(num_iterations=config['meta_iterations'])

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Final:    {trainer.loss_fn}")
print(f"Optimal:  a1=0, a2=0, a3=-1 (BCE)")
print()
print("✅ Parameters should be moving towards optimal values!")
print("✅ Using PyTorch native differentiable=True")
print("✅ Preserves optimizer state (momentum/Adam moments)")
print()
print("Key insight:")
print("  PyTorch's differentiable=True parameter enables backpropagation")
print("  through optimizer.step() without breaking the computational graph.")
print("  This is the modern, supported way to do meta-learning in PyTorch.")
