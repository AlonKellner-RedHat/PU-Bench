#!/usr/bin/env python3
"""Train with torch.func meta-trainer (THE MODERN SOLUTION).

This uses PyTorch's NATIVE torch.func API (introduced in PyTorch 2.0):
- torch.func.functional_call: Pass arbitrary params without modifying model state
- torch.func.grad: Functional gradient computation (JAX-style)
- No external libraries needed (higher, torchopt, learn2learn are obsolete)

This is the actively maintained, official PyTorch solution for meta-learning.

Reference: https://pytorch.org/docs/stable/func.html
"""

import torch
import yaml
from pathlib import Path

from meta_trainer_torch_func import ToyMetaTrainerTorchFunc


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
print("TESTING TORCH.FUNC META-TRAINER")
print("="*70)
print(f"Device: {device}")
print(f"PyTorch version: {torch.__version__}")
print(f"Inner steps: {config.get('inner_steps', 3)}")
print(f"Inner LR: {config.get('inner_lr', 0.01)}")
print()
print("Modern PyTorch 2.0+ Solution:")
print("  - torch.func.functional_call (stateless model evaluation)")
print("  - torch.func.grad (functional gradients)")
print("  - No external libraries (higher/torchopt/learn2learn obsolete)")
print("  - Clean, readable, officially supported")
print()

# Create torch.func trainer
trainer = ToyMetaTrainerTorchFunc(config, device=device)

# Train
trainer.train(num_iterations=config['meta_iterations'])

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Final:    {trainer.loss_fn}")
print(f"Optimal:  a1=0, a2=0, a3=-1 (BCE)")
print()
print("✅ This is the MODERN, NATIVE PyTorch solution!")
print("✅ No external dependencies")
print("✅ Actively maintained by PyTorch core team")
print("✅ Cleaner than manual functional optimization")
print()
print("torch.func is the future of meta-learning in PyTorch!")
