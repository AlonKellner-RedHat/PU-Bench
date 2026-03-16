#!/usr/bin/env python3
"""Train with FIXED meta-trainer that preserves computational graph."""

import torch
import yaml
from pathlib import Path

from meta_trainer_fixed import ToyMetaTrainerFixed


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
config['meta_iterations'] = 100  # Shorter test

device = get_device(config)

print("="*70)
print("TESTING FIXED META-TRAINER")
print("="*70)
print(f"Device: {device}")
print(f"Config: toy_gaussian_meta.yaml (modified for testing)")
print()

# Create fixed trainer
trainer = ToyMetaTrainerFixed(config, device=device)

# Train
trainer.train(num_iterations=config['meta_iterations'])

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Initial:  a1=?, a2=?, a3=? (random)")
print(f"Final:    {trainer.loss_fn}")
print(f"Optimal:  a1=0, a2=0, a3=-1 (BCE)")
print()
print("Check if parameters moved towards optimal values!")
