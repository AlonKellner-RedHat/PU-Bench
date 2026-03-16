#!/usr/bin/env python3
"""Train with FOMAML meta-trainer (First-Order MAML).

Based on PyTorch Lightning tutorial:
https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/12-meta-learning.html

FOMAML is simpler and faster than full second-order MAML:
- Uses only first-order gradients
- No need for create_graph=True
- Standard optimizers work fine
- Detach-and-reattach trick preserves gradients
"""

import torch
import yaml
from pathlib import Path

from meta_trainer_fomaml import ToyMetaTrainerFOMAML


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
print("TESTING FOMAML META-TRAINER")
print("="*70)
print(f"Device: {device}")
print(f"Inner optimizer: {config.get('inner_optimizer', 'sgd')} (standard, no special tricks)")
print(f"Inner steps: {config.get('inner_steps', 3)}")
print(f"Inner LR: {config.get('inner_lr', 0.01)}")
print()
print("First-Order MAML (FOMAML):")
print("  1. Run standard optimizer in inner loop (fast!)")
print("  2. Use detach-and-reattach trick to preserve gradients")
print("  3. No create_graph=True needed")
print("  4. Almost same performance as full MAML")
print()

# Create FOMAML trainer
trainer = ToyMetaTrainerFOMAML(config, device=device)

# Train
trainer.train(num_iterations=config['meta_iterations'])

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Final:    {trainer.loss_fn}")
print(f"Optimal:  a1=0, a2=0, a3=-1 (BCE)")
print()
print("✅ Parameters should be moving towards optimal values!")
print("✅ Using First-Order MAML (simpler, faster)")
print("✅ Standard PyTorch optimizers")
print()
print("FOMAML is the recommended approach for most meta-learning tasks!")
print("Source: https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/12-meta-learning.html")
