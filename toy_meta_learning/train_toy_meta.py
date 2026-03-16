#!/usr/bin/env python3
"""Main script for toy meta-learning example.

Usage:
    python train_toy_meta.py --config config/toy_gaussian_meta.yaml
"""

import torch
import yaml
import argparse
from pathlib import Path

from meta_trainer import ToyMetaTrainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Flatten nested dict (if any)
    # Some configs might have a top-level key with the experiment name
    if len(config) == 1 and isinstance(list(config.values())[0], dict):
        config = list(config.values())[0]

    return config


def get_device(config: dict) -> str:
    """Get device from config.

    Args:
        config: Configuration dictionary

    Returns:
        Device string ('cpu', 'cuda', or 'mps')
    """
    device_config = config.get('device', 'auto')

    if device_config == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    else:
        return device_config


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Toy Meta-Learning for PN Loss')
    parser.add_argument(
        '--config',
        type=str,
        default='config/toy_gaussian_meta.yaml',
        help='Path to configuration file',
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Get device
    device = get_device(config)

    print("="*70)
    print("TOY META-LEARNING SETUP")
    print("="*70)
    print(f"Config: {args.config}")
    print(f"Mode: {config.get('mode', 'pu')}")
    print(f"Device: {device}")
    print(f"Loss init mode: {config.get('loss_init_mode', 'random')}")
    print(f"Meta-iterations: {config.get('meta_iterations', 200)}")
    print()

    # Create meta-trainer
    meta_trainer = ToyMetaTrainer(config, device=device)

    # Run meta-training
    meta_trainer.train(num_iterations=config.get('meta_iterations', 200))

    print("\nTOY META-LEARNING COMPLETE!")
    print(f"Output saved to: {meta_trainer.save_dir}")


if __name__ == '__main__':
    main()
