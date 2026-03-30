#!/usr/bin/env python3
"""
Generate dataset configuration files for VPU variants rerun.

Creates configs for:
- 6 core datasets (MNIST, FashionMNIST, IMDB, 20News, Mushrooms, Spambase)
- 5 seeds (42, 123, 456, 789, 2024)
- All c and prior combinations

Outputs to config/vpu_rerun/
"""

import os
import yaml
from pathlib import Path

# Configuration
DATASETS = {
    'MNIST': {
        'dataset_class': 'MNIST',
        'label_scheme': {
            'positive_classes': [0, 2, 4, 6, 8],
            'negative_classes': [1, 3, 5, 7, 9]
        }
    },
    'FashionMNIST': {
        'dataset_class': 'FashionMNIST',
        'label_scheme': None  # Uses default binary labels
    },
    'IMDB': {
        'dataset_class': 'IMDB',
        'label_scheme': None
    },
    '20News': {
        'dataset_class': '20News',
        'label_scheme': None
    },
    'Mushrooms': {
        'dataset_class': 'Mushrooms',
        'label_scheme': None
    },
    'Spambase': {
        'dataset_class': 'Spambase',
        'label_scheme': None
    }
}

SEEDS = [42, 123, 456, 789, 2024]

# Hyperparameters (from existing vary_c and vary_prior configs)
C_VALUES = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
PRIOR_VALUES = [None, 0.1, 0.3, 0.5, 0.7, 0.9]  # None = natural prior

# Output directory
OUTPUT_DIR = Path('config/vpu_rerun')


def generate_config(dataset_name, dataset_info):
    """Generate config for a dataset with all seeds and hyperparameter combinations."""
    config = {
        'dataset_class': dataset_info['dataset_class'],
        'data_dir': './datasets',
        'random_seeds': SEEDS,
        'c_values': C_VALUES,
        'scenarios': ['case-control'],
        'selection_strategies': ['random'],
        'val_ratio': 0.01,
        'target_prevalence': PRIOR_VALUES,
        'with_replacement': True,
        'also_print_dataset_stats': False
    }

    # Add label_scheme if specified
    if dataset_info['label_scheme'] is not None:
        config['label_scheme'] = dataset_info['label_scheme']

    return config


def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating VPU rerun configs in {OUTPUT_DIR}/")
    print(f"Datasets: {list(DATASETS.keys())}")
    print(f"Seeds: {SEEDS}")
    print(f"C values: {C_VALUES}")
    print(f"Prior values: {PRIOR_VALUES}")
    print()

    for dataset_name, dataset_info in DATASETS.items():
        config = generate_config(dataset_name, dataset_info)

        # Save config file
        output_file = OUTPUT_DIR / f"{dataset_name.lower()}.yaml"
        with open(output_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"✓ Created {output_file}")

    print()
    print(f"Generated {len(DATASETS)} config files.")
    print()
    print("Next steps:")
    print("1. Review configs in config/vpu_rerun/")
    print("2. Run: bash scripts/run_vpu_core_datasets.sh")


if __name__ == '__main__':
    main()
