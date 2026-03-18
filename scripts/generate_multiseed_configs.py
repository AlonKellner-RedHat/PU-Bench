#!/usr/bin/env python3
"""Generate multi-seed configs for all datasets."""

from pathlib import Path

SEEDS = [123, 456, 789, 2024]
C_VALUES = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
PRIOR_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]

DATASETS = {
    'MNIST': None,
    'FashionMNIST': None,
    'IMDB': './scripts/embeddings/imdb_sbert_embeddings.npz',
    '20News': './scripts/embeddings/20news_sbert_embeddings.npz',
    'Connect4': None,
    'Spambase': None,
    'Mushrooms': None,
}


def create_vary_c_config(dataset, embeddings_path):
    """Create vary_c config for a dataset."""
    config = f"""dataset_class: {dataset}
data_dir: ./datasets
random_seeds: {SEEDS}

c_values: {C_VALUES}
scenarios: [case-control]
selection_strategies: ["random"]

val_ratio: 0.01
target_prevalence: null
with_replacement: true
also_print_dataset_stats: false
"""

    if embeddings_path:
        config += f"""
# SBERT encoding settings - use precomputed embeddings!
sbert_model_name: all-MiniLM-L6-v2
sbert_embeddings_path: {embeddings_path}
"""

    config += """
label_scheme:
  true_positive_label: 1
  true_negative_label: 0
  pu_labeled_label: 1
  pu_unlabeled_label: -1
"""

    filename = f"config/datasets_multiseed/vary_c_{dataset.lower()}_multiseed.yaml"
    Path(filename).write_text(config)
    return filename


def create_vary_prior_config(dataset, embeddings_path):
    """Create vary_prior config for a dataset."""
    config = f"""dataset_class: {dataset}
data_dir: ./datasets
random_seeds: {SEEDS}

c_values: [0.1]
scenarios: [case-control]
selection_strategies: ["random"]

val_ratio: 0.01
target_prevalence: {PRIOR_VALUES}
with_replacement: true
also_print_dataset_stats: false
"""

    if embeddings_path:
        config += f"""
# SBERT encoding settings - use precomputed embeddings!
sbert_model_name: all-MiniLM-L6-v2
sbert_embeddings_path: {embeddings_path}
"""

    config += """
label_scheme:
  true_positive_label: 1
  true_negative_label: 0
  pu_labeled_label: 1
  pu_unlabeled_label: -1
"""

    filename = f"config/datasets_multiseed/vary_prior_{dataset.lower()}_multiseed.yaml"
    Path(filename).write_text(config)
    return filename


def main():
    print("Creating multi-seed configs...")

    created = []
    for dataset, embeddings_path in DATASETS.items():
        f1 = create_vary_c_config(dataset, embeddings_path)
        f2 = create_vary_prior_config(dataset, embeddings_path)
        created.extend([f1, f2])
        print(f"  Created: {Path(f1).name}, {Path(f2).name}")

    print(f"\nDone! Created {len(created)} configs")
    print(f"\nEstimated runs: {len(DATASETS)} datasets × 12 configs × 5 methods × 4 seeds = {len(DATASETS) * 12 * 5 * 4}")


if __name__ == '__main__':
    main()
