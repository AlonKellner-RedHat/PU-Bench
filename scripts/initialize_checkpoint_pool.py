"""Initialize checkpoint pool from PN naive training.

This script implements Phase 1 of the meta-learning pipeline:
1. Loads training task split from task_split.yaml
2. For each training task, trains a model with PN naive loss
3. Saves checkpoints at multiple training stages (epochs 0, 1, 5, 10, 15, 20, 25, 30, 35, 40)
4. Uses multiple random seeds for diversity (4 seeds per task)
5. Stores all checkpoints in a CheckpointPool for meta-learning

Usage:
    python scripts/initialize_checkpoint_pool.py \\
        --split task_split.yaml \\
        --output ./meta_checkpoints \\
        --epochs 0,1,5,10,15,20,25,30,35,40 \\
        --seeds 42,123,456,789
"""

import argparse
import copy
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
import os
import random

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
from tqdm import tqdm

from meta_learning.checkpoint_pool import CheckpointPool, load_task_split
from config.run_param_sweep import load_dataset_config, expand_dataset_grid


# Map dataset names to their config file paths
DATASET_CONFIG_PATHS = {
    "mnist": "config/datasets_typical/param_sweep_mnist_single.yaml",
    "fashionmnist": "config/datasets_typical/param_sweep_fashionmnist_single.yaml",
    "cifar10": "config/datasets_typical/param_sweep_cifar10_single.yaml",
    "alzheimermri": "config/datasets_typical/param_sweep_alzheimermri_single.yaml",
    "mushrooms": "config/datasets_typical/param_sweep_mushrooms_single.yaml",
    "spambase": "config/datasets_typical/param_sweep_spambase_single.yaml",
    "connect4": "config/datasets_typical/param_sweep_connect4_single.yaml",
    "20news": "config/datasets_typical/param_sweep_20news_single.yaml",
    "imdb": "config/datasets_typical/param_sweep_imdb_single.yaml",
}


def get_dataset_class(dataset: str) -> str:
    """Convert dataset name to dataset_class format."""
    mapping = {
        'mnist': 'MNIST',
        'fashionmnist': 'FashionMNIST',
        'cifar10': 'CIFAR10',
        'alzheimermri': 'AlzheimerMRI',
        'mushrooms': 'Mushrooms',
        'spambase': 'Spambase',
        'connect4': 'Connect4',
        '20news': '20News',
        'imdb': 'IMDB'
    }
    return mapping.get(dataset, dataset.upper())


def train_checkpoint_with_trainer(
    dataset: str,
    c_value: float,
    prior: float,
    seed: int,
    target_epoch: int,
    batch_size: int = 256,
    lr: float = 0.001
) -> torch.nn.Module:
    """Train checkpoint using PNNaiveTrainer (same as benchmark).

    Args:
        dataset: Dataset name (e.g., 'mnist', 'mushrooms', '20news')
        c_value: Label noise level
        prior: Class prior
        seed: Random seed
        target_epoch: Number of epochs to train
        batch_size: Training batch size
        lr: Learning rate

    Returns:
        Trained model
    """
    # Import trainer
    from train.pn_naive_trainer import PNNaiveTrainer

    # Get dataset config path
    config_path = DATASET_CONFIG_PATHS[dataset]

    # Load dataset config (same as benchmark and profiling)
    dataset_cfg = load_dataset_config(config_path)
    dataset_class, data_runs = expand_dataset_grid(dataset_cfg)

    # Get the first run config (includes SBERT settings for text datasets)
    base_params = copy.deepcopy(data_runs[0]) if data_runs else {}

    # Override with checkpoint-specific values
    params = base_params
    params.update({
        # Override dataset params for checkpoint
        'labeled_ratio': c_value,
        'target_prevalence': prior,
        'random_seed': seed,
        'val_ratio': 0.1,

        # Training config
        'num_epochs': target_epoch,
        'batch_size': batch_size,
        'lr': lr,
        'weight_decay': 1e-4,
        'optimizer': 'adam',

        # Model config
        'init_bias_from_prior': True,  # Initialize from prior

        # Logging (suppress during pool creation)
        'log_interval': 999,
        'silence_metrics_before_epoch': 999,
    })

    # Create experiment name
    exp_name = f"{dataset}_c{c_value:.1f}_prior{prior:.1f}_seed{seed}"

    # Train using PNNaiveTrainer
    trainer = PNNaiveTrainer(
        method='pn_naive',
        experiment=exp_name,
        params=params
    )

    # Handle epoch 0 (random initialization)
    if target_epoch == 0:
        trainer.before_training()  # Initialize model only
        trainer.after_training()   # Cleanup
    else:
        trainer.run()  # Full training

    return trainer.model


def train_task_worker(args):
    """Worker function to train all checkpoints for a single task.

    Args:
        args: Tuple of (task_idx, total_tasks, dataset, c_value, prior, seed,
              checkpoint_epochs, batch_size, lr, checkpoints_dir)

    Returns:
        Tuple of (task_id, trained_count, skipped_count)
    """
    (task_idx, total_tasks, dataset, c_value, prior, seed,
     checkpoint_epochs, batch_size, lr, checkpoints_dir) = args

    task_id = f"{dataset}_c{c_value:.1f}_prior{prior:.1f}_seed{seed}"

    # Check which checkpoints already exist
    existing_epochs = []
    missing_epochs = []
    for epoch in checkpoint_epochs:
        checkpoint_filename = f"{task_id}_epoch{epoch}_pn_naive.pt"
        checkpoint_path = checkpoints_dir / checkpoint_filename
        if checkpoint_path.exists():
            existing_epochs.append(epoch)
        else:
            missing_epochs.append(epoch)

    # Skip if all exist
    if not missing_epochs:
        return (task_id, 0, len(checkpoint_epochs))

    # Train missing checkpoints
    trained = 0
    for epoch in missing_epochs:
        try:
            model = train_checkpoint_with_trainer(
                dataset=dataset,
                c_value=c_value,
                prior=prior,
                seed=seed,
                target_epoch=epoch,
                batch_size=batch_size,
                lr=lr
            )

            # Save checkpoint immediately
            checkpoint_filename = f"{task_id}_epoch{epoch}_pn_naive.pt"
            checkpoint_path = checkpoints_dir / checkpoint_filename
            torch.save(model.state_dict(), checkpoint_path)
            trained += 1
        except Exception as e:
            print(f"ERROR training {task_id} epoch {epoch}: {e}")

    return (task_id, trained, len(existing_epochs))


def initialize_pool(
    split_file: str,
    output_dir: str,
    checkpoint_epochs: list,
    checkpoint_seeds: list,
    batch_size: int = 256,
    lr: float = 0.001,
    num_workers: int = 4
):
    """Initialize checkpoint pool from training tasks.

    Args:
        split_file: Path to task_split.yaml
        output_dir: Directory to save checkpoint pool
        checkpoint_epochs: List of epochs to save checkpoints
        checkpoint_seeds: List of random seeds for diversity
        batch_size: Training batch size
        lr: Learning rate
        num_workers: Number of parallel workers (default: 4)
    """
    # Load task split
    training_tasks, test_tasks = load_task_split(split_file)

    print("=" * 70)
    print("CHECKPOINT POOL INITIALIZATION")
    print("=" * 70)
    print()
    print(f"Training tasks: {len(training_tasks)}")
    print(f"Test tasks (held out): {len(test_tasks)}")
    print(f"Checkpoint epochs: {checkpoint_epochs}")
    print(f"Checkpoint seeds: {checkpoint_seeds}")
    print(f"Total checkpoints: {len(training_tasks) * len(checkpoint_epochs) * len(checkpoint_seeds)}")
    print()

    # Create checkpoint pool
    pool = CheckpointPool(output_dir)

    # Create individual checkpoints directory
    checkpoints_dir = pool.checkpoint_dir / "individual_checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    # Load existing checkpoints from disk if they exist
    existing_checkpoint_files = list(checkpoints_dir.glob("*.pt"))
    print(f"Found {len(existing_checkpoint_files)} existing checkpoint files")

    # Parse existing checkpoint metadata and add to pool
    for ckpt_path in existing_checkpoint_files:
        # Parse filename: {task_id}_epoch{epoch}_{source}.pt
        filename = ckpt_path.stem  # Remove .pt extension
        parts = filename.split('_epoch')
        if len(parts) == 2:
            task_id = parts[0]
            epoch_and_source = parts[1].split('_')
            epoch = int(epoch_and_source[0])
            source = '_'.join(epoch_and_source[1:]) if len(epoch_and_source) > 1 else 'pn_naive'

            # Parse task_id to extract metadata
            # Format: {dataset}_c{c_value}_prior{prior}_seed{seed}
            task_parts = task_id.rsplit('_seed', 1)
            if len(task_parts) == 2:
                base_task = task_parts[0]
                seed = int(task_parts[1])

                # Parse c_value and prior
                c_prior_parts = base_task.split('_c')[1].split('_prior')
                c_value = float(c_prior_parts[0])
                prior = float(c_prior_parts[1])
                dataset = base_task.split('_c')[0]

                # Load model state
                model_state = torch.load(ckpt_path, map_location='cpu')

                # Add to pool
                checkpoint = {
                    'task_id': task_id,
                    'epoch': epoch,
                    'seed': seed,
                    'model_state': model_state,
                    'dataset': dataset,
                    'c_value': c_value,
                    'prior': prior,
                    'source': source,
                    'created_at': 0
                }
                pool.add_checkpoint(checkpoint)

    print(f"Loaded {len(pool.checkpoints)} existing checkpoints into pool")
    print()

    # Device selection handled automatically by BaseTrainer (MPS → CUDA → CPU)
    print(f"Using {num_workers} parallel workers for training")
    print()

    # Build list of all tasks to process
    total_tasks = len(training_tasks) * len(checkpoint_seeds)
    task_args = []
    task_idx = 0

    for task_config in training_tasks:
        dataset = task_config['dataset']
        c_value = task_config['c_value']
        prior = task_config['prior']

        for seed in checkpoint_seeds:
            task_idx += 1
            task_args.append((
                task_idx, total_tasks, dataset, c_value, prior, seed,
                checkpoint_epochs, batch_size, lr, checkpoints_dir
            ))

    # Shuffle task order to avoid bias (mix datasets, configs, seeds)
    random.shuffle(task_args)
    print(f"Shuffled {len(task_args)} tasks for balanced processing")

    # Train tasks in parallel
    print(f"Processing {total_tasks} tasks in parallel...")
    print()

    trained_count = 0
    skipped_count = 0

    with Pool(num_workers) as pool_workers:
        results = pool_workers.map(train_task_worker, task_args)

    # Process results
    for task_id, trained, skipped in results:
        trained_count += trained
        skipped_count += skipped
        print(f"✓ {task_id}: trained {trained}, skipped {skipped}")

    # Reload all checkpoint files into pool for final save
    print()
    print("Reloading all checkpoints into pool...")
    pool.checkpoints = []  # Clear existing
    existing_checkpoint_files = list(checkpoints_dir.glob("*.pt"))
    for ckpt_path in existing_checkpoint_files:
        filename = ckpt_path.stem
        parts = filename.split('_epoch')
        if len(parts) == 2:
            task_id = parts[0]
            epoch_and_source = parts[1].split('_')
            epoch = int(epoch_and_source[0])
            source = '_'.join(epoch_and_source[1:]) if len(epoch_and_source) > 1 else 'pn_naive'

            task_parts = task_id.rsplit('_seed', 1)
            if len(task_parts) == 2:
                base_task = task_parts[0]
                seed = int(task_parts[1])

                c_prior_parts = base_task.split('_c')[1].split('_prior')
                c_value = float(c_prior_parts[0])
                prior = float(c_prior_parts[1])
                dataset = base_task.split('_c')[0]

                model_state = torch.load(ckpt_path, map_location='cpu')

                checkpoint = {
                    'task_id': task_id,
                    'epoch': epoch,
                    'seed': seed,
                    'model_state': model_state,
                    'dataset': dataset,
                    'c_value': c_value,
                    'prior': prior,
                    'source': source,
                    'created_at': 0
                }
                pool.add_checkpoint(checkpoint)

    # Print statistics
    print()
    print("=" * 70)
    print("CHECKPOINT SUMMARY")
    print("=" * 70)
    print(f"Checkpoints trained: {trained_count}")
    print(f"Checkpoints skipped (already exist): {skipped_count}")
    print(f"Total checkpoints in pool: {len(pool.checkpoints)}")
    print()

    pool.print_statistics()

    # Save pool metadata to disk
    print()
    pool.save_to_disk()
    print()

    print("=" * 70)
    print("INITIALIZATION COMPLETE")
    print("=" * 70)
    print(f"Total checkpoints: {len(pool.checkpoints)}")
    print(f"New checkpoints trained: {trained_count}")
    print(f"Existing checkpoints reused: {skipped_count}")
    print(f"Saved to: {output_dir}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Initialize checkpoint pool from PN naive training"
    )
    parser.add_argument(
        '--split',
        type=str,
        default='task_split.yaml',
        help='Path to task split YAML file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./meta_checkpoints',
        help='Output directory for checkpoint pool'
    )
    parser.add_argument(
        '--epochs',
        type=str,
        default='0,1,5,10,15,20,25,30,35,40',
        help='Comma-separated list of epochs to save checkpoints'
    )
    parser.add_argument(
        '--seeds',
        type=str,
        default='42,123,456,789',
        help='Comma-separated list of random seeds'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Training batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )

    args = parser.parse_args()

    # Parse epochs and seeds
    checkpoint_epochs = [int(e) for e in args.epochs.split(',')]
    checkpoint_seeds = [int(s) for s in args.seeds.split(',')]

    # Initialize pool
    initialize_pool(
        split_file=args.split,
        output_dir=args.output,
        checkpoint_epochs=checkpoint_epochs,
        checkpoint_seeds=checkpoint_seeds,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.workers
    )


if __name__ == '__main__':
    main()
