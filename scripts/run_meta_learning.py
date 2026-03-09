"""Run K=3 inner loop meta-learning for MonotonicBasisLoss.

This script implements K=3 inner loop vmap-parallelized meta-learning:
1. Loads checkpoint pool (~1,800 pre-trained model checkpoints)
2. Samples meta-batches and groups by architecture
3. Runs K=3 gradient steps with learned loss for each checkpoint
4. Measures improvement on validation data (with second-order gradients)
5. Updates loss parameters to maximize improvement across all checkpoints

Usage:
    # K=3 inner loop (default):
    python scripts/run_meta_learning.py \\
        --config config/methods/monotonic_basis_meta.yaml \\
        --checkpoint-dir ./meta_checkpoints

    # Original frozen-model approach:
    python scripts/run_meta_learning.py \\
        --config config/methods/monotonic_basis_meta.yaml \\
        --checkpoint-dir ./meta_checkpoints \\
        --no-k3

    # With evaluation on test tasks:
    python scripts/run_meta_learning.py \\
        --config config/methods/monotonic_basis_meta.yaml \\
        --checkpoint-dir ./meta_checkpoints \\
        --evaluate \\
        --split task_split.yaml
"""

import argparse
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml

from meta_learning.checkpoint_pool import CheckpointPool, load_task_split
from meta_learning.meta_trainer import MetaTrainer
from meta_learning.vmap_utils import get_device, check_vmap_compatibility


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration YAML

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    # Extract monotonic_basis_meta config
    if 'monotonic_basis_meta' in config_data:
        config = config_data['monotonic_basis_meta']
    else:
        config = config_data

    return config


def run_meta_learning(
    config_path: str,
    checkpoint_dir: str,
    device: str = 'auto',
    resume_from: str = None,
    use_k3: bool = True
):
    """Run meta-learning pipeline with K=3 inner loop.

    Args:
        config_path: Path to configuration YAML
        checkpoint_dir: Directory containing checkpoint pool
        device: Device string ('auto', 'mps', 'cuda', or 'cpu')
        resume_from: Optional path to loss checkpoint to resume from
        use_k3: If True, use K=3 inner loop; if False, use frozen-model approach
    """
    print("=" * 70)
    if use_k3:
        print("K=3 INNER LOOP META-LEARNING FOR MONOTONIC BASIS LOSS")
    else:
        print("FROZEN-MODEL META-LEARNING FOR MONOTONIC BASIS LOSS")
    print("=" * 70)
    print()

    # Load configuration
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    print()

    # Setup device (MPS → CUDA → CPU priority)
    if device == 'auto' or config.get('device') == 'auto':
        device = get_device()
    else:
        device = torch.device(device)

    config['device'] = device
    print(f"Using device: {device}")

    # Check vmap compatibility if enabled
    if config.get('use_vmap', True) and use_k3:
        vmap_ok = check_vmap_compatibility(device)
        if vmap_ok:
            print("✓ Vmap is compatible with this device")
        else:
            print("✗ Vmap not compatible, will use sequential processing")
            config['use_vmap'] = False
    print()

    # Load checkpoint pool
    print(f"Loading checkpoint pool from: {checkpoint_dir}")
    pool = CheckpointPool(checkpoint_dir)
    pool.load_from_disk()
    print()

    # Print pool statistics
    pool.print_statistics()
    print()

    # Create meta-trainer
    print("Initializing meta-trainer...")
    trainer = MetaTrainer(config, pool, device=device)
    print()

    # Resume from checkpoint if provided
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        trainer.load_checkpoint(resume_from)
        print()

    # Print configuration summary
    print("=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print(f"  Meta-learning rate: {config.get('meta_lr', 1e-4)}")
    if use_k3:
        print(f"  Inner loop LR: {config.get('inner_lr', 1e-3)}")
        print(f"  K inner steps: {config.get('K_inner_steps', 3)}")
    print(f"  Meta-batch size: {config.get('meta_batch_size', 8)}")
    print(f"  Meta-iterations: {config.get('meta_iterations', 1000)}")
    print(f"  Use vmap: {config.get('use_vmap', True)}")
    print(f"  L1 weight (baseline): {config.get('l1_weight', 1e-4)}")
    print(f"  L2 weight (Fourier): {config.get('l2_weight', 1e-3)}")
    print(f"  Total loss parameters: {sum(p.numel() for p in trainer.learned_loss.parameters())}")
    print("=" * 70)
    print()

    # Run meta-training
    num_iterations = config.get('meta_iterations', 1000)
    print(f"Starting meta-training for {num_iterations} iterations...")
    print()

    start_time = time.time()

    if use_k3:
        # K=3 inner loop meta-training
        for iteration in range(trainer.iteration, num_iterations):
            iter_start = time.time()

            # Sample meta-batch
            meta_batch = pool.sample_meta_batch(config.get('meta_batch_size', 8))

            # Meta-training step with K=3
            metrics = trainer.meta_train_step_k3(meta_batch)

            iter_time = time.time() - iter_start
            trainer.iteration = iteration + 1

            # Logging
            if (iteration + 1) % config.get('log_freq', 10) == 0:
                elapsed = time.time() - start_time
                iter_since_start = iteration + 1 - (trainer.iteration - iteration - 1)
                avg_time = elapsed / max(iter_since_start, 1)

                print(f"Iteration {iteration + 1}/{num_iterations}")
                print(f"  Avg improvement: {metrics['avg_improvement']:+.6f}")
                print(f"  Num checkpoints: {metrics['num_checkpoints']}")
                print(f"  Reg loss: {metrics['reg_loss']:.6f}")
                print(f"  Time: {iter_time:.2f}s (avg: {avg_time:.2f}s/iter)")
                print()

            # Save checkpoint
            if (iteration + 1) % config.get('save_freq', 50) == 0:
                output_dir = Path(config.get('loss_checkpoint_dir', './learned_losses'))
                output_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = output_dir / f"loss_iter{iteration + 1:04d}.pth"
                torch.save({
                    'iteration': iteration + 1,
                    'state_dict': trainer.learned_loss.state_dict(),
                    'optimizer_state': trainer.optimizer_loss.state_dict(),
                    'config': config,
                    'metrics': metrics
                }, checkpoint_path)
                print(f"✓ Saved checkpoint: {checkpoint_path}")
                print()

        # Save final checkpoint
        output_dir = Path(config.get('loss_checkpoint_dir', './learned_losses'))
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / "loss_final.pth"
        torch.save({
            'iteration': trainer.iteration,
            'state_dict': trainer.learned_loss.state_dict(),
            'optimizer_state': trainer.optimizer_loss.state_dict(),
            'config': config
        }, final_path)
        print(f"✓ Saved final checkpoint: {final_path}")
        print()

    else:
        # Original frozen-model approach
        trainer.train(num_iterations)

    total_time = time.time() - start_time

    print()
    print("=" * 70)
    print("META-LEARNING COMPLETE")
    print("=" * 70)
    print(f"  Total iterations: {trainer.iteration}")
    print(f"  Total time: {total_time / 3600:.2f} hours")
    print(f"  Avg time/iter: {total_time / trainer.iteration:.2f}s")
    print(f"  Checkpoints saved to: {config.get('loss_checkpoint_dir', './learned_losses')}")
    print("=" * 70)
    print()


def evaluate_on_test_tasks(
    learned_loss_path: str,
    split_file: str,
    config_path: str,
    device: str = 'auto'
):
    """Evaluate learned loss on held-out test tasks.

    Args:
        learned_loss_path: Path to learned loss checkpoint
        split_file: Path to task_split.yaml
        config_path: Path to configuration YAML
        device: Device string ('auto', 'mps', 'cuda', or 'cpu')
    """
    print("=" * 70)
    print("EVALUATING ON HELD-OUT TEST TASKS")
    print("=" * 70)
    print()

    # Load configuration
    config = load_config(config_path)

    # Setup device (MPS → CUDA → CPU priority)
    if device == 'auto':
        device = get_device()
    else:
        device = torch.device(device)

    # Load task split
    training_tasks, test_tasks = load_task_split(split_file)
    print(f"Training tasks: {len(training_tasks)}")
    print(f"Test tasks: {len(test_tasks)}")
    print()

    # Load learned loss
    from loss.loss_monotonic_basis import MonotonicBasisLoss

    print(f"Loading learned loss from: {learned_loss_path}")
    checkpoint = torch.load(learned_loss_path, map_location=device)

    learned_loss = MonotonicBasisLoss(
        num_repetitions=config.get('num_repetitions', 3),
        num_fourier=config.get('num_fourier', 16),
        use_prior=config.get('use_prior', True),
        l1_weight=config.get('l1_weight', 1e-4),
        l2_weight=config.get('l2_weight', 1e-3),
        oracle_mode=config.get('oracle_mode', False),
        init_scale=config.get('init_scale', 0.01)
    ).to(device)

    learned_loss.load_state_dict(checkpoint['state_dict'])
    learned_loss.eval()
    print()

    # Print learned loss statistics
    print("Learned Loss Statistics:")
    print(f"  Iteration: {checkpoint.get('iteration', 'unknown')}")
    print(f"  Meta-loss: {checkpoint.get('losses', {}).get('meta_loss', 'unknown')}")
    print()

    # Evaluate on test tasks
    print("Note: Full evaluation requires training models with learned loss")
    print("      and comparing to baselines. This is a placeholder for now.")
    print()

    # TODO: Implement full evaluation pipeline
    # For each test task:
    #   1. Train model with learned loss
    #   2. Train model with random init loss
    #   3. Train model with baseline (PN naive)
    #   4. Evaluate all models on test set
    #   5. Compute F1 scores and compare

    print("Test tasks to evaluate:")
    for i, task in enumerate(test_tasks, 1):
        print(f"  {i}. {task['dataset']:15s} c={task['c_value']:.1f} prior={task['prior']:.1f} "
              f"({task['data_type']})")

    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Run meta-learning for MonotonicBasisLoss"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/methods/monotonic_basis_meta.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='./meta_checkpoints',
        help='Directory containing checkpoint pool'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'mps', 'cuda', 'cpu'],
        help='Device to use for training (auto = MPS → CUDA → CPU)'
    )
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='Path to loss checkpoint to resume from'
    )
    parser.add_argument(
        '--k3',
        dest='use_k3',
        action='store_true',
        default=True,
        help='Use K=3 inner loop meta-training (default)'
    )
    parser.add_argument(
        '--no-k3',
        dest='use_k3',
        action='store_false',
        help='Use original frozen-model meta-training'
    )
    parser.add_argument(
        '--K',
        type=int,
        default=None,
        help='Number of inner loop steps (overrides config, default: 3)'
    )
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate on held-out test tasks after training'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='task_split.yaml',
        help='Path to task split YAML file (required for evaluation)'
    )
    parser.add_argument(
        '--loss-checkpoint',
        type=str,
        default=None,
        help='Path to learned loss checkpoint (for evaluation only mode)'
    )

    args = parser.parse_args()

    # Override K if specified
    if args.K is not None:
        # Load config to modify it
        config = load_config(args.config)
        config['K_inner_steps'] = args.K
        # Save modified config temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({'monotonic_basis_meta': config}, f)
            temp_config_path = f.name
        config_path = temp_config_path
    else:
        config_path = args.config

    # Run meta-learning
    if not args.loss_checkpoint:
        run_meta_learning(
            config_path=config_path,
            checkpoint_dir=args.checkpoint_dir,
            device=args.device,
            resume_from=args.resume_from,
            use_k3=args.use_k3
        )

        # Clean up temp config if created
        if args.K is not None:
            import os
            os.unlink(temp_config_path)

    # Evaluate on test tasks
    if args.evaluate:
        # Use provided loss checkpoint or look for final checkpoint
        if args.loss_checkpoint:
            loss_path = args.loss_checkpoint
        else:
            # Load config to get loss checkpoint dir
            config = load_config(args.config)
            loss_dir = Path(config.get('loss_checkpoint_dir', './learned_losses'))
            loss_path = str(loss_dir / 'loss_iter_final.pt')

        evaluate_on_test_tasks(
            learned_loss_path=loss_path,
            split_file=args.split,
            config_path=args.config,
            device=args.device
        )


if __name__ == '__main__':
    main()
