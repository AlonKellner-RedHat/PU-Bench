#!/usr/bin/env python3
"""Gradient Matching (Cosine Similarity) with VPU-Initialized SimpleNeuralPULoss.

Meta-objective: Maximize cosine similarity between learned PU loss and Oracle BCE gradients.
Equivalent to minimizing: 1 - cos_sim(∇L_PU, ∇L_BCE)

Key innovations:
1. **VPU-equivalent initialization**: SimpleNeuralPULoss initialized to EXACTLY match VPU loss
2. **Cosine similarity objective**: Direction matching (unlike MSE which also matches magnitude)
3. Full dataset (batch_size=1000) for more stable gradient estimation
4. SimpleNeuralPULoss with 40 features and learned W1/W2 weights (5,376 parameters)
5. Zero regularization (L1=0.0, L0.5=0.0) to allow free optimization
6. Checkpoint pool with 256 checkpoints advancing through training
7. Learning rate scheduling (cosine annealing with warmup)
8. Early stopping based on validation BCE
9. Extended training (2000 iterations)

Hypothesis: Cosine similarity focuses on gradient direction, potentially better than MSE for meta-learning.
Previous result with MSE: 0.309 BCE (3.3% worse than VPU 0.299).
"""

import torch
import yaml
from pathlib import Path
import numpy as np
import time
from tqdm import tqdm

from models.simple_mlp import SimpleMLP
from loss.simple_neural_pu_loss import SimpleNeuralPULoss
from tasks.gaussian_task import GaussianBlobTask
from tasks.gradient_matching_pool import GradientMatchingCheckpointPool
from gradient_matching_utils import (
    compute_gradient_mse,
    advance_checkpoint_one_step,
    create_model_from_checkpoint,
    sample_batch_deterministic,
)
from validation_utils import train_from_scratch_validation
from initialize_vpu_equivalent import initialize_to_vpu


def set_global_seed(seed: int = 42):
    """Set global random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Enable deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For PyTorch >= 1.8, set deterministic mode
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass  # Older PyTorch versions don't have this


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # Unwrap if nested
    if len(config) == 1 and isinstance(list(config.values())[0], dict):
        config = list(config.values())[0]
    return config


def get_device(config: dict) -> str:
    """Determine device from config."""
    device_config = config.get('device', 'auto')
    if device_config == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    return device_config


def get_lr_scheduler(optimizer, config: dict):
    """Create learning rate scheduler based on config.

    Args:
        optimizer: PyTorch optimizer
        config: Configuration dict

    Returns:
        scheduler or None if disabled
    """
    if not config.get('use_lr_scheduler', False):
        return None

    scheduler_type = config.get('scheduler_type', 'cosine_annealing')

    if scheduler_type == 'cosine_annealing':
        # Cosine annealing from meta_lr to meta_lr * lr_min_factor
        total_iters = config['meta_iterations']
        warmup_iters = config.get('lr_warmup_iterations', 0)

        # After warmup, use cosine annealing for remaining iterations
        annealing_iters = total_iters - warmup_iters
        min_lr = config['meta_lr'] * config.get('lr_min_factor', 0.1)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=annealing_iters,
            eta_min=min_lr,
        )
        return scheduler

    elif scheduler_type == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=50,
            verbose=True,
        )
        return scheduler

    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")


def apply_lr_warmup(optimizer, current_iter: int, config: dict):
    """Apply linear warmup to learning rate.

    Args:
        optimizer: PyTorch optimizer
        current_iter: Current iteration number (0-indexed)
        config: Configuration dict
    """
    warmup_iters = config.get('lr_warmup_iterations', 0)

    if warmup_iters > 0 and current_iter < warmup_iters:
        # Linear warmup from 0 to meta_lr
        warmup_factor = (current_iter + 1) / warmup_iters
        lr = config['meta_lr'] * warmup_factor

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def get_hybrid_alpha(current_iter: int, config: dict) -> float:
    """Compute hybrid objective alpha (weight for gradient matching).

    Curriculum learning: gradually transition from pure gradient matching
    to hybrid gradient matching + validation BCE.

    Args:
        current_iter: Current iteration number (0-indexed)
        config: Configuration dict

    Returns:
        alpha: Weight for gradient matching (1-alpha for validation BCE)
    """
    if not config.get('use_hybrid_objective', False):
        return 1.0  # Pure gradient matching

    alpha_start = config.get('hybrid_alpha_start', 1.0)
    alpha_end = config.get('hybrid_alpha_end', 0.5)
    curriculum_start = config.get('hybrid_curriculum_start_iter', 200)
    curriculum_end = config.get('hybrid_curriculum_end_iter', 1000)

    if current_iter < curriculum_start:
        return alpha_start
    elif current_iter >= curriculum_end:
        return alpha_end
    else:
        # Linear interpolation
        progress = (current_iter - curriculum_start) / (curriculum_end - curriculum_start)
        return alpha_start + progress * (alpha_end - alpha_start)


def compute_validation_bce_loss(
    learned_loss: NeuralPULoss,
    val_tasks: list,
    device: str,
    num_samples: int = 500,
) -> torch.Tensor:
    """Compute validation BCE loss for meta-optimization.

    Trains a fresh model with learned loss, evaluates with BCE on held-out data.
    This provides end-to-end gradient signal for meta-learning.

    Args:
        learned_loss: Current learned loss
        val_tasks: List of validation tasks
        device: Device
        num_samples: Number of training samples per task

    Returns:
        avg_bce: Average BCE loss across tasks (WITH gradients)
    """
    bce_fn = torch.nn.BCEWithLogitsLoss()
    total_bce = torch.tensor(0.0, device=device)

    for task in val_tasks:
        # Create fresh model
        model = SimpleMLP(2, [32, 32]).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        # Get data
        dataloaders = task.get_dataloaders(
            batch_size=64,
            num_train=num_samples,
            num_val=200,
            num_test=200,
        )

        # Train for a few steps with learned loss
        model.train()
        num_train_steps = 10  # Quick training

        for step in range(num_train_steps):
            for batch in dataloaders['train']:
                if step >= num_train_steps:
                    break

                x, y_true, y_pu = batch[0].to(device), batch[1].to(device), batch[2].to(device)

                optimizer.zero_grad()
                outputs = model(x).squeeze(-1)
                loss = learned_loss(outputs, y_pu, mode='pu')
                loss.backward()
                optimizer.step()

                step += 1
                if step >= num_train_steps:
                    break

        # Evaluate with BCE on validation set
        model.eval()
        val_bce = torch.tensor(0.0, device=device)
        num_batches = 0

        for batch in dataloaders['val']:
            x, y_true = batch[0].to(device), batch[1].to(device)
            outputs = model(x).squeeze(-1)
            val_bce = val_bce + bce_fn(outputs, y_true)
            num_batches += 1

        val_bce = val_bce / max(num_batches, 1)
        total_bce = total_bce + val_bce

    avg_bce = total_bce / len(val_tasks)
    return avg_bce


def main():
    # Set global seed for deterministic results
    set_global_seed(42)

    # Load configuration
    config_path = 'config/gradient_matching_meta_improved.yaml'
    config = load_config(config_path)

    # Force CPU if deterministic mode is enabled and MPS is default
    if config.get('force_deterministic', False):
        device = 'cpu'
        print("⚠️  DETERMINISTIC MODE: Forcing CPU (MPS has non-deterministic behavior)")
    else:
        device = get_device(config)

    print("=" * 70)
    print("IMPROVED GRADIENT MATCHING META-LEARNING")
    print("=" * 70)
    print("Configuration:")
    print(f"  - Loss: SimpleNeuralPULoss (40 input features, direct aggregation)")
    print(f"  - Hidden dim: {config['loss_hidden_dim']}")
    print(f"  - L1 lambda: {config.get('loss_l1_lambda', 0.0)}")
    print(f"  - L0.5 lambda: {config.get('loss_l05_lambda', 0.0)}")
    print(f"  - Max weight norm: {config.get('loss_max_weight_norm', 10.0)}")
    print(f"  - Num checkpoints: {config['pool_size']} (ALL processed each iteration)")
    print(f"  - Refresh rate: {config['num_to_refresh']} checkpoints per iteration ({config['num_to_refresh']/config['pool_size']*100:.1f}%)")
    print(f"  - Inner steps per meta-update: {config.get('inner_steps_per_meta_update', 1)}")
    print(f"  - Batch size: {config['batch_size']} (for each checkpoint's training step)")
    print(f"  - Meta iterations: {config['meta_iterations']}")
    print(f"  - Gradient accumulation: {config['meta_grad_accumulation_steps']} steps")
    print(f"  - Meta LR: {config['meta_lr']} (betas={config['meta_betas']})")
    print(f"  - LR scheduler: {config.get('scheduler_type', 'none')}")
    if config.get('use_lr_scheduler'):
        print(f"  - LR warmup: {config.get('lr_warmup_iterations', 0)} iterations")
    print(f"  - Hybrid objective: {'enabled' if config.get('use_hybrid_objective') else 'disabled'}")
    if config.get('use_hybrid_objective'):
        print(f"    Alpha: {config['hybrid_alpha_start']:.1f} → {config['hybrid_alpha_end']:.1f} (iter {config['hybrid_curriculum_start_iter']}-{config['hybrid_curriculum_end_iter']})")
    print(f"  - Early stopping: {'enabled' if config.get('use_early_stopping') else 'disabled'}")
    if config.get('use_early_stopping'):
        print(f"    Patience: {config['early_stopping_patience']} checks, min delta: {config['early_stopping_min_delta']}")
    print(f"  - Device: {device}")
    print("=" * 70)
    print()

    # Initialize learned loss (SimpleNeuralPULoss with 40 features)
    learned_loss = SimpleNeuralPULoss(
        hidden_dim=config['loss_hidden_dim'],
        eps=1e-7,
        l1_lambda=config.get('loss_l1_lambda', 0.0),
        l05_lambda=config.get('loss_l05_lambda', 0.0),
        init_mode=config['loss_init_mode'],
        init_scale=1.0,
        max_weight_norm=config.get('loss_max_weight_norm', 10.0),
    ).to(device)

    # Initialize to VPU-equivalent
    print("Initializing learned loss to VPU-equivalent...")
    learned_loss = initialize_to_vpu(learned_loss)
    print("✓ Learned loss initialized to exactly match VPU")
    print()

    print("Initial loss:")
    print(learned_loss)
    print()

    # Meta-optimizer with improved settings
    meta_optimizer = torch.optim.AdamW(
        learned_loss.parameters(),
        lr=config['meta_lr'],
        betas=tuple(config['meta_betas']),
        weight_decay=config['meta_weight_decay'],
    )

    # Learning rate scheduler
    lr_scheduler = get_lr_scheduler(meta_optimizer, config)
    if lr_scheduler:
        print(f"Using LR scheduler: {config.get('scheduler_type')}")
        print()

    # Initialize checkpoint pool
    pool = GradientMatchingCheckpointPool(
        config=config,
        pool_size=config['pool_size'],
        input_dim=config.get('num_dimensions', 2),
        hidden_dims=config.get('model_hidden_dims', [32, 32]),
        inner_lr=config['inner_lr'],
        inner_momentum=config['inner_momentum'],
    )
    pool.initialize_pool(device)

    print(f"Checkpoint pool initialized:")
    print(pool)
    print()

    # Create validation tasks (fixed for reproducibility)
    val_tasks = []
    for i in range(config['num_val_tasks']):
        task = GaussianBlobTask(
            num_dimensions=2,
            mean_separation=2.5,
            std=1.0,
            prior=0.5,
            labeling_freq=0.3,
            num_samples=1000,
            seed=9000 + i,
            mode='pu',
            negative_labeling_freq=0.3,
        )
        val_tasks.append(task)

    cached_baselines = None

    # Initial validation (iteration 0) - calibrate baselines
    print("Running iteration 0 baseline calibration...")
    val_results, cached_baselines = train_from_scratch_validation(
        val_tasks, learned_loss, config, device, cached_baselines
    )

    # Now evaluate initialized learned loss using the calibrated step count
    print("\nEvaluating initialized learned loss (before meta-learning)...")
    val_results, cached_baselines = train_from_scratch_validation(
        val_tasks, learned_loss, config, device, cached_baselines
    )

    print()
    print("=" * 70)
    print("ITERATION 0 - BASELINE CALIBRATION & INITIAL LEARNED LOSS")
    print("=" * 70)
    print(f"  Training steps for convergence: {cached_baselines['converged_steps']}")
    print()
    print("  Baseline BCE scores (converged at step end):")
    print(f"    Oracle BCE:   {val_results['oracle']:.6f} (best: {val_results['oracle_best']:.6f})")
    print(f"    Naive BCE:    {val_results['naive']:.6f} (best: {val_results['naive_best']:.6f})")
    print(f"    PUDRa-naive:  {val_results['pudra_naive']:.6f} (best: {val_results['pudra_naive_best']:.6f})")
    print(f"    VPU-NoMixUp:  {val_results['vpu_nomixup']:.6f} (best: {val_results['vpu_nomixup_best']:.6f})")
    print()
    print("  Initialized learned loss (BEFORE meta-learning):")
    print(f"    Final:  {val_results['learned']:.6f}")
    print(f"    Best:   {val_results['learned_best']:.6f}")
    print("=" * 70)
    print()

    # Setup checkpoint directory
    output_dir = Path("gradient_matching_output_improved")
    output_dir.mkdir(exist_ok=True)

    # Track best validation performance
    best_val_bce = val_results['learned_best']
    best_iteration = 0
    patience_counter = 0

    # Early stopping
    early_stopping_patience = config.get('early_stopping_patience', 200)
    early_stopping_min_delta = config.get('early_stopping_min_delta', 0.001)

    # Gradient accumulation
    grad_accumulation_steps = config.get('meta_grad_accumulation_steps', 1)

    # Training loop
    print("Starting improved gradient matching meta-learning...")
    print()
    start_time = time.time()

    # Initialize optimizer (zero gradients before first iteration)
    meta_optimizer.zero_grad()

    for meta_iter in tqdm(range(config['meta_iterations']), desc="Meta-training"):
        # Zero gradients at start of accumulation window
        if meta_iter % grad_accumulation_steps == 0:
            meta_optimizer.zero_grad()

        # Apply learning rate warmup if needed
        if config.get('use_lr_scheduler') and config.get('lr_warmup_iterations', 0) > 0:
            apply_lr_warmup(meta_optimizer, meta_iter, config)

        # Get curriculum alpha for hybrid objective
        alpha = get_hybrid_alpha(meta_iter, config)

        # Use ALL checkpoints every iteration (no sampling)
        checkpoints = pool.checkpoints

        total_grad_matching_loss = torch.tensor(0.0, device=device)
        all_diagnostics = []
        updated_checkpoints = []

        # Number of inner steps per meta-update
        inner_steps_per_meta = config.get('inner_steps_per_meta_update', 1)

        # Process all checkpoints
        for checkpoint in checkpoints:
            # Take multiple inner steps for this checkpoint
            current_checkpoint = checkpoint
            checkpoint_grad_loss = torch.tensor(0.0, device=device)
            checkpoint_diagnostics = []

            for inner_step in range(inner_steps_per_meta):
                # Create model and load checkpoint state
                model = create_model_from_checkpoint(current_checkpoint, device)

                # Get parameters as dict (requires_grad=True for meta-learning)
                params = {
                    name: param.clone().detach().requires_grad_(True)
                    for name, param in model.named_parameters()
                }

                # Get batch from checkpoint's task (full dataset)
                task = GaussianBlobTask(**current_checkpoint['task_config'])
                x, y_true, y_pu = sample_batch_deterministic(
                    task,
                    current_checkpoint['step_count'],
                    batch_size=config['batch_size'],
                    device=device,
                )

                # Compute gradient matching loss
                grad_match_loss, diagnostics = compute_gradient_mse(
                    model, params, x, y_pu, y_true,
                    learned_loss, device
                )

                checkpoint_grad_loss = checkpoint_grad_loss + grad_match_loss
                checkpoint_diagnostics.append(diagnostics)

                # Advance checkpoint one step using its assigned objective
                current_checkpoint = advance_checkpoint_one_step(
                    current_checkpoint, learned_loss, device,
                    batch_size=config['batch_size'],
                )

            # Average gradient matching loss over inner steps
            avg_checkpoint_grad_loss = checkpoint_grad_loss / inner_steps_per_meta
            total_grad_matching_loss = total_grad_matching_loss + avg_checkpoint_grad_loss

            # Average diagnostics over inner steps
            avg_checkpoint_diagnostics = {
                key: np.mean([d[key] for d in checkpoint_diagnostics])
                for key in checkpoint_diagnostics[0].keys()
            }
            all_diagnostics.append(avg_checkpoint_diagnostics)

            # Save final checkpoint state after all inner steps
            current_checkpoint['last_updated_iteration'] = meta_iter
            updated_checkpoints.append(current_checkpoint)

        # Average gradient matching loss (this is the meta-objective)
        meta_loss = total_grad_matching_loss / len(checkpoints)
        oracle_bce_component = torch.tensor(0.0, device=device)

        # Gradient accumulation
        meta_loss = meta_loss / grad_accumulation_steps
        meta_loss.backward()

        # Meta-update every grad_accumulation_steps
        if (meta_iter + 1) % grad_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(learned_loss.parameters(), max_norm=1.0)
            meta_optimizer.step()
            # Note: zero_grad() is called at the start of next accumulation window

            # Step LR scheduler (after warmup)
            if lr_scheduler and config.get('scheduler_type') == 'cosine_annealing':
                if meta_iter >= config.get('lr_warmup_iterations', 0):
                    lr_scheduler.step()

        # Update pool (refresh fewer checkpoints for better progression)
        pool.update_pool(
            updated_checkpoints,
            num_to_refresh=config['num_to_refresh'],
            current_iteration=meta_iter,
            device=device,
        )

        # Compute average diagnostics
        avg_diagnostics = {
            key: np.mean([d[key] for d in all_diagnostics])
            for key in all_diagnostics[0].keys()
        }

        # Validation every N iterations
        if (meta_iter + 1) % config['val_freq'] == 0:
            # Use cached baselines (computed once at iteration 0)
            val_results, cached_baselines = train_from_scratch_validation(
                val_tasks, learned_loss, config, device, cached_baselines
            )

            elapsed = time.time() - start_time
            iters_per_min = (meta_iter + 1) / (elapsed / 60)

            # Pool statistics
            pool_stats = pool.get_statistics()

            # Loss parameter statistics (include all learnable parameters)
            weights = learned_loss.linear.weight.detach().cpu().numpy()
            bias = learned_loss.linear.bias.detach().cpu().numpy()
            w1 = learned_loss.W1.detach().cpu().numpy()
            w2 = learned_loss.W2.detach().cpu().numpy()
            all_params = np.concatenate([weights.flatten(), bias, w1, w2])
            near_zero = np.sum(np.abs(all_params) < 0.01)
            sparsity_pct = near_zero / len(all_params) * 100

            # Get current LR
            current_lr = meta_optimizer.param_groups[0]['lr']

            tqdm.write(f"\nIteration {meta_iter + 1}/{config['meta_iterations']}")
            tqdm.write(f"  Speed: {iters_per_min:.1f} iters/min, LR: {current_lr:.6f}")
            tqdm.write(f"  --- Meta-Objective (Cosine Similarity) ---")
            tqdm.write(f"  Cosine loss (1-cos): {avg_diagnostics['cosine_loss']:.6f}  ← META-OBJECTIVE")
            tqdm.write(f"  Cosine similarity:   {avg_diagnostics['cosine_similarity']:.4f}")
            tqdm.write(f"  Gradient MSE:        {avg_diagnostics['gradient_mse']:.6f}")
            tqdm.write(f"  PU grad norm:        {avg_diagnostics['pu_grad_norm']:.4f}")
            tqdm.write(f"  BCE grad norm:       {avg_diagnostics['bce_grad_norm']:.4f}")
            tqdm.write(f"  --- End-to-End Validation (Final / Best) ---")
            if 'learned' in val_results and val_results['learned'] is not None:
                tqdm.write(f"  Learned:      {val_results['learned']:.6f} / {val_results['learned_best']:.6f}")
            tqdm.write(f"  Oracle BCE:   {val_results['oracle']:.6f} / {val_results['oracle_best']:.6f}")
            tqdm.write(f"  Naive BCE:    {val_results['naive']:.6f} / {val_results['naive_best']:.6f}")
            tqdm.write(f"  PUDRa-naive:  {val_results['pudra_naive']:.6f} / {val_results['pudra_naive_best']:.6f}")
            tqdm.write(f"  VPU-NoMixUp:  {val_results['vpu_nomixup']:.6f} / {val_results['vpu_nomixup_best']:.6f}")

            # Objective distribution
            obj_counts = {}
            for c in pool.checkpoints:
                obj = c['objective']
                obj_counts[obj] = obj_counts.get(obj, 0) + 1

            tqdm.write(f"  --- Checkpoint Pool ---")
            tqdm.write(f"  Step range: [{pool_stats['min_steps']}, {pool_stats['max_steps']}]")
            tqdm.write(f"  Mean steps: {pool_stats['mean_steps']:.1f} ± {pool_stats['std_steps']:.1f}")
            tqdm.write(f"  Objectives: BCE={obj_counts.get('oracle_bce', 0)} PUDRa={obj_counts.get('pudra', 0)} VPU={obj_counts.get('vpu', 0)} Learned={obj_counts.get('learned', 0)}")
            tqdm.write(f"  --- Loss Parameters ---")
            tqdm.write(f"  Sparsity: {near_zero}/{len(all_params)} ({sparsity_pct:.1f}%)")
            tqdm.write(f"  Weight range: [{weights.min():.4f}, {weights.max():.4f}]")

            # Check for improvement (early stopping and best checkpoint)
            if 'learned' in val_results and val_results['learned'] is not None:
                current_bce = val_results['learned_best']  # Use best, not final

                # Check if this is a new best
                if current_bce < best_val_bce - early_stopping_min_delta:
                    improvement = best_val_bce - current_bce
                    best_val_bce = current_bce
                    best_iteration = meta_iter + 1
                    patience_counter = 0

                    checkpoint_data = {
                        'iteration': meta_iter + 1,
                        'loss_state_dict': learned_loss.state_dict(),
                        'optimizer_state_dict': meta_optimizer.state_dict(),
                        'val_bce': current_bce,
                        'val_results': val_results,
                        'pool_stats': pool_stats,
                        'diagnostics': avg_diagnostics,
                        'config': config,
                        'alpha': alpha,
                        'lr': current_lr,
                    }

                    best_path = output_dir / "best_checkpoint.pt"
                    torch.save(checkpoint_data, best_path)
                    tqdm.write(f"  ✓ NEW BEST! BCE: {current_bce:.6f} (improved by {improvement:.6f})")
                else:
                    patience_counter += 1
                    tqdm.write(f"  No improvement ({patience_counter}/{early_stopping_patience})")

            # Save periodic checkpoints every 100 iterations
            if (meta_iter + 1) % 100 == 0 and 'learned' in val_results and val_results['learned'] is not None:
                current_bce = val_results['learned_best']
                checkpoint_data = {
                    'iteration': meta_iter + 1,
                    'loss_state_dict': learned_loss.state_dict(),
                    'optimizer_state_dict': meta_optimizer.state_dict(),
                    'val_bce': current_bce,
                    'val_results': val_results,
                    'pool_stats': pool_stats,
                    'diagnostics': avg_diagnostics,
                    'config': config,
                    'alpha': alpha,
                    'lr': current_lr,
                }

                periodic_path = output_dir / f"checkpoint_iter_{meta_iter + 1}.pt"
                torch.save(checkpoint_data, periodic_path)

            tqdm.write("")

            # Early stopping check
            if config.get('use_early_stopping') and patience_counter >= early_stopping_patience:
                tqdm.write(f"\n{'='*70}")
                tqdm.write(f"EARLY STOPPING: No improvement for {early_stopping_patience} validation checks")
                tqdm.write(f"Best BCE: {best_val_bce:.6f} at iteration {best_iteration}")
                tqdm.write(f"{'='*70}\n")
                break

    # Final results
    elapsed_time = time.time() - start_time
    actual_iters = meta_iter + 1  # In case of early stopping

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Training time: {elapsed_time / 60:.1f} minutes")
    print(f"Iterations completed: {actual_iters}/{config['meta_iterations']}")
    print(f"Speed: {actual_iters / (elapsed_time / 60):.1f} iterations/min")
    print()

    # Final learned loss
    print("Final learned loss:")
    print(learned_loss)
    print()

    # Final pool statistics
    print("Final checkpoint pool:")
    print(pool)
    print()

    pool_stats = pool.get_statistics()
    print("Pool statistics:")
    print(f"  Step range: [{pool_stats['min_steps']}, {pool_stats['max_steps']}]")
    print(f"  Mean: {pool_stats['mean_steps']:.1f}")
    print(f"  Std: {pool_stats['std_steps']:.1f}")
    print(f"  Median: {pool_stats['median_steps']:.1f}")
    print()

    # Parameter statistics
    weights = learned_loss.linear.weight.detach().cpu().numpy()
    bias = learned_loss.linear.bias.detach().cpu().numpy()
    all_params = np.concatenate([weights.flatten(), bias])

    print("Loss parameter statistics:")
    print(f"  Sparsity: {np.sum(np.abs(all_params) < 0.01)}/{len(all_params)} ({np.sum(np.abs(all_params) < 0.01) / len(all_params) * 100:.1f}%)")
    print(f"  Mean absolute value: {np.abs(all_params).mean():.6f}")
    print(f"  Median absolute value: {np.median(np.abs(all_params)):.6f}")
    print(f"  Max absolute value: {np.abs(all_params).max():.6f}")
    print()

    # Performance summary
    pudra_baseline = val_results['pudra_naive']
    vpu_baseline = val_results['vpu_nomixup']

    print("Performance summary:")
    if 'learned' in val_results and val_results['learned'] is not None:
        final_learned = val_results['learned_best']  # Use best
        print(f"  Final (best):         {final_learned:.6f}")
        print(f"  PUDRa baseline:       {pudra_baseline:.6f}")
        print(f"  VPU baseline:         {vpu_baseline:.6f}")
        print()

        if final_learned < pudra_baseline:
            improvement = (pudra_baseline - final_learned) / pudra_baseline * 100
            print(f"  ✓ BEATS PUDRa by {improvement:.1f}%")
        elif final_learned < pudra_baseline + 0.02:
            print(f"  ~ MATCHES PUDRa (within 2%)")
        else:
            gap = (final_learned - pudra_baseline) / pudra_baseline * 100
            print(f"  Below PUDRa by {gap:.1f}%")

        if final_learned < vpu_baseline:
            improvement = (vpu_baseline - final_learned) / vpu_baseline * 100
            print(f"  ✓ BEATS VPU by {improvement:.1f}%")
        else:
            gap = (final_learned - vpu_baseline) / vpu_baseline * 100
            print(f"  Below VPU by {gap:.1f}%")
    else:
        print(f"  Learned loss not evaluated (run final validation separately)")
        print(f"  PUDRa baseline:       {pudra_baseline:.6f}")
        print(f"  VPU baseline:         {vpu_baseline:.6f}")
        print()

    # Best checkpoint info
    print()
    if best_val_bce < float('inf'):
        print("Best checkpoint during training:")
        print(f"  Iteration: {best_iteration}")
        print(f"  BCE: {best_val_bce:.6f}")

        if best_val_bce < pudra_baseline:
            improvement = (pudra_baseline - best_val_bce) / pudra_baseline * 100
            print(f"  ✓ BEAT PUDRa by {improvement:.1f}%")
        else:
            gap = (best_val_bce - pudra_baseline) / pudra_baseline * 100
            print(f"  Below PUDRa by {gap:.1f}%")

        if best_val_bce < vpu_baseline:
            improvement = (vpu_baseline - best_val_bce) / vpu_baseline * 100
            print(f"  ✓ BEAT VPU by {improvement:.1f}%")
        else:
            gap = (best_val_bce - vpu_baseline) / vpu_baseline * 100
            print(f"  Below VPU by {gap:.1f}%")
    else:
        print("No best checkpoint saved (learned loss not evaluated during training)")

    print("=" * 70)

    # Save final checkpoint pool
    pool.save_pool(output_dir / "final_checkpoint_pool.pt")
    torch.save(learned_loss.state_dict(), output_dir / "final_learned_loss.pt")

    print()
    print(f"✓ Checkpoint pool saved to {output_dir / 'final_checkpoint_pool.pt'}")
    print(f"✓ Learned loss saved to {output_dir / 'final_learned_loss.pt'}")


if __name__ == '__main__':
    main()
