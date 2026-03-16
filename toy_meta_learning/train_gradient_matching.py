#!/usr/bin/env python3
"""Meta-learning with Gradient Matching (Simplified, Oracle BCE progression).

Approach:
- Meta-objective: Minimize cosine + log(MSE) between PU loss gradients and oracle BCE gradients
- Inner loop: All checkpoints advance one step with oracle BCE (stable progression)
- Replacement: 2% of checkpoints replaced with fresh initialization each iteration

Key features:
- Dense supervision: Meta-gradient signal at EVERY training step
- Fixed checkpoint set: 256 checkpoints, ALL processed each iteration
- Minimal replacement: 98% persist (only ~5 replaced per iteration)
- Guaranteed maturation: Checkpoints reliably progress through training stages
"""

import torch
import yaml
from pathlib import Path
import numpy as np
import time
from tqdm import tqdm

from models.simple_mlp import SimpleMLP
from loss.neural_pu_loss import NeuralPULoss
from tasks.gaussian_task import GaussianBlobTask
from tasks.gradient_matching_pool import GradientMatchingCheckpointPool
from gradient_matching_utils import (
    compute_gradient_mse,
    advance_checkpoint_one_step,
    create_model_from_checkpoint,
    sample_batch_deterministic,
)
from validation_utils import train_from_scratch_validation


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


def main():
    # Load configuration
    config_path = 'config/gradient_matching_meta.yaml'
    config = load_config(config_path)
    device = get_device(config)

    print("=" * 70)
    print("GRADIENT MATCHING META-LEARNING (SIMPLIFIED)")
    print("=" * 70)
    print("Configuration:")
    print(f"  - Loss: NeuralPULoss (13 input features)")
    print(f"  - Hidden dim: {config['loss_hidden_dim']}")
    print(f"  - Num checkpoints: {config['pool_size']} (ALL processed each iteration)")
    print(f"  - Replacement: 8 random checkpoints per iteration (3.1% refresh rate)")
    print(f"  - Batch size: {config['batch_size']} (for each checkpoint's training step)")
    print(f"  - Meta iterations: {config['meta_iterations']}")
    print(f"  - Inner LR: {config['inner_lr']}")
    print(f"  - Meta LR: {config['meta_lr']}")
    print(f"  - Inner loop: Oracle BCE (stable progression)")
    print(f"  - Meta-objective: Cosine similarity + log(MSE) of gradients")
    print(f"  - Device: {device}")
    print("=" * 70)
    print()

    # Initialize learned loss
    learned_loss = NeuralPULoss(
        hidden_dim=config['loss_hidden_dim'],
        eps=1e-7,
        l05_lambda=config['loss_l05_lambda'],
        init_mode=config['loss_init_mode'],
        init_scale=1.0,
    ).to(device)

    print("Initial loss:")
    print(learned_loss)
    print()

    # Meta-optimizer
    meta_optimizer = torch.optim.AdamW(
        learned_loss.parameters(),
        lr=config['meta_lr'],
        weight_decay=config['meta_weight_decay'],
    )

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
    output_dir = Path("gradient_matching_output")
    output_dir.mkdir(exist_ok=True)

    # Track best validation performance
    best_val_bce = float('inf')
    best_iteration = -1

    # Training loop
    print("Starting gradient matching meta-learning...")
    print()
    start_time = time.time()

    for meta_iter in tqdm(range(config['meta_iterations']), desc="Meta-training"):
        # Use ALL checkpoints every iteration (no sampling)
        checkpoints = pool.checkpoints

        total_meta_loss = torch.tensor(0.0, device=device)
        all_diagnostics = []
        updated_checkpoints = []

        for checkpoint in checkpoints:
            # Create model and load checkpoint state
            model = create_model_from_checkpoint(checkpoint, device)

            # Get parameters as dict (requires_grad=True for meta-learning)
            params = {
                name: param.clone().detach().requires_grad_(True)
                for name, param in model.named_parameters()
            }

            # Get batch from checkpoint's task
            task = GaussianBlobTask(**checkpoint['task_config'])
            x, y_true, y_pu = sample_batch_deterministic(
                task,
                checkpoint['step_count'],
                batch_size=config['batch_size'],
                device=device,
            )

            # Compute gradient matching loss (cosine + log(MSE))
            # ALWAYS match learned loss gradients to oracle BCE gradients
            # (checkpoint's objective only affects its training, not meta-objective)
            meta_loss, diagnostics = compute_gradient_mse(
                model, params, x, y_pu, y_true,
                learned_loss, device
            )

            total_meta_loss = total_meta_loss + meta_loss
            all_diagnostics.append(diagnostics)

            # Advance checkpoint one step using its assigned objective
            updated_ckpt = advance_checkpoint_one_step(
                checkpoint, learned_loss, device,
                batch_size=config['batch_size'],
            )
            updated_ckpt['last_updated_iteration'] = meta_iter
            updated_checkpoints.append(updated_ckpt)

        # Meta-update: Minimize meta-loss (cosine + magnitude)
        avg_meta_loss = total_meta_loss / len(checkpoints)

        meta_optimizer.zero_grad()
        avg_meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(learned_loss.parameters(), max_norm=1.0)
        meta_optimizer.step()

        # Update pool (refresh 8 random checkpoints)
        pool.update_pool(
            updated_checkpoints,
            num_to_refresh=8,
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

            # Loss parameter statistics
            weights = learned_loss.linear.weight.detach().cpu().numpy()
            bias = learned_loss.linear.bias.detach().cpu().numpy()
            all_params = np.concatenate([weights.flatten(), bias])
            near_zero = np.sum(np.abs(all_params) < 0.01)
            sparsity_pct = near_zero / len(all_params) * 100

            tqdm.write(f"\nIteration {meta_iter + 1}/{config['meta_iterations']}")
            tqdm.write(f"  Speed: {iters_per_min:.1f} iters/min")
            tqdm.write(f"  --- Gradient Matching ---")
            tqdm.write(f"  Meta loss:           {avg_diagnostics['meta_loss']:.6f}")
            tqdm.write(f"  Cosine loss:         {avg_diagnostics['cosine_loss']:.6f}")
            tqdm.write(f"  Magnitude loss:      {avg_diagnostics['magnitude_loss']:.6f}")
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

            # Save best checkpoint (only if learned loss was evaluated)
            # Use the best score during training, not final score
            if 'learned' in val_results and val_results['learned'] is not None:
                current_bce = val_results['learned_best']  # Use best, not final
                if current_bce < best_val_bce:
                    best_val_bce = current_bce
                    best_iteration = meta_iter + 1

                    checkpoint_data = {
                        'iteration': meta_iter + 1,
                        'loss_state_dict': learned_loss.state_dict(),
                        'optimizer_state_dict': meta_optimizer.state_dict(),
                    'val_bce': current_bce,
                    'val_results': val_results,
                    'pool_stats': pool_stats,
                    'diagnostics': avg_diagnostics,
                    'config': config,
                }

                    best_path = output_dir / "best_checkpoint.pt"
                    torch.save(checkpoint_data, best_path)
                    tqdm.write(f"  ✓ NEW BEST! BCE: {current_bce:.6f} (saved to {best_path.name})")

            # Save periodic checkpoints every 100 iterations
            if (meta_iter + 1) % 100 == 0 and 'learned' in val_results and val_results['learned'] is not None:
                current_bce = val_results['learned_best']  # Use best
                checkpoint_data = {
                    'iteration': meta_iter + 1,
                    'loss_state_dict': learned_loss.state_dict(),
                    'optimizer_state_dict': meta_optimizer.state_dict(),
                    'val_bce': current_bce,
                    'val_results': val_results,
                    'pool_stats': pool_stats,
                    'diagnostics': avg_diagnostics,
                    'config': config,
                }

                periodic_path = output_dir / f"checkpoint_iter_{meta_iter + 1}.pt"
                torch.save(checkpoint_data, periodic_path)

            tqdm.write("")

    # Final results
    elapsed_time = time.time() - start_time
    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Training time: {elapsed_time / 60:.1f} minutes")
    print(f"Speed: {config['meta_iterations'] / (elapsed_time / 60):.1f} iterations/min")
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
        final_learned = val_results['learned']
        print(f"  Final (learned):      {final_learned:.6f}")
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
            print(f"  ✗ Below PUDRa by {gap:.1f}%")
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
    else:
        print("No best checkpoint saved (learned loss not evaluated during training)")

    print("=" * 70)

    # Save final checkpoint pool
    output_dir = Path("gradient_matching_output")
    output_dir.mkdir(exist_ok=True)

    pool.save_pool(output_dir / "final_checkpoint_pool.pt")
    torch.save(learned_loss.state_dict(), output_dir / "final_learned_loss.pt")

    print()
    print(f"✓ Checkpoint pool saved to {output_dir / 'final_checkpoint_pool.pt'}")
    print(f"✓ Learned loss saved to {output_dir / 'final_learned_loss.pt'}")


if __name__ == '__main__':
    main()
