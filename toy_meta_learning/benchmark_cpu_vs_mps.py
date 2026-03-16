#!/usr/bin/env python3
"""Benchmark CPU vs MPS performance for meta-training."""

import torch
import yaml
import time
import numpy as np
from pathlib import Path

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


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if len(config) == 1 and isinstance(list(config.values())[0], dict):
        config = list(config.values())[0]
    return config


def benchmark_device(device: str, num_iters: int = 60) -> float:
    """Benchmark meta-training on specified device.

    Args:
        device: 'cpu' or 'mps'
        num_iters: Number of iterations to benchmark

    Returns:
        Average iterations per minute
    """
    print(f"\n{'='*70}")
    print(f"BENCHMARKING: {device.upper()}")
    print(f"{'='*70}")

    # Load config
    config = load_config('config/gradient_matching_meta_improved.yaml')

    # Initialize learned loss
    learned_loss = NeuralPULoss(
        hidden_dim=config['loss_hidden_dim'],
        eps=1e-7,
        l05_lambda=0.0,
        init_mode='xavier_uniform',
        init_scale=1.0,
    ).to(device)

    # Meta-optimizer
    meta_optimizer = torch.optim.AdamW(
        learned_loss.parameters(),
        lr=config['meta_lr'],
        betas=tuple(config['meta_betas']),
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

    # Warmup (2 iterations)
    print("Warming up...")
    meta_optimizer.zero_grad()
    for _ in range(2):
        checkpoints = pool.checkpoints
        total_grad_matching_loss = torch.tensor(0.0, device=device)
        updated_checkpoints = []

        for checkpoint in checkpoints:
            model = create_model_from_checkpoint(checkpoint, device)
            params = {
                name: param.clone().detach().requires_grad_(True)
                for name, param in model.named_parameters()
            }

            task = GaussianBlobTask(**checkpoint['task_config'])
            x, y_true, y_pu = sample_batch_deterministic(
                task, checkpoint['step_count'],
                batch_size=config['batch_size'], device=device
            )

            grad_match_loss, _ = compute_gradient_mse(
                model, params, x, y_pu, y_true,
                learned_loss, device
            )
            total_grad_matching_loss = total_grad_matching_loss + grad_match_loss

            updated_ckpt = advance_checkpoint_one_step(
                checkpoint, learned_loss, device,
                batch_size=config['batch_size']
            )
            updated_checkpoints.append(updated_ckpt)

        meta_loss = total_grad_matching_loss / len(checkpoints)
        meta_loss.backward()
        meta_optimizer.step()
        meta_optimizer.zero_grad()

        pool.checkpoints = updated_checkpoints

    # Actual benchmark
    print(f"Running benchmark for {num_iters} iterations...")
    iteration_times = []

    for i in range(num_iters):
        iter_start = time.time()

        checkpoints = pool.checkpoints
        total_grad_matching_loss = torch.tensor(0.0, device=device)
        updated_checkpoints = []

        for checkpoint in checkpoints:
            model = create_model_from_checkpoint(checkpoint, device)
            params = {
                name: param.clone().detach().requires_grad_(True)
                for name, param in model.named_parameters()
            }

            task = GaussianBlobTask(**checkpoint['task_config'])
            x, y_true, y_pu = sample_batch_deterministic(
                task, checkpoint['step_count'],
                batch_size=config['batch_size'], device=device
            )

            grad_match_loss, _ = compute_gradient_mse(
                model, params, x, y_pu, y_true,
                learned_loss, device
            )
            total_grad_matching_loss = total_grad_matching_loss + grad_match_loss

            updated_ckpt = advance_checkpoint_one_step(
                checkpoint, learned_loss, device,
                batch_size=config['batch_size']
            )
            updated_checkpoints.append(updated_ckpt)

        meta_loss = total_grad_matching_loss / len(checkpoints)
        meta_loss.backward()
        meta_optimizer.step()
        meta_optimizer.zero_grad()

        pool.checkpoints = updated_checkpoints

        iter_time = time.time() - iter_start
        iteration_times.append(iter_time)

        if (i + 1) % 10 == 0:
            avg_time = np.mean(iteration_times[-10:])
            iters_per_min = 60.0 / avg_time
            print(f"  Iter {i+1}/{num_iters}: {iters_per_min:.1f} iters/min (last 10 avg)")

    # Calculate stats
    avg_time_per_iter = np.mean(iteration_times)
    iters_per_min = 60.0 / avg_time_per_iter

    print(f"\n{device.upper()} Results:")
    print(f"  Average time per iteration: {avg_time_per_iter:.3f}s")
    print(f"  Iterations per minute: {iters_per_min:.1f}")
    print(f"  Std dev: {np.std(iteration_times):.3f}s")

    return iters_per_min


def main():
    print("META-TRAINING PERFORMANCE BENCHMARK")
    print("Comparing CPU vs MPS for gradient matching meta-learning")
    print(f"Configuration: 256 checkpoints, hidden_dim=128")

    # Benchmark CPU
    cpu_speed = benchmark_device('cpu', num_iters=60)

    # Benchmark MPS (if available)
    if torch.backends.mps.is_available():
        mps_speed = benchmark_device('mps', num_iters=60)

        # Compare
        print(f"\n{'='*70}")
        print("COMPARISON")
        print(f"{'='*70}")
        print(f"CPU:  {cpu_speed:.1f} iters/min")
        print(f"MPS:  {mps_speed:.1f} iters/min")

        if cpu_speed > mps_speed:
            speedup = cpu_speed / mps_speed
            print(f"\nCPU is {speedup:.2f}x FASTER than MPS")
        else:
            speedup = mps_speed / cpu_speed
            print(f"\nMPS is {speedup:.2f}x FASTER than CPU")

        print(f"\nRecommendation: Use {'CPU' if cpu_speed > mps_speed else 'MPS'} for this workload")
    else:
        print("\nMPS not available on this system")


if __name__ == '__main__':
    main()
