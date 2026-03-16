#!/usr/bin/env python3
"""Validation utilities for PU meta-learning."""

import torch
import torch.nn as nn
import numpy as np
from models.simple_mlp import SimpleMLP
from loss.baseline_losses import PUDRaNaiveLoss, VPUNoMixUpLoss


def set_deterministic_seed(task_idx, model_type):
    """Set deterministic seed based on task index and model type.

    This ensures each validation run uses the same random state for
    model initialization and training, making results directly comparable.
    """
    # Use a consistent seed for each (task, model) pair
    base_seed = 50000  # High number to avoid collision with task generation seeds
    seed = base_seed + task_idx * 100 + hash(model_type) % 100

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_from_scratch_validation(val_tasks, loss_fn, config, device, cached_baselines=None):
    """Train models from scratch on validation tasks and evaluate with BCE.

    DETERMINISTIC: Uses fixed random seeds for each task to ensure all validations
    are directly comparable across iterations.

    Args:
        val_tasks: List of GaussianBlobTask validation tasks
        loss_fn: Learned PU loss function
        config: Configuration dictionary
        device: Device to train on
        cached_baselines: Optional dict with cached baseline results

    Returns:
        results: Dict with averaged BCE scores for each method
        cached_baselines: Updated baseline cache
    """
    bce_fn = nn.BCEWithLogitsLoss()

    # Convergence-based training parameters
    max_train_steps = config.get('max_train_steps', 500)
    patience = config.get('convergence_patience', 20)  # Steps without improvement before stopping

    results = {
        'learned': [],
        'learned_best': [],  # Best score during training (not just final)
    }

    # Initialize baseline cache if needed
    if cached_baselines is None:
        cached_baselines = {
            'oracle': [],
            'oracle_best': [],  # Best during training
            'naive': [],
            'naive_best': [],
            'pudra_naive': [],
            'pudra_naive_best': [],
            'vpu_nomixup': [],
            'vpu_nomixup_best': [],
            'converged_steps': None  # Will store the fixed step count
        }
        compute_baselines = True
    else:
        compute_baselines = False

    # Get fixed step count for learned loss (from baseline convergence)
    fixed_steps = cached_baselines.get('converged_steps', None)

    # Track maximum steps needed across all tasks (for baseline calibration)
    max_steps_across_tasks = 0

    # For baseline calibration, we need to train ALL tasks together
    # until ALL baselines on ALL tasks converge
    if compute_baselines:
        # Initialize header
        print(f"\n{'='*70}")
        print("BASELINE CALIBRATION - Fair comparison training")
        print(f"{'='*70}")
        print(f"  All baselines on ALL tasks train together")
        print(f"  Stop when: ALL baselines on ALL tasks converge + {patience} extra steps")
        print(f"  Convergence: {patience} consecutive steps without improvement")
        print(f"  Max steps: {max_train_steps}")
        print(f"  Training 4 baselines x {len(val_tasks)} tasks = {4 * len(val_tasks)} total")
        print(f"{'='*70}\n")

        # Initialize models and optimizers for all tasks
        task_data = []
        for task_idx, task in enumerate(val_tasks):
            # Set global seed before creating dataloaders for determinism
            set_deterministic_seed(task_idx, 'dataloader')

            dataloaders = task.get_dataloaders(
                batch_size=64,
                num_train=1000,
                num_val=500,
                num_test=500
            )

            # Initialize all baseline models for this task
            set_deterministic_seed(task_idx, 'oracle')
            model_oracle = SimpleMLP(2, config.get('model_hidden_dims', [32, 32])).to(device)
            optimizer_oracle = torch.optim.Adam(model_oracle.parameters(), lr=0.001)

            set_deterministic_seed(task_idx, 'naive')
            model_naive = SimpleMLP(2, config.get('model_hidden_dims', [32, 32])).to(device)
            optimizer_naive = torch.optim.Adam(model_naive.parameters(), lr=0.001)

            set_deterministic_seed(task_idx, 'pudra')
            model_pudra = SimpleMLP(2, config.get('model_hidden_dims', [32, 32])).to(device)
            optimizer_pudra = torch.optim.Adam(model_pudra.parameters(), lr=0.001)
            pudra_loss = PUDRaNaiveLoss().to(device)

            set_deterministic_seed(task_idx, 'vpu')
            model_vpu = SimpleMLP(2, config.get('model_hidden_dims', [32, 32])).to(device)
            optimizer_vpu = torch.optim.Adam(model_vpu.parameters(), lr=0.001)
            vpu_loss = VPUNoMixUpLoss().to(device)

            task_data.append({
                'task_idx': task_idx,
                'dataloaders': dataloaders,
                'models': {
                    'oracle': model_oracle,
                    'naive': model_naive,
                    'pudra': model_pudra,
                    'vpu': model_vpu
                },
                'optimizers': {
                    'oracle': optimizer_oracle,
                    'naive': optimizer_naive,
                    'pudra': optimizer_pudra,
                    'vpu': optimizer_vpu
                },
                'losses': {
                    'pudra': pudra_loss,
                    'vpu': vpu_loss
                },
                'best_losses': {
                    'oracle': float('inf'),
                    'naive': float('inf'),
                    'pudra': float('inf'),
                    'vpu': float('inf')
                },
                'steps_since_improvement': {
                    'oracle': 0,
                    'naive': 0,
                    'pudra': 0,
                    'vpu': 0
                },
                'converged': {
                    'oracle': False,
                    'naive': False,
                    'pudra': False,
                    'vpu': False
                },
                'batch_iterator': iter(dataloaders['train'])
            })

        # Train all tasks together until all converge
        step_count = 0
        all_converged_across_all_tasks = False
        steps_after_all_converged = 0
        training_complete = False

        print("Training all baselines on all tasks together...")

        while step_count < max_train_steps and not training_complete:
            # Train one step on each task
            for task_info in task_data:
                try:
                    batch = next(task_info['batch_iterator'])
                except StopIteration:
                    # Restart iterator for this task
                    task_info['batch_iterator'] = iter(task_info['dataloaders']['train'])
                    batch = next(task_info['batch_iterator'])

                x, y_true, y_pu = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                task_idx = task_info['task_idx']

                # Train Oracle
                task_info['optimizers']['oracle'].zero_grad()
                outputs_oracle = task_info['models']['oracle'](x).squeeze(-1)
                loss_oracle = bce_fn(outputs_oracle, y_true)
                loss_oracle.backward()
                task_info['optimizers']['oracle'].step()

                # Train Naive
                y_naive = (y_pu + 1) / 2
                task_info['optimizers']['naive'].zero_grad()
                outputs_naive = task_info['models']['naive'](x).squeeze(-1)
                loss_naive = bce_fn(outputs_naive, y_naive)
                loss_naive.backward()
                task_info['optimizers']['naive'].step()

                # Train PUDRa
                task_info['optimizers']['pudra'].zero_grad()
                outputs_pudra = task_info['models']['pudra'](x).squeeze(-1)
                loss_pudra = task_info['losses']['pudra'](outputs_pudra, y_pu, mode='pu')
                loss_pudra.backward()
                task_info['optimizers']['pudra'].step()

                # Train VPU
                task_info['optimizers']['vpu'].zero_grad()
                outputs_vpu = task_info['models']['vpu'](x).squeeze(-1)
                loss_vpu = task_info['losses']['vpu'](outputs_vpu, y_pu, mode='pu')
                loss_vpu.backward()
                task_info['optimizers']['vpu'].step()

                # Check convergence for each baseline on this task
                for method in ['oracle', 'naive', 'pudra', 'vpu']:
                    if method == 'oracle':
                        current_loss = loss_oracle.item()
                    elif method == 'naive':
                        current_loss = loss_naive.item()
                    elif method == 'pudra':
                        current_loss = loss_pudra.item()
                    else:  # vpu
                        current_loss = loss_vpu.item()

                    if not task_info['converged'][method]:
                        # Check if new best
                        if current_loss < task_info['best_losses'][method]:
                            task_info['best_losses'][method] = current_loss
                            task_info['steps_since_improvement'][method] = 0
                        else:
                            task_info['steps_since_improvement'][method] += 1

                        # Check convergence
                        if task_info['steps_since_improvement'][method] >= patience:
                            task_info['converged'][method] = True
                            print(f"  Task {task_idx}, {method}: CONVERGED at step {step_count+1} (best loss: {task_info['best_losses'][method]:.6f})")

            step_count += 1

            # Check if ALL baselines on ALL tasks have converged
            if not all_converged_across_all_tasks:
                all_converged_across_all_tasks = all(
                    all(task_info['converged'].values())
                    for task_info in task_data
                )
                if all_converged_across_all_tasks:
                    print(f"\nAll baselines on all tasks converged at step {step_count}. Training {patience} more steps...")

            # Count steps after all converged
            if all_converged_across_all_tasks:
                steps_after_all_converged += 1
                if steps_after_all_converged >= patience:
                    print(f"Completed {patience} extra steps. Final step count: {step_count}")
                    training_complete = True
                    break

        max_steps_across_tasks = step_count

        # Evaluate all baselines on all tasks
        print("\nEvaluating baselines on test sets...")
        for task_info in task_data:
            task_idx = task_info['task_idx']

            # Oracle
            task_info['models']['oracle'].eval()
            with torch.no_grad():
                test_loss = 0.0
                for batch in task_info['dataloaders']['test']:
                    x, y_true = batch[0].to(device), batch[1].to(device)
                    outputs = task_info['models']['oracle'](x).squeeze(-1)
                    loss = bce_fn(outputs, y_true)
                    test_loss += loss.item()
                test_loss /= len(task_info['dataloaders']['test'])
            cached_baselines['oracle'].append(test_loss)
            cached_baselines['oracle_best'].append(test_loss)

            # Naive
            task_info['models']['naive'].eval()
            with torch.no_grad():
                test_loss = 0.0
                for batch in task_info['dataloaders']['test']:
                    x, y_true = batch[0].to(device), batch[1].to(device)
                    outputs = task_info['models']['naive'](x).squeeze(-1)
                    loss = bce_fn(outputs, y_true)
                    test_loss += loss.item()
                test_loss /= len(task_info['dataloaders']['test'])
            cached_baselines['naive'].append(test_loss)
            cached_baselines['naive_best'].append(test_loss)

            # PUDRa
            task_info['models']['pudra'].eval()
            with torch.no_grad():
                test_loss = 0.0
                for batch in task_info['dataloaders']['test']:
                    x, y_true = batch[0].to(device), batch[1].to(device)
                    outputs = task_info['models']['pudra'](x).squeeze(-1)
                    loss = bce_fn(outputs, y_true)
                    test_loss += loss.item()
                test_loss /= len(task_info['dataloaders']['test'])
            cached_baselines['pudra_naive'].append(test_loss)
            cached_baselines['pudra_naive_best'].append(test_loss)

            # VPU
            task_info['models']['vpu'].eval()
            with torch.no_grad():
                test_loss = 0.0
                for batch in task_info['dataloaders']['test']:
                    x, y_true = batch[0].to(device), batch[1].to(device)
                    outputs = task_info['models']['vpu'](x).squeeze(-1)
                    loss = bce_fn(outputs, y_true)
                    test_loss += loss.item()
                test_loss /= len(task_info['dataloaders']['test'])
            cached_baselines['vpu_nomixup'].append(test_loss)
            cached_baselines['vpu_nomixup_best'].append(test_loss)

            print(f"  Task {task_idx}: Oracle={cached_baselines['oracle'][-1]:.6f}, Naive={cached_baselines['naive'][-1]:.6f}, PUDRa={cached_baselines['pudra_naive'][-1]:.6f}, VPU={cached_baselines['vpu_nomixup'][-1]:.6f}")

    # Train learned loss on each task (only if we have fixed_steps)
    for task_idx, task in enumerate(val_tasks):
        # Set global seed before creating dataloaders for determinism
        set_deterministic_seed(task_idx, 'dataloader')

        dataloaders = task.get_dataloaders(
            batch_size=64,
            num_train=1000,
            num_val=500,
            num_test=500
        )

        # === Train with LEARNED LOSS (only if we have fixed_steps) ===
        if fixed_steps is not None:
            set_deterministic_seed(task_idx, 'learned')
            model_learned = SimpleMLP(2, config.get('model_hidden_dims', [32, 32])).to(device)
            optimizer_learned = torch.optim.Adam(model_learned.parameters(), lr=0.001)

            # Train for exactly fixed_steps (determined from baseline convergence)
            step_count = 0
            best_val_loss = float('inf')
            model_learned.train()

            # Evaluate every 10 steps to track best
            eval_freq = 10

            while step_count < fixed_steps:
                for batch in dataloaders['train']:
                    if step_count >= fixed_steps:
                        break

                    x, y_pu = batch[0].to(device), batch[2].to(device)
                    optimizer_learned.zero_grad()
                    outputs = model_learned(x).squeeze(-1)
                    loss = loss_fn(outputs, y_pu, mode='pu')
                    loss.backward()
                    optimizer_learned.step()
                    step_count += 1

                    # Evaluate periodically to track best
                    if step_count % eval_freq == 0 or step_count == fixed_steps:
                        model_learned.eval()
                        with torch.no_grad():
                            val_loss = 0.0
                            for val_batch in dataloaders['test']:
                                x_val, y_val = val_batch[0].to(device), val_batch[1].to(device)
                                outputs_val = model_learned(x_val).squeeze(-1)
                                val_loss += bce_fn(outputs_val, y_val).item()
                            val_loss /= len(dataloaders['test'])
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                        model_learned.train()

            # Final evaluation
            model_learned.eval()
            with torch.no_grad():
                test_loss = 0.0
                for batch in dataloaders['test']:
                    x, y_true = batch[0].to(device), batch[1].to(device)
                    outputs = model_learned(x).squeeze(-1)
                    loss = bce_fn(outputs, y_true)
                    test_loss += loss.item()
                test_loss /= len(dataloaders['test'])
            results['learned'].append(test_loss)
            results['learned_best'].append(best_val_loss)

        # Baselines already trained (all together before this loop)

    # Save converged step count (only when computing baselines)
    if compute_baselines:
        cached_baselines['converged_steps'] = max_steps_across_tasks
        print(f"\n✓ Baseline convergence complete. Fixed training steps: {max_steps_across_tasks}")

    # Average results across tasks
    if fixed_steps is not None:
        results['learned'] = np.mean(results['learned'])
        results['learned_best'] = np.mean(results['learned_best'])
    for method in ['oracle', 'naive', 'pudra_naive', 'vpu_nomixup']:
        results[method] = np.mean(cached_baselines[method])
        results[f'{method}_best'] = np.mean(cached_baselines[f'{method}_best'])

    return results, cached_baselines
