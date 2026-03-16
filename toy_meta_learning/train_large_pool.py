#!/usr/bin/env python3
"""Train PU meta-learning with LARGE POOL + CHECKPOINT CURRICULUM.

Key innovations:
- 144 checkpoints (vs 24) for better diversity
- Checkpoint curriculum: periodically refresh 10% of pool with checkpoints
  trained using the current learned loss (co-evolution)
- Fixed validation sets for stable, deterministic metrics
- Independent oracle/naive/meta measurements

Expected: Better generalization, less overfitting
"""

import torch
import yaml
from pathlib import Path
import numpy as np
from torch.func import functional_call, grad
import torch.nn as nn
import time
from tqdm import tqdm
import sys

from models.simple_mlp import SimpleMLP
from loss.hierarchical_pu_loss import HierarchicalPULoss
from loss.baseline_losses import PUDRaNaiveLoss, VPUNoMixUpLoss
from tasks.task_pool import CheckpointPool
from tasks.gaussian_task import GaussianBlobTask


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if len(config) == 1 and isinstance(list(config.values())[0], dict):
        config = list(config.values())[0]
    return config


def get_device(config: dict) -> str:
    device_config = config.get('device', 'auto')
    if device_config == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    return device_config


if __name__ == '__main__':

if __name__ == '__main__':
            # Load large pool config
        config = load_config('config/toy_gaussian_meta_large_pool.yaml')
        device = get_device(config)

            # Create learnable hierarchical PU loss with PUDRa-naive initialization
        loss_l1_lambda = float(config.get('loss_l1_lambda', 0.0))
        loss_fn = HierarchicalPULoss(
        init_mode='pudra_inspired',  # Start from PUDRa-naive baseline
        init_scale=float(config.get('loss_init_scale', 0.01)),
        l1_lambda=loss_l1_lambda
        ).to(device)

            # Meta-optimizer
        meta_lr = float(config.get('meta_lr', 0.001))
        meta_weight_decay = float(config.get('meta_weight_decay', 1e-4))
        meta_optimizer = torch.optim.AdamW(
        loss_fn.parameters(),
        lr=meta_lr,
        weight_decay=meta_weight_decay
        )

            # Create large checkpoint pool
        pool = CheckpointPool(config)

            # Try to load cached checkpoint pool first
        if not pool.load_checkpoint_pool():
        print("Creating LARGE checkpoint pool (288 checkpoints)...")
        print("This will include:")
        print("  - 4 difficulties × 2 stds × 3 seeds × 4 methods × 3 epochs")
        print("  - Oracle checkpoints (trained with PN labels)")
        print("  - Naive checkpoints (trained with PU labels)")
        print("  - PUDRa-naive checkpoints (trained with PUDRa baseline)")
        print("  - VPU-NoMixUp checkpoints (trained with VPU baseline)")
        print()
        pool.create_checkpoint_pool(device=device)

        print()
        print("="*70)
        print("LARGE POOL + CHECKPOINT CURRICULUM PU META-LEARNING")
        print("="*70)
        print(f"Device: {device}")
        print(f"Total checkpoints: {len(pool.checkpoints)}")
        print(f"Initial loss:\n{loss_fn}")
        print(f"L1 regularization: lambda={loss_l1_lambda}")
        print()

            # Count checkpoints by training method
        oracle_count = sum(1 for ckpt in pool.checkpoints if ckpt['task_config']['training_method'] == 'oracle')
        naive_count = sum(1 for ckpt in pool.checkpoints if ckpt['task_config']['training_method'] == 'naive')
        pudra_count = sum(1 for ckpt in pool.checkpoints if ckpt['task_config']['training_method'] == 'pudra_naive')
        vpu_count = sum(1 for ckpt in pool.checkpoints if ckpt['task_config']['training_method'] == 'vpu_nomixup')
        print(f"Oracle checkpoints (PN-trained):       {oracle_count}")
        print(f"Naive checkpoints (PU-trained):        {naive_count}")
        print(f"PUDRa-naive checkpoints (baseline):    {pudra_count}")
        print(f"VPU-NoMixUp checkpoints (baseline):    {vpu_count}")
        print()

            # Analyze checkpoint epochs distribution
        from collections import Counter
        epoch_counts = Counter([ckpt['epoch'] for ckpt in pool.checkpoints])
        print("Checkpoints by epoch:")
        for epoch in sorted(epoch_counts.keys()):
        print(f"  Epoch {epoch:3d}: {epoch_counts[epoch]:3d} checkpoints")
        print()

            # Print optimization details
        print("="*70)
        print("CONFIGURATION")
        print("="*70)
        print(f"Inner steps: {config.get('inner_steps', 3)} (maximum task adaptation)")
        print(f"Meta optimizer: AdamW (lr={meta_lr}, weight_decay={meta_weight_decay})")
        print(f"L1 regularization: lambda={loss_l1_lambda} (moderate sparsity)")
        print(f"Meta batch size: {config.get('meta_batch_size', 12)}")
        print(f"Meta iterations: {config.get('meta_iterations', 300)}")
        print(f"Checkpoint curriculum: {config.get('use_checkpoint_curriculum', False)}")
        if config.get('use_checkpoint_curriculum', False):
        print(f"  - Start iteration: {config.get('curriculum_start_iter', 100)}")
        print(f"  - Refresh frequency: every {config.get('curriculum_refresh_freq', 50)} iterations")
        print(f"  - Refresh ratio: {config.get('curriculum_refresh_ratio', 0.1)*100:.0f}% of pool")
        print()


            def normalize_loss_parameters(loss_fn):
        """Normalize loss parameters so max absolute value = 1."""
        with torch.no_grad():
            params = loss_fn.get_parameters()
            max_abs_val = torch.abs(params).max()
            if max_abs_val > 1e-8:
                scale_factor = 1.0 / max_abs_val
                for param in loss_fn.parameters():
                    param.data *= scale_factor


            def compute_task_loss(model, params, x, y_pu, loss_fn):
        """Compute PU loss for inner loop training."""
        outputs = functional_call(model, params, x)
        return loss_fn(outputs, y_pu, mode='pu')


            def inner_loop_step(model, params, x, y_pu, loss_fn, lr):
        """Single PU adaptation step."""
        grads = grad(lambda m, p, x, y: compute_task_loss(m, p, x, y, loss_fn), argnums=1)(
            model, params, x, y_pu
        )
        return {name: param - lr * grads[name] for name, param in params.items()}


            def inner_loop(model, train_loader, loss_fn, num_steps, lr, device):
        """Inner loop: Adapt model on PU task using learned loss."""
        params = dict(model.named_parameters())
        for step in range(num_steps):
            for batch in train_loader:
                x, y_pu = batch[0].to(device), batch[2].to(device)
                params = inner_loop_step(model, params, x, y_pu, loss_fn, lr)
                break  # Only one step per batch
        return params


            def evaluate_bce_on_gt(model, params, val_loader, device):
        """Evaluate adapted model on ground truth labels using BCE."""
        bce_fn = nn.BCEWithLogitsLoss()
        total_loss = 0.0
        total_samples = 0
        for batch in val_loader:
            x = batch[0].to(device)
            y_true = batch[1].to(device)
            outputs = functional_call(model, params, x).squeeze()
            loss = bce_fn(outputs, y_true)
            total_loss += loss.item() * len(x)
            total_samples += len(x)
        return total_loss / total_samples if total_samples > 0 else 0.0


            def refresh_checkpoint_pool(pool, loss_fn, config, device, refresh_ratio=0.1):
        """Generate fresh checkpoints with current learned loss and temporarily replace pool subset.

        Returns: (temp_indices, original_checkpoints) for restoration
        """
        num_to_replace = max(1, int(len(pool.checkpoints) * refresh_ratio))

        # Sample random indices to replace
        replace_indices = np.random.choice(len(pool.checkpoints), size=num_to_replace, replace=False)

        # Save original checkpoints for restoration
        original_checkpoints = [pool.checkpoints[i] for i in replace_indices]

        # Generate fresh task configs
        fresh_checkpoints = []
        for _ in range(num_to_replace):
            # Random task configuration
            task_config = {
                'num_dimensions': config.get('num_dimensions', 2),
                'mean_separation': float(np.random.choice(config.get('mean_separations', [2.5]))),
                'std': float(np.random.choice(config.get('stds', [1.0]))),
                'prior': 0.5,
                'labeling_freq': 0.3,
                'num_samples': config.get('num_samples_per_task', 1000),
                'mode': 'pu',
                'negative_labeling_freq': 0.3,
                'training_method': np.random.choice(['oracle', 'naive', 'pudra_naive', 'vpu_nomixup']),
            }

            # Create task (exclude training_method from task creation)
            task_params = {k: v for k, v in task_config.items() if k != 'training_method'}
            task = GaussianBlobTask(**task_params, seed=np.random.randint(0, 10000))
            dataloaders = task.get_dataloaders(batch_size=64, num_train=1000, num_val=500, num_test=500)

            # Train fresh model with LEARNED LOSS (curriculum)
            model = SimpleMLP(task_config['num_dimensions'], config.get('model_hidden_dims', [32, 32])).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.get('checkpoint_train_lr', 0.001))

            # Quick training (20 epochs with learned loss)
            for epoch in range(20):
                model.train()
                for batch in dataloaders['train']:
                    x, y_true, y_pu = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                    optimizer.zero_grad()
                    outputs = model(x).squeeze(-1)

                    # Use learned loss for curriculum checkpoints
                    loss = loss_fn(outputs, y_pu, mode='pu')
                    loss.backward()
                    optimizer.step()

            # Create checkpoint entry
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'task_config': task_config,
                'epoch': 20,  # Mark as curriculum checkpoint
            }
            fresh_checkpoints.append(checkpoint)

        # Temporarily replace checkpoints
        for idx, fresh_ckpt in zip(replace_indices, fresh_checkpoints):
            pool.checkpoints[idx] = fresh_ckpt

        return replace_indices, original_checkpoints


            def train_from_scratch_validation(val_tasks, loss_fn, config, device, cached_baselines=None):
        """Train models from scratch on validation tasks and evaluate.

        Args:
            val_tasks: List of validation task objects
            loss_fn: Learned hierarchical loss
            config: Configuration dict
            device: Device to train on
            cached_baselines: Dict of cached baseline results (oracle, naive, pudra, vpu) or None

        Returns:
            results: Dict with 'oracle', 'naive', 'learned', 'pudra_naive', 'vpu_nomixup' BCE scores
            cached_baselines: Updated cache for next iteration
        """
        bce_fn = nn.BCEWithLogitsLoss()
        train_epochs = 50  # Train for 50 epochs per method

        results = {
            'learned': []
        }

        # Initialize results and cache
        if cached_baselines is None:
            cached_baselines = {
                'oracle': [],
                'naive': [],
                'pudra_naive': [],
                'vpu_nomixup': []
            }
            compute_baselines = True
        else:
            compute_baselines = False
            # Use cached results
            for method in ['oracle', 'naive', 'pudra_naive', 'vpu_nomixup']:
                results[method] = cached_baselines[method].copy()

        for task_idx, task in enumerate(val_tasks):
            dataloaders = task.get_dataloaders(batch_size=64, num_train=1000, num_val=500, num_test=500)

            # === Learned loss (always compute, changes every iteration) ===
            model_learned = SimpleMLP(2, config.get('model_hidden_dims', [32, 32])).to(device)
            optimizer_learned = torch.optim.Adam(model_learned.parameters(), lr=0.001)

            for epoch in range(train_epochs):
                model_learned.train()
                for batch in dataloaders['train']:
                    x, y_true, y_pu = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                    optimizer_learned.zero_grad()
                    outputs = model_learned(x).squeeze(-1)
                    loss = loss_fn(outputs, y_pu, mode='pu')
                    loss.backward()
                    optimizer_learned.step()

            # Evaluate learned
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

            # === Baselines (only compute if not cached) ===
            if compute_baselines:
                # Oracle
                model_oracle = SimpleMLP(2, config.get('model_hidden_dims', [32, 32])).to(device)
                optimizer_oracle = torch.optim.Adam(model_oracle.parameters(), lr=0.001)

                for epoch in range(train_epochs):
                    model_oracle.train()
                    for batch in dataloaders['train']:
                        x, y_true = batch[0].to(device), batch[1].to(device)
                        optimizer_oracle.zero_grad()
                        outputs = model_oracle(x).squeeze(-1)
                        loss = bce_fn(outputs, y_true)
                        loss.backward()
                        optimizer_oracle.step()

                model_oracle.eval()
                with torch.no_grad():
                    test_loss = 0.0
                    for batch in dataloaders['test']:
                        x, y_true = batch[0].to(device), batch[1].to(device)
                        outputs = model_oracle(x).squeeze(-1)
                        loss = bce_fn(outputs, y_true)
                        test_loss += loss.item()
                    test_loss /= len(dataloaders['test'])
                cached_baselines['oracle'].append(test_loss)

                # Naive
                model_naive = SimpleMLP(2, config.get('model_hidden_dims', [32, 32])).to(device)
                optimizer_naive = torch.optim.Adam(model_naive.parameters(), lr=0.001)

                for epoch in range(train_epochs):
                    model_naive.train()
                    for batch in dataloaders['train']:
                        x, y_pu = batch[0].to(device), batch[2].to(device)
                        y_naive = torch.where(y_pu == 1, torch.ones_like(y_pu), torch.zeros_like(y_pu))
                        optimizer_naive.zero_grad()
                        outputs = model_naive(x).squeeze(-1)
                        loss = bce_fn(outputs, y_naive)
                        loss.backward()
                        optimizer_naive.step()

                model_naive.eval()
                with torch.no_grad():
                    test_loss = 0.0
                    for batch in dataloaders['test']:
                        x, y_true = batch[0].to(device), batch[1].to(device)
                        outputs = model_naive(x).squeeze(-1)
                        loss = bce_fn(outputs, y_true)
                        test_loss += loss.item()
                    test_loss /= len(dataloaders['test'])
                cached_baselines['naive'].append(test_loss)

                # PUDRa-naive
                model_pudra = SimpleMLP(2, config.get('model_hidden_dims', [32, 32])).to(device)
                optimizer_pudra = torch.optim.Adam(model_pudra.parameters(), lr=0.001)
                pudra_loss = PUDRaNaiveLoss().to(device)

                for epoch in range(train_epochs):
                    model_pudra.train()
                    for batch in dataloaders['train']:
                        x, y_pu = batch[0].to(device), batch[2].to(device)
                        optimizer_pudra.zero_grad()
                        outputs = model_pudra(x).squeeze(-1)
                        loss = pudra_loss(outputs, y_pu, mode='pu')
                        loss.backward()
                        optimizer_pudra.step()

                model_pudra.eval()
                with torch.no_grad():
                    test_loss = 0.0
                    for batch in dataloaders['test']:
                        x, y_true = batch[0].to(device), batch[1].to(device)
                        outputs = model_pudra(x).squeeze(-1)
                        loss = bce_fn(outputs, y_true)
                        test_loss += loss.item()
                    test_loss /= len(dataloaders['test'])
                cached_baselines['pudra_naive'].append(test_loss)

                # VPU-NoMixUp
                model_vpu = SimpleMLP(2, config.get('model_hidden_dims', [32, 32])).to(device)
                optimizer_vpu = torch.optim.Adam(model_vpu.parameters(), lr=0.001)
                vpu_loss = VPUNoMixUpLoss().to(device)

                for epoch in range(train_epochs):
                    model_vpu.train()
                    for batch in dataloaders['train']:
                        x, y_pu = batch[0].to(device), batch[2].to(device)
                        optimizer_vpu.zero_grad()
                        outputs = model_vpu(x).squeeze(-1)
                        loss = vpu_loss(outputs, y_pu, mode='pu')
                        loss.backward()
                        optimizer_vpu.step()

                model_vpu.eval()
                with torch.no_grad():
                    test_loss = 0.0
                    for batch in dataloaders['test']:
                        x, y_true = batch[0].to(device), batch[1].to(device)
                        outputs = model_vpu(x).squeeze(-1)
                        loss = bce_fn(outputs, y_true)
                        test_loss += loss.item()
                    test_loss /= len(dataloaders['test'])
                cached_baselines['vpu_nomixup'].append(test_loss)

        # Average results across validation tasks
        results['learned'] = np.mean(results['learned'])
        for method in ['oracle', 'naive', 'pudra_naive', 'vpu_nomixup']:
            results[method] = np.mean(cached_baselines[method])

        return results, cached_baselines


            # Create fixed validation tasks for train-from-scratch evaluation
        print("Creating validation tasks...")
        num_val_tasks = 3  # Small number for faster validation
        val_tasks = []
        for i in range(num_val_tasks):
        task = GaussianBlobTask(
            num_dimensions=2,
            mean_separation=2.5,
            std=1.0,
            prior=0.5,
            labeling_freq=0.3,
            num_samples=1000,
            seed=9000 + i,  # Fixed seeds for reproducibility
            mode='pu',
            negative_labeling_freq=0.3,
        )
        val_tasks.append(task)

        print(f"Created {len(val_tasks)} validation tasks for train-from-scratch evaluation")
        print()

            # Initialize baseline cache (will be computed once, then reused)
        cached_baselines = None

            # Training loop
        print("Starting large pool + curriculum meta-training...")
        print(f"  - {len(pool.checkpoints)} diverse checkpoints")
        print(f"  - 8 inner steps for maximum adaptation")
        print(f"  - Meta LR: {meta_lr}")
        print(f"  - L1 regularization: λ={loss_l1_lambda}")
        print()
        sys.stdout.flush()

        start_time = time.time()
        total_samples_processed = 0

            # Progress bar
        pbar = tqdm(range(config['meta_iterations']), desc="Meta-training",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        for iteration in pbar:
        iter_start = time.time()

        # Checkpoint curriculum: periodically refresh pool
        if (config.get('use_checkpoint_curriculum', False) and
            iteration >= config.get('curriculum_start_iter', 100) and
            iteration % config.get('curriculum_refresh_freq', 50) == 0):

            tqdm.write(f"\n[Curriculum] Refreshing {config.get('curriculum_refresh_ratio', 0.1)*100:.0f}% of pool...")
            sys.stdout.flush()

            temp_indices, original_ckpts = refresh_checkpoint_pool(
                pool, loss_fn, config, device,
                refresh_ratio=config.get('curriculum_refresh_ratio', 0.1)
            )

        # Sample checkpoints
        checkpoint_indices = pool.sample_batch(config['meta_batch_size'])

        total_meta_loss = torch.tensor(0.0, device=device)
        iter_samples = 0

        for ckpt_idx in checkpoint_indices:
            checkpoint, task, dataloaders = pool.get_checkpoint(ckpt_idx)

            # Create model
            model = SimpleMLP(
                input_dim=checkpoint['task_config']['num_dimensions'],
                hidden_dims=config.get('model_hidden_dims', [32, 32]),
            ).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])

            # Inner loop: Train on PU labels
            adapted_params = inner_loop(
                model=model,
                train_loader=dataloaders['train'],
                loss_fn=loss_fn,
                num_steps=config.get('inner_steps', 3),
                lr=config.get('inner_lr', 0.01),
                device=device,
            )

            iter_samples += config.get('num_samples_per_task', 1000) * config.get('inner_steps', 1)

            # Meta-objective: Evaluate on ground truth with BCE
            val_batch = next(iter(dataloaders['val']))
            x_val = val_batch[0].to(device)
            y_true_val = val_batch[1].to(device)

            outputs_val = functional_call(model, adapted_params, x_val).squeeze()
            bce_fn = nn.BCEWithLogitsLoss()
            val_bce = bce_fn(outputs_val, y_true_val)

            total_meta_loss = total_meta_loss + val_bce

        # Average and optimize
        avg_meta_loss = total_meta_loss / len(checkpoint_indices)

        meta_optimizer.zero_grad()
        avg_meta_loss.backward()
        meta_optimizer.step()

        # Normalize loss parameters
        normalize_loss_parameters(loss_fn)

        # Track throughput
        total_samples_processed += iter_samples
        iter_time = time.time() - iter_start
        iter_throughput = iter_samples / iter_time if iter_time > 0 else 0

        # Update progress bar
        pbar.set_postfix({
            'meta_loss': f'{avg_meta_loss.item():.4f}',
            'samples/s': f'{int(iter_throughput)}'
        })

        # Restore original pool after curriculum evaluation
        if (config.get('use_checkpoint_curriculum', False) and
            iteration >= config.get('curriculum_start_iter', 100) and
            iteration % config.get('curriculum_refresh_freq', 50) == 0):

            # Restore original checkpoints
            for idx, orig_ckpt in zip(temp_indices, original_ckpts):
                pool.checkpoints[idx] = orig_ckpt

        # Log with train-from-scratch validation
        if (iteration + 1) % config['log_freq'] == 0:
            elapsed_time = time.time() - start_time
            avg_throughput = total_samples_processed / elapsed_time if elapsed_time > 0 else 0

            # Train from scratch on validation tasks
            val_results, cached_baselines = train_from_scratch_validation(
                val_tasks, loss_fn, config, device, cached_baselines
            )

            tqdm.write(f"\nIteration {iteration + 1}/{config['meta_iterations']}")
            tqdm.write(f"  Training meta-loss:             {avg_meta_loss.item():.6f}")
            tqdm.write(f"  --- Train-from-Scratch Validation ---")
            tqdm.write(f"  Learned BCE:                    {val_results['learned']:.6f}")
            tqdm.write(f"  Oracle BCE (cached):            {val_results['oracle']:.6f}")
            tqdm.write(f"  Naive BCE (cached):             {val_results['naive']:.6f}")
            tqdm.write(f"  PUDRa-naive BCE (cached):       {val_results['pudra_naive']:.6f}")
            tqdm.write(f"  VPU-NoMixUp BCE (cached):       {val_results['vpu_nomixup']:.6f}")
            tqdm.write(f"  Throughput: {avg_throughput:,.0f} samples/min")

            # Show sparsity progress
            params = loss_fn.get_parameters().detach().cpu().numpy()
            near_zero = np.sum(np.abs(params) < 0.01)
            tqdm.write(f"  Sparsity: {near_zero}/27 params near zero ({near_zero/27*100:.1f}%)")
            tqdm.write("")
            sys.stdout.flush()

        print()
        print("="*70)
        print("FINAL RESULTS")
        print("="*70)
        print(f"\nFinal learned PU loss (L1 λ={loss_l1_lambda}):")
        print(loss_fn)

            # Final sparsity analysis
        params = loss_fn.get_parameters().detach().cpu().numpy()
        near_zero = np.sum(np.abs(params) < 0.01)
        print(f"\nParameter sparsity:")
        print(f"  Near-zero (|p| < 0.01): {near_zero}/27 ({near_zero/27*100:.1f}%)")
        print(f"  Max |parameter|: {np.abs(params).max():.4f}")
        print(f"  Mean |parameter|: {np.abs(params).mean():.4f}")
        print("="*70)

            # ======================================================================
            # TRAIN FROM SCRATCH EVALUATION WITH BASELINES
            # ======================================================================
        print()
        print("="*70)
        print("TRAIN FROM SCRATCH EVALUATION (5 METHODS)")
        print("="*70)
        print()
        print("Training five models from scratch on fresh tasks:")
        print("  1. Oracle: BCE loss on ground truth PN labels")
        print("  2. Naive: BCE loss on PU labels (unlabeled = negative)")
        print("  3. Learned: Learned hierarchical loss on PU labels")
        print("  4. PUDRa-naive: E_P[-log p + p] + E_U[p] baseline")
        print("  5. VPU-NoMixUp: log(E_all[φ]) - E_P[log φ] baseline")
        print()
        sys.stdout.flush()

        from loss.baseline_losses import PUDRaNaiveLoss, VPUNoMixUpLoss

            # Create baseline losses
        pudra_naive_loss = PUDRaNaiveLoss().to(device)
        vpu_nomixup_loss = VPUNoMixUpLoss().to(device)

            # Train on multiple fresh tasks and average results
        num_test_tasks = 5
        train_epochs = 100
        test_results = {
        'oracle': [],
        'naive': [],
        'learned': [],
        'pudra_naive': [],
        'vpu_nomixup': []
        }

        for task_idx in tqdm(range(num_test_tasks), desc="Testing on fresh tasks", leave=True):
        tqdm.write(f"\nTest task {task_idx + 1}/{num_test_tasks}")
        sys.stdout.flush()

        # Create a fresh test task (different seed)
        test_task = GaussianBlobTask(
            num_dimensions=2,
            mean_separation=2.5,
            std=1.0,
            prior=0.5,
            labeling_freq=0.3,
            num_samples=1000,
            seed=10000 + task_idx,
            mode='pu',
            negative_labeling_freq=0.3,
        )

        # Get dataloaders
        test_dataloaders = test_task.get_dataloaders(
            batch_size=64,
            num_train=1000,
            num_val=500,
            num_test=500,
        )

        # --- 1. Oracle: Train with BCE on ground truth ---
        model_oracle = SimpleMLP(2, [32, 32]).to(device)
        optimizer_oracle = torch.optim.Adam(model_oracle.parameters(), lr=0.001)
        bce_fn = nn.BCEWithLogitsLoss()

        for epoch in range(train_epochs):
            model_oracle.train()
            for batch in test_dataloaders['train']:
                x, y_true, y_pu = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                optimizer_oracle.zero_grad()
                outputs = model_oracle(x).squeeze(-1)
                loss = bce_fn(outputs, y_true)  # Ground truth
                loss.backward()
                optimizer_oracle.step()

        # Evaluate oracle
        model_oracle.eval()
        with torch.no_grad():
            test_loss = 0.0
            for batch in test_dataloaders['test']:
                x, y_true = batch[0].to(device), batch[1].to(device)
                outputs = model_oracle(x).squeeze(-1)
                loss = bce_fn(outputs, y_true)
                test_loss += loss.item()
            test_loss /= len(test_dataloaders['test'])
        test_results['oracle'].append(test_loss)

        # --- 2. Naive: Train with BCE on PU labels ---
        model_naive = SimpleMLP(2, [32, 32]).to(device)
        optimizer_naive = torch.optim.Adam(model_naive.parameters(), lr=0.001)

        for epoch in range(train_epochs):
            model_naive.train()
            for batch in test_dataloaders['train']:
                x, y_true, y_pu = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                # Convert PU labels: labeled positive (1) and unlabeled (-1) → treat unlabeled as negative (0)
                y_naive = torch.where(y_pu == 1, torch.ones_like(y_pu), torch.zeros_like(y_pu))
                optimizer_naive.zero_grad()
                outputs = model_naive(x).squeeze(-1)
                loss = bce_fn(outputs, y_naive)
                loss.backward()
                optimizer_naive.step()

        # Evaluate naive
        model_naive.eval()
        with torch.no_grad():
            test_loss = 0.0
            for batch in test_dataloaders['test']:
                x, y_true = batch[0].to(device), batch[1].to(device)
                outputs = model_naive(x).squeeze(-1)
                loss = bce_fn(outputs, y_true)
                test_loss += loss.item()
            test_loss /= len(test_dataloaders['test'])
        test_results['naive'].append(test_loss)

        # --- 3. Learned: Train with learned hierarchical loss ---
        model_learned = SimpleMLP(2, [32, 32]).to(device)
        optimizer_learned = torch.optim.Adam(model_learned.parameters(), lr=0.001)

        for epoch in range(train_epochs):
            model_learned.train()
            for batch in test_dataloaders['train']:
                x, y_true, y_pu = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                optimizer_learned.zero_grad()
                outputs = model_learned(x).squeeze(-1)
                loss = loss_fn(outputs, y_pu, mode='pu')  # Learned PU loss
                loss.backward()
                optimizer_learned.step()

        # Evaluate learned
        model_learned.eval()
        with torch.no_grad():
            test_loss = 0.0
            for batch in test_dataloaders['test']:
                x, y_true = batch[0].to(device), batch[1].to(device)
                outputs = model_learned(x).squeeze(-1)
                loss = bce_fn(outputs, y_true)
                test_loss += loss.item()
            test_loss /= len(test_dataloaders['test'])
        test_results['learned'].append(test_loss)

        # --- 4. PUDRa-naive: Train with PUDRa baseline ---
        model_pudra = SimpleMLP(2, [32, 32]).to(device)
        optimizer_pudra = torch.optim.Adam(model_pudra.parameters(), lr=0.001)

        for epoch in range(train_epochs):
            model_pudra.train()
            for batch in test_dataloaders['train']:
                x, y_true, y_pu = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                optimizer_pudra.zero_grad()
                outputs = model_pudra(x).squeeze(-1)
                loss = pudra_naive_loss(outputs, y_pu, mode='pu')
                loss.backward()
                optimizer_pudra.step()

        # Evaluate PUDRa-naive
        model_pudra.eval()
        with torch.no_grad():
            test_loss = 0.0
            for batch in test_dataloaders['test']:
                x, y_true = batch[0].to(device), batch[1].to(device)
                outputs = model_pudra(x).squeeze(-1)
                loss = bce_fn(outputs, y_true)
                test_loss += loss.item()
            test_loss /= len(test_dataloaders['test'])
        test_results['pudra_naive'].append(test_loss)

        # --- 5. VPU-NoMixUp: Train with VPU baseline ---
        model_vpu = SimpleMLP(2, [32, 32]).to(device)
        optimizer_vpu = torch.optim.Adam(model_vpu.parameters(), lr=0.001)

        for epoch in range(train_epochs):
            model_vpu.train()
            for batch in test_dataloaders['train']:
                x, y_true, y_pu = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                optimizer_vpu.zero_grad()
                outputs = model_vpu(x).squeeze(-1)
                loss = vpu_nomixup_loss(outputs, y_pu, mode='pu')
                loss.backward()
                optimizer_vpu.step()

        # Evaluate VPU-NoMixUp
        model_vpu.eval()
        with torch.no_grad():
            test_loss = 0.0
            for batch in test_dataloaders['test']:
                x, y_true = batch[0].to(device), batch[1].to(device)
                outputs = model_vpu(x).squeeze(-1)
                loss = bce_fn(outputs, y_true)
                test_loss += loss.item()
            test_loss /= len(test_dataloaders['test'])
        test_results['vpu_nomixup'].append(test_loss)

        tqdm.write(f"  Oracle:       {test_results['oracle'][-1]:.6f}")
        tqdm.write(f"  Naive:        {test_results['naive'][-1]:.6f}")
        tqdm.write(f"  Learned:      {test_results['learned'][-1]:.6f}")
        tqdm.write(f"  PUDRa-naive:  {test_results['pudra_naive'][-1]:.6f}")
        tqdm.write(f"  VPU-NoMixUp:  {test_results['vpu_nomixup'][-1]:.6f}")
        sys.stdout.flush()

            # Print final averages
        print()
        print("="*70)
        print(f"TRAIN FROM SCRATCH RESULTS (averaged over {num_test_tasks} tasks)")
        print("="*70)

        oracle_avg = np.mean(test_results['oracle'])
        naive_avg = np.mean(test_results['naive'])
        learned_avg = np.mean(test_results['learned'])
        pudra_avg = np.mean(test_results['pudra_naive'])
        vpu_avg = np.mean(test_results['vpu_nomixup'])

        print(f"Oracle BCE (ground truth):  {oracle_avg:.6f} ± {np.std(test_results['oracle']):.6f}")
        print(f"Naive BCE (PU as PN):       {naive_avg:.6f} ± {np.std(test_results['naive']):.6f}")
        print(f"Learned Loss (PU):          {learned_avg:.6f} ± {np.std(test_results['learned']):.6f}")
        print(f"PUDRa-naive (baseline):     {pudra_avg:.6f} ± {np.std(test_results['pudra_naive']):.6f}")
        print(f"VPU-NoMixUp (baseline):     {vpu_avg:.6f} ± {np.std(test_results['vpu_nomixup']):.6f}")
        print()

            # Rank methods by performance
        methods = {
        'Oracle': oracle_avg,
        'Naive': naive_avg,
        'Learned': learned_avg,
        'PUDRa-naive': pudra_avg,
        'VPU-NoMixUp': vpu_avg
        }
        ranked = sorted(methods.items(), key=lambda x: x[1])

        print("Performance ranking (lower is better):")
        for rank, (method, score) in enumerate(ranked, 1):
        gap_vs_oracle = (score / oracle_avg - 1) * 100
        print(f"  {rank}. {method:15s} {score:.6f}  ({gap_vs_oracle:+.1f}% vs Oracle)")

            # Check if learned loss beats naive and baselines
        print()
        if learned_avg < naive_avg:
        print(f"✓ Learned loss BEATS naive BCE ({(naive_avg/learned_avg - 1)*100:.1f}% improvement)")
        else:
        print(f"✗ Learned loss does NOT beat naive BCE ({(learned_avg/naive_avg - 1)*100:.1f}% worse)")

        if learned_avg < pudra_avg:
        print(f"✓ Learned loss BEATS PUDRa-naive ({(pudra_avg/learned_avg - 1)*100:.1f}% improvement)")
        else:
        print(f"✗ Learned loss does NOT beat PUDRa-naive ({(learned_avg/pudra_avg - 1)*100:.1f}% worse)")

        if learned_avg < vpu_avg:
        print(f"✓ Learned loss BEATS VPU-NoMixUp ({(vpu_avg/learned_avg - 1)*100:.1f}% improvement)")
        else:
        print(f"✗ Learned loss does NOT beat VPU-NoMixUp ({(learned_avg/vpu_avg - 1)*100:.1f}% worse)")

        print("="*70)
