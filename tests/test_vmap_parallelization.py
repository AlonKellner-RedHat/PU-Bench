"""Tests for vmap parallelization of K=3 inner loops."""

import pytest
import time
import torch
from meta_learning.meta_trainer import MetaTrainer
from meta_learning.checkpoint_pool import CheckpointPool


def test_checkpoint_grouping_by_dataset():
    """Test checkpoints are correctly grouped by dataset."""
    config = {
        'meta_batch_size': 16,
        'batch_size': 128,
        'meta_lr': 1e-4,
    }

    pool = CheckpointPool('./meta_checkpoints', lazy_load=True)
    pool.load_from_disk()

    trainer = MetaTrainer(config, pool, device='cpu')

    # Sample meta-batch with mixed datasets
    meta_batch = pool.sample_meta_batch(16)

    # Group by dataset
    groups = trainer._group_by_dataset(meta_batch)

    # Verify grouping
    assert isinstance(groups, dict)
    assert len(groups) > 0

    # Verify all checkpoints in each group have same dataset
    for dataset, checkpoints in groups.items():
        for cp in checkpoints:
            cp_dataset = cp.get('dataset', pool.get_dataset_from_task_id(cp['task_id']))
            assert cp_dataset == dataset


def test_vmap_vs_sequential_equivalence():
    """Test vmap produces identical results to sequential processing."""
    config = {
        'meta_batch_size': 4,
        'batch_size': 64,
        'meta_lr': 1e-4,
        'K_inner_steps': 3,
        'inner_lr': 1e-3,
        'use_vmap': True,
    }

    pool = CheckpointPool('./meta_checkpoints', lazy_load=True)
    pool.load_from_disk()

    trainer = MetaTrainer(config, pool, device='cpu')

    # Sample checkpoints from single dataset (to ensure compatibility)
    mnist_checkpoints = [cp for cp in pool._checkpoint_metadata if cp['dataset'] == 'mnist'][:4]
    meta_batch = [pool._load_checkpoint_state(cp) for cp in mnist_checkpoints]

    # Process with vmap (new implementation)
    torch.manual_seed(42)
    improvements_vmap = trainer._vmapped_inner_loops(meta_batch, K=3)

    # Process sequentially (old implementation)
    torch.manual_seed(42)
    improvements_seq = trainer._vmapped_inner_loops_sequential(meta_batch, K=3)

    # Results should be identical (or very close due to floating point)
    assert torch.allclose(improvements_vmap, improvements_seq, rtol=1e-4, atol=1e-5)


def test_gradient_flow_through_vmap():
    """Test gradients flow correctly through vmapped K=3 steps to loss parameters."""
    config = {
        'meta_batch_size': 2,
        'batch_size': 32,
        'meta_lr': 1e-4,
        'K_inner_steps': 3,
        'use_vmap': True,
    }

    pool = CheckpointPool('./meta_checkpoints', lazy_load=True)
    pool.load_from_disk()

    trainer = MetaTrainer(config, pool, device='cpu')

    # Sample checkpoints
    mnist_checkpoints = [cp for cp in pool._checkpoint_metadata if cp['dataset'] == 'mnist'][:2]
    meta_batch = [pool._load_checkpoint_state(cp) for cp in mnist_checkpoints]

    # Get initial loss parameter values
    initial_params = {name: param.clone() for name, param in trainer.learned_loss.named_parameters()}

    # Run one meta-training step
    trainer.optimizer_loss.zero_grad()
    improvements = trainer._vmapped_inner_loops(meta_batch, K=3)
    loss = -improvements.sum()  # Maximize improvement = minimize negative
    loss.backward()

    # Verify gradients exist for loss parameters
    for name, param in trainer.learned_loss.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
        assert not torch.isinf(param.grad).any(), f"Inf gradient for {name}"

    # Apply gradients
    trainer.optimizer_loss.step()

    # Verify parameters changed
    for name, param in trainer.learned_loss.named_parameters():
        assert not torch.allclose(param, initial_params[name]), f"Parameter {name} didn't change"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_vmap_performance_improvement():
    """Test vmap is faster than sequential processing."""
    config = {
        'meta_batch_size': 8,
        'batch_size': 128,
        'meta_lr': 1e-4,
        'K_inner_steps': 3,
    }

    pool = CheckpointPool('./meta_checkpoints', lazy_load=True)
    pool.load_from_disk()

    # Sample checkpoints from single dataset
    mnist_checkpoints = [cp for cp in pool._checkpoint_metadata if cp['dataset'] == 'mnist'][:8]
    meta_batch = [pool._load_checkpoint_state(cp) for cp in mnist_checkpoints]

    # Benchmark sequential
    config['use_vmap'] = False
    trainer_seq = MetaTrainer(config, pool, device='mps')
    start = time.time()
    _ = trainer_seq._vmapped_inner_loops(meta_batch, K=3)
    sequential_time = time.time() - start

    # Benchmark vmap
    config['use_vmap'] = True
    trainer_vmap = MetaTrainer(config, pool, device='mps')
    start = time.time()
    _ = trainer_vmap._vmapped_inner_loops(meta_batch, K=3)
    vmap_time = time.time() - start

    # Vmap should be at least 1.5x faster
    speedup = sequential_time / vmap_time
    assert speedup > 1.5, f"Vmap speedup only {speedup:.2f}x (expected >1.5x)"
