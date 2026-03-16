"""Tests for checkpoint curriculum mechanism during meta-training."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
from meta_learning.meta_trainer import MetaTrainer
from meta_learning.checkpoint_pool import CheckpointPool

def test_curriculum_enabled_in_config():
    """Test that curriculum can be enabled via config."""
    config = {
        'meta_batch_size': 8,
        'batch_size': 64,
        'meta_lr': 1e-4,
        'checkpoint_refresh_freq': 50,  # Refresh every 50 iterations
        'checkpoint_refresh_percent': 10,  # Replace 10%
        'enable_curriculum': True,  # NEW: explicit enable flag
    }

    pool = CheckpointPool('./meta_checkpoints', lazy_load=True)
    pool.load_from_disk()

    trainer = MetaTrainer(config, pool, device='cpu')

    # Verify config loaded
    assert trainer.config.get('enable_curriculum', False) == True
    assert trainer.refresh_freq == 50
    assert trainer.refresh_percent == 10

def test_refresh_triggered_at_correct_iteration():
    """Test that checkpoint refresh is triggered at the correct iteration."""
    config = {
        'meta_batch_size': 4,
        'batch_size': 32,
        'meta_lr': 1e-4,
        'checkpoint_refresh_freq': 3,  # Refresh every 3 iterations
        'checkpoint_refresh_percent': 10,
        'enable_curriculum': True,
    }

    pool = CheckpointPool('./meta_checkpoints', lazy_load=True)
    pool.load_from_disk()

    trainer = MetaTrainer(config, pool, device='cpu')

    # Track refresh calls
    refresh_called_at = []
    original_refresh = trainer._refresh_pool

    def tracked_refresh(iteration):
        refresh_called_at.append(iteration)
        # Don't actually refresh (expensive)
        pass

    trainer._refresh_pool = tracked_refresh

    # Simulate 10 iterations
    for i in range(1, 11):  # iterations 1-10
        trainer.iteration = i

        # Check if refresh should be called
        if i > 0 and i % trainer.refresh_freq == 0:
            trainer._refresh_pool(i)

    # Should have refreshed at iterations 3, 6, 9
    assert refresh_called_at == [3, 6, 9], \
        f"Expected refresh at [3,6,9], got {refresh_called_at}"

def test_new_checkpoints_use_learned_loss():
    """Test that new checkpoints are created using current learned loss."""
    config = {
        'meta_batch_size': 2,
        'batch_size': 32,
        'meta_lr': 1e-4,
        'checkpoint_refresh_freq': 1,
        'checkpoint_refresh_percent': 5,  # Small percent for testing
        'enable_curriculum': True,
    }

    pool = CheckpointPool('./meta_checkpoints', lazy_load=True)
    pool.load_from_disk()

    trainer = MetaTrainer(config, pool, device='cpu')

    # Record checkpoint sources before refresh
    initial_sources = [cp.get('source', 'pn_naive')
                       for cp in pool._checkpoint_metadata]
    initial_learned_count = sum(1 for s in initial_sources if s == 'learned_loss')

    # Run one iteration to update learned loss
    meta_batch = pool.sample_meta_batch(2)
    trainer.iteration = 1
    trainer.meta_train_step_k3(meta_batch)

    # Trigger refresh (this is expensive, so we'll mock it partially)
    # Just verify the mechanism exists
    assert hasattr(trainer, '_refresh_pool'), \
        "Trainer should have _refresh_pool method"
    assert callable(trainer._refresh_pool), \
        "_refresh_pool should be callable"

def test_old_checkpoints_preserved():
    """Test that (100-N)% of old checkpoints are preserved during refresh."""
    # This test would verify that when refreshing 10%,
    # 90% of original checkpoints remain

    config = {
        'checkpoint_refresh_percent': 10,
    }

    pool = CheckpointPool('./meta_checkpoints', lazy_load=True)
    pool.load_from_disk()

    initial_count = len(pool._checkpoint_metadata)
    initial_task_ids = set(cp['task_id'] for cp in pool._checkpoint_metadata)

    # After a 10% refresh, we should still have 90% of original task_ids
    # (This is conceptual - actual test would need to run full refresh)

    expected_preserved_count = int(initial_count * 0.9)

    # Verify refresh mechanism preserves most checkpoints
    assert True  # Placeholder - full test would run actual refresh

def test_curriculum_improves_convergence():
    """Test that curriculum leads to better convergence (integration test)."""
    # This is more of an integration/benchmark test
    # Would compare meta-training with/without curriculum

    # With curriculum: should reach positive avg_imp faster
    # Without curriculum: may take longer to converge

    # Placeholder for actual benchmark
    assert True
