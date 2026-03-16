"""Tests for adaptive meta-batch sizes based on dataset characteristics."""

import pytest
import yaml
import torch
from meta_learning.meta_trainer import MetaTrainer
from meta_learning.checkpoint_pool import CheckpointPool


def test_dataset_specific_batch_sizes():
    """Test different datasets use appropriate batch sizes."""
    config = {
        'meta_batch_size': 16,  # Default
        'batch_size': 128,
        'meta_lr': 1e-4,
        'adaptive_batch_sizes': {
            'mnist': 16,
            'fashionmnist': 16,
            'mushrooms': 48,
            'spambase': 48,
            'imdb': 8,
            '20news': 8,
        }
    }

    pool = CheckpointPool('./meta_checkpoints', lazy_load=True)
    pool.load_from_disk()

    trainer = MetaTrainer(config, pool, device='cpu')

    # Test getting batch size for each dataset
    assert trainer._get_meta_batch_size_for_dataset('mnist') == 16
    assert trainer._get_meta_batch_size_for_dataset('mushrooms') == 48
    assert trainer._get_meta_batch_size_for_dataset('imdb') == 8

    # Fallback to default for unknown dataset
    assert trainer._get_meta_batch_size_for_dataset('unknown') == 16


def test_batch_size_config_loading():
    """Test batch size config is loaded correctly from YAML."""
    # Load config file
    with open('config/methods/monotonic_basis_meta.yaml', 'r') as f:
        config_file = yaml.safe_load(f)

    # Extract the method config (nested under monotonic_basis_meta key)
    config = config_file.get('monotonic_basis_meta', config_file)

    # Verify adaptive_batch_sizes key exists
    assert 'adaptive_batch_sizes' in config, f"adaptive_batch_sizes not found in config keys: {list(config.keys())}"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_memory_safety_with_large_batches():
    """Test large batch sizes don't cause OOM."""
    import torch

    config = {
        'meta_batch_size': 48,  # Large batch for small models
        'batch_size': 256,
        'meta_lr': 1e-4,
        'K_inner_steps': 3,
    }

    pool = CheckpointPool('./meta_checkpoints', lazy_load=True)
    pool.load_from_disk()

    trainer = MetaTrainer(config, pool, device='mps')

    # Sample large batch of small models (mushrooms)
    mushroom_checkpoints = [cp for cp in pool._checkpoint_metadata if cp['dataset'] == 'mushrooms'][:48]
    meta_batch = [pool._load_checkpoint_state(cp) for cp in mushroom_checkpoints]

    # This should not crash with OOM
    try:
        improvements = trainer._vmapped_inner_loops(meta_batch, K=3)
        assert improvements.shape[0] == 48
    except RuntimeError as e:
        if "out of memory" in str(e):
            pytest.fail("OOM error with batch_size=48 for mushrooms (should be safe)")
        raise


def test_adaptive_sampling_distribution():
    """Test that adaptive sampling maintains dataset diversity."""
    config = {
        'meta_batch_size': 16,
        'batch_size': 128,
        'meta_lr': 1e-4,
        'adaptive_batch_sizes': {
            'mnist': 8,
            'mushrooms': 16,
            'imdb': 4,
        }
    }

    pool = CheckpointPool('./meta_checkpoints', lazy_load=True)
    pool.load_from_disk()

    trainer = MetaTrainer(config, pool, device='cpu')

    # Sample adaptive meta-batch
    meta_batch = trainer.sample_meta_batch_adaptive()

    # Should have mixed datasets according to adaptive sizes
    datasets = [cp['dataset'] for cp in meta_batch]
    unique_datasets = set(datasets)

    # Verify we have multiple datasets (diversity)
    assert len(unique_datasets) >= 2, "Adaptive sampling should include multiple datasets"
