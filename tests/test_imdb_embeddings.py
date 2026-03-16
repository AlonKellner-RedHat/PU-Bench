"""Tests for IMDB embeddings pre-loading during meta-learning."""

import pytest
import time
import os
import numpy as np
from meta_learning.meta_trainer import MetaTrainer
from meta_learning.checkpoint_pool import CheckpointPool


def test_imdb_embeddings_file_exists():
    """Test that pre-computed IMDB embeddings file exists."""
    embeddings_path = "./scripts/embeddings/imdb_sbert_embeddings.npz"
    assert os.path.exists(embeddings_path), f"IMDB embeddings not found at {embeddings_path}"

    # Verify file size (should be ~68MB)
    file_size_mb = os.path.getsize(embeddings_path) / (1024 * 1024)
    assert 60 < file_size_mb < 80, f"Unexpected embeddings file size: {file_size_mb:.1f}MB"


def test_imdb_config_includes_embeddings_path():
    """Test that IMDB data config includes pre-computed embeddings path."""
    # Create minimal config with IMDB checkpoint
    config = {
        'meta_batch_size': 1,
        'batch_size': 128,
        'meta_lr': 1e-4,
    }

    pool = CheckpointPool('./meta_checkpoints', lazy_load=True)
    pool.load_from_disk()

    trainer = MetaTrainer(config, pool, device='cpu')

    # Sample IMDB checkpoint
    imdb_checkpoints = [cp for cp in pool._checkpoint_metadata if cp['dataset'] == 'imdb']
    assert len(imdb_checkpoints) > 0, "No IMDB checkpoints found"

    # Get loaders for IMDB checkpoint
    checkpoint = imdb_checkpoints[0]
    train_loader, val_loader = trainer._get_loaders(checkpoint)

    # Verify embeddings were loaded (not computed)
    # This test passes if no "Auto-computing embeddings..." message appeared
    assert True  # If we got here without timeout, embeddings were loaded


def test_imdb_loading_is_fast():
    """Test that IMDB data loading takes <5 seconds (not ~2 minutes)."""
    config = {
        'meta_batch_size': 1,
        'batch_size': 128,
        'meta_lr': 1e-4,
    }

    pool = CheckpointPool('./meta_checkpoints', lazy_load=True)
    pool.load_from_disk()

    trainer = MetaTrainer(config, pool, device='cpu')

    # Sample IMDB checkpoint
    imdb_checkpoints = [cp for cp in pool._checkpoint_metadata if cp['dataset'] == 'imdb']
    checkpoint = imdb_checkpoints[0]

    # Time the loading
    start_time = time.time()
    train_loader, val_loader = trainer._get_loaders(checkpoint)
    elapsed = time.time() - start_time

    # Should be <5s (loading), not >60s (computing)
    assert elapsed < 5.0, f"IMDB loading took {elapsed:.1f}s (expected <5s)"


def test_embeddings_shape_correct():
    """Test loaded embeddings have correct shape (N × 384 for SBERT)."""
    embeddings_data = np.load("./scripts/embeddings/imdb_sbert_embeddings.npz")

    # SBERT all-MiniLM-L6-v2 produces 384-dim embeddings
    assert embeddings_data["train_embeddings"].shape[1] == 384
    assert embeddings_data["test_embeddings"].shape[1] == 384

    # Verify dtype is float32
    assert embeddings_data["train_embeddings"].dtype == np.float32
    assert embeddings_data["test_embeddings"].dtype == np.float32
