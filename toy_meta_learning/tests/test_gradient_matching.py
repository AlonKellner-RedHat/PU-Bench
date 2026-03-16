#!/usr/bin/env python3
"""Tests for Gradient Matching utilities."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from models.simple_mlp import SimpleMLP
from loss.neural_pu_loss import NeuralPULoss
from tasks.gaussian_task import GaussianBlobTask
from tasks.gradient_matching_pool import GradientMatchingCheckpointPool, generate_random_task_config
from gradient_matching_utils import (
    compute_gradient_mse,
    advance_checkpoint_one_step,
    create_model_from_checkpoint,
    sample_batch_deterministic,
)


def test_gradient_mse_computation():
    """Test that gradient MSE computation works correctly."""
    print("Testing gradient MSE computation...")

    device = 'cpu'
    model = SimpleMLP(input_dim=2, hidden_dims=[32, 32]).to(device)
    learned_loss = NeuralPULoss(hidden_dim=64).to(device)

    # Create synthetic batch
    x = torch.randn(32, 2)
    y_pu = torch.tensor([1] * 16 + [-1] * 16)
    y_true = torch.bernoulli(torch.ones(32) * 0.5)

    # Get parameters
    params = {
        name: param.clone().detach().requires_grad_(True)
        for name, param in model.named_parameters()
    }

    # Compute gradient MSE
    grad_mse, diagnostics = compute_gradient_mse(
        model, params, x, y_pu, y_true, learned_loss, device
    )

    # Check outputs
    assert grad_mse.shape == ()  # Scalar
    assert grad_mse.requires_grad  # Should have computational graph
    assert not torch.isnan(grad_mse)
    assert not torch.isinf(grad_mse)

    print(f"  ✓ Gradient MSE: {grad_mse.item():.6f}")
    print(f"  ✓ Cosine similarity: {diagnostics['cosine_similarity']:.4f}")
    print(f"  ✓ PU grad norm: {diagnostics['pu_grad_norm']:.4f}")
    print(f"  ✓ BCE grad norm: {diagnostics['bce_grad_norm']:.4f}")

    # Test that we can backprop through gradient MSE
    grad_mse.backward()
    assert learned_loss.linear.weight.grad is not None
    print(f"  ✓ Meta-gradients flow to loss parameters")


def test_checkpoint_advancement():
    """Test advancing a checkpoint one step."""
    print("\nTesting checkpoint advancement...")

    device = 'cpu'
    learned_loss = NeuralPULoss(hidden_dim=64).to(device)

    # Create a checkpoint
    task_config = {
        'num_dimensions': 2,
        'mean_separation': 2.5,
        'std': 1.0,
        'prior': 0.5,
        'labeling_freq': 0.3,
        'num_samples': 1000,
        'mode': 'pu',
        'negative_labeling_freq': 0.3,
        'seed': 42,
    }

    model = SimpleMLP(input_dim=2, hidden_dims=[32, 32]).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    checkpoint = {
        'task_id': 'test_task',
        'task_config': task_config,
        'model_state': {k: v.cpu().clone() for k, v in model.state_dict().items()},
        'optimizer_state': {
            k: v.cpu().clone() if torch.is_tensor(v) else v
            for k, v in optimizer.state_dict().items()
        },
        'step_count': 0,
        'training_history': [],
        'last_updated_iteration': -1,
    }

    # Advance checkpoint
    updated_checkpoint = advance_checkpoint_one_step(
        checkpoint, learned_loss, device, batch_size=64
    )

    # Check updates
    assert updated_checkpoint['step_count'] == 1
    assert len(updated_checkpoint['training_history']) == 1
    assert updated_checkpoint['task_id'] == 'test_task'

    # Model state should be different
    old_weight = checkpoint['model_state']['network.0.weight']
    new_weight = updated_checkpoint['model_state']['network.0.weight']
    assert not torch.allclose(old_weight, new_weight)

    print(f"  ✓ Checkpoint advanced from step {checkpoint['step_count']} to {updated_checkpoint['step_count']}")
    print(f"  ✓ Training loss: {updated_checkpoint['training_history'][0]:.6f}")
    print(f"  ✓ Model parameters updated")


def test_checkpoint_pool():
    """Test checkpoint pool initialization and operations."""
    print("\nTesting checkpoint pool...")

    config = {
        'num_dimensions': 2,
        'mean_separations': [2.0, 2.5, 3.0],
        'stds': [0.8, 1.0],
        'labeling_freqs': [0.3],
        'priors': [0.5],
        'num_samples_per_task': 1000,
    }

    pool = GradientMatchingCheckpointPool(
        config=config,
        pool_size=20,  # Small pool for testing
        input_dim=2,
        hidden_dims=[32, 32],
        inner_lr=0.01,
        inner_momentum=0.9,
    )

    # Initialize pool
    pool.initialize_pool(device='cpu')
    assert len(pool) == 20
    print(f"  ✓ Pool initialized with {len(pool)} checkpoints")

    # Sample batch
    batch = pool.sample_batch(5)
    assert len(batch) == 5
    print(f"  ✓ Sampled batch of {len(batch)} checkpoints")

    # Check initial statistics
    stats = pool.get_statistics()
    assert stats['min_steps'] == 0
    assert stats['max_steps'] == 0
    assert stats['mean_steps'] == 0.0
    print(f"  ✓ Initial stats: all checkpoints at step 0")

    # Advance some checkpoints
    learned_loss = NeuralPULoss(hidden_dim=64).to('cpu')
    updated_checkpoints = []

    for checkpoint in batch:
        updated = advance_checkpoint_one_step(
            checkpoint, learned_loss, 'cpu', batch_size=64
        )
        updated_checkpoints.append(updated)

    # Update pool
    pool.update_pool(updated_checkpoints, persist_ratio=0.9, current_iteration=0, device='cpu')

    # Check updated statistics
    stats = pool.get_statistics()
    assert stats['max_steps'] == 1  # Some advanced to step 1
    print(f"  ✓ After update: step range [{stats['min_steps']}, {stats['max_steps']}]")

    # Test persistence/replacement
    initial_task_ids = {ckpt['task_id'] for ckpt in pool.checkpoints}

    # Run multiple iterations to see replacement
    for _ in range(5):
        batch = pool.sample_batch(5)
        updated_batch = []
        for ckpt in batch:
            updated = advance_checkpoint_one_step(ckpt, learned_loss, 'cpu', batch_size=64)
            updated_batch.append(updated)
        pool.update_pool(updated_batch, persist_ratio=0.9, current_iteration=1, device='cpu')

    final_task_ids = {ckpt['task_id'] for ckpt in pool.checkpoints}
    replaced_count = len(final_task_ids - initial_task_ids)

    print(f"  ✓ After 5 iterations: {replaced_count} checkpoints replaced")
    print(f"  ✓ Final step range: [{pool.get_statistics()['min_steps']}, {pool.get_statistics()['max_steps']}]")


def test_deterministic_batching():
    """Test that batching is deterministic based on step_count."""
    print("\nTesting deterministic batching...")

    task = GaussianBlobTask(
        num_dimensions=2,
        mean_separation=2.5,
        std=1.0,
        prior=0.5,
        labeling_freq=0.3,
        num_samples=1000,
        seed=42,
        mode='pu',
        negative_labeling_freq=0.3,
    )

    # Sample same batch twice
    x1, y_true1, y_pu1 = sample_batch_deterministic(task, step_count=5, batch_size=64, device='cpu')
    x2, y_true2, y_pu2 = sample_batch_deterministic(task, step_count=5, batch_size=64, device='cpu')

    # Should be identical
    assert torch.allclose(x1, x2)
    assert torch.equal(y_true1, y_true2)
    assert torch.equal(y_pu1, y_pu2)

    print(f"  ✓ Batches are deterministic for same step_count")

    # Different step_count should give different batch
    x3, _, _ = sample_batch_deterministic(task, step_count=10, batch_size=64, device='cpu')
    assert not torch.allclose(x1, x3)

    print(f"  ✓ Different step_count gives different batch")


if __name__ == '__main__':
    print("=" * 70)
    print("Testing Gradient Matching Components")
    print("=" * 70)
    print()

    test_gradient_mse_computation()
    test_checkpoint_advancement()
    test_checkpoint_pool()
    test_deterministic_batching()

    print()
    print("=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
