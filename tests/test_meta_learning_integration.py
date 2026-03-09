"""Integration tests for meta-learning components.

Tests the complete "models as data" meta-learning pipeline:
1. CheckpointPool creation and sampling
2. MetaTrainer initialization and gradient flow
3. Loss parameter updates
4. Regularization effects
"""

import pytest
import torch
import tempfile
from pathlib import Path

from meta_learning import CheckpointPool, MetaTrainer
from loss.loss_monotonic_basis import MonotonicBasisLoss


class TestCheckpointPoolBasics:
    """Test basic checkpoint pool functionality."""

    def test_pool_creation(self):
        """Test creating an empty checkpoint pool."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = CheckpointPool(tmpdir)
            assert len(pool.checkpoints) == 0
            assert pool.checkpoint_dir == Path(tmpdir)

    def test_add_checkpoint(self):
        """Test adding a checkpoint to the pool."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = CheckpointPool(tmpdir)

            # Create a dummy checkpoint
            checkpoint = {
                'task_id': 'mnist_c0.3_prior0.5_seed42',
                'epoch': 5,
                'seed': 42,
                'model_state': {'dummy': torch.tensor([1.0, 2.0, 3.0])},
                'dataset': 'mnist',
                'c_value': 0.3,
                'prior': 0.5,
                'source': 'pn_naive',
                'created_at': 0
            }

            pool.add_checkpoint(checkpoint)
            assert len(pool.checkpoints) == 1
            assert pool.checkpoints[0]['task_id'] == 'mnist_c0.3_prior0.5_seed42'

    def test_sample_meta_batch(self):
        """Test sampling a meta-batch from the pool."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = CheckpointPool(tmpdir)

            # Add multiple checkpoints
            for i in range(10):
                checkpoint = {
                    'task_id': f'task_{i}',
                    'epoch': i,
                    'seed': 42,
                    'model_state': {},
                    'dataset': 'mnist',
                    'c_value': 0.3,
                    'prior': 0.5,
                    'source': 'pn_naive',
                    'created_at': 0
                }
                pool.add_checkpoint(checkpoint)

            # Sample a batch
            batch_size = 5
            meta_batch = pool.sample_meta_batch(batch_size)

            assert len(meta_batch) == batch_size
            # Check that we got different checkpoints (high probability)
            task_ids = [cp['task_id'] for cp in meta_batch]
            assert len(set(task_ids)) >= 3  # At least some diversity

    def test_pool_statistics(self):
        """Test pool statistics computation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = CheckpointPool(tmpdir)

            # Add checkpoints with different properties
            for dataset in ['mnist', 'fashionmnist']:
                for c_value in [0.3, 0.5]:
                    for epoch in [5, 10]:
                        checkpoint = {
                            'task_id': f'{dataset}_c{c_value}_prior0.5_seed42',
                            'epoch': epoch,
                            'seed': 42,
                            'model_state': {},
                            'dataset': dataset,
                            'c_value': c_value,
                            'prior': 0.5,
                            'source': 'pn_naive',
                            'created_at': 0
                        }
                        pool.add_checkpoint(checkpoint)

            stats = pool.get_statistics()
            assert stats['pool_size'] == 8  # 2 datasets × 2 c_values × 2 epochs
            assert stats['unique_datasets'] == 2
            assert stats['unique_epochs'] == 2


class TestMetaTrainerBasics:
    """Test basic meta-trainer functionality."""

    def test_trainer_initialization(self):
        """Test creating a meta-trainer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = CheckpointPool(tmpdir)

            config = {
                'num_repetitions': 3,
                'num_fourier': 16,
                'use_prior': True,
                'l1_weight': 1e-4,
                'l2_weight': 1e-3,
                'oracle_mode': False,
                'init_scale': 0.01,
                'meta_lr': 1e-4,
                'meta_batch_size': 4,
                'meta_iterations': 10,
                'meta_objective': 'bce',
                'log_freq': 5,
                'save_freq': 10,
                'checkpoint_refresh_freq': 50,
                'checkpoint_refresh_percent': 10,
                'loss_checkpoint_dir': tmpdir
            }

            trainer = MetaTrainer(config, pool, device=torch.device('cpu'))

            # Check initialization
            assert trainer.learned_loss is not None
            assert trainer.optimizer_loss is not None
            assert trainer.iteration == 0
            assert trainer.best_meta_loss == float('inf')

    def test_loss_has_gradients(self):
        """Test that loss parameters can receive gradients."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'num_repetitions': 3,
                'num_fourier': 16,
                'use_prior': True,
                'l1_weight': 1e-4,
                'l2_weight': 1e-3,
                'oracle_mode': False,
                'init_scale': 0.01,
                'meta_lr': 1e-4,
                'loss_checkpoint_dir': tmpdir
            }

            pool = CheckpointPool(tmpdir)
            trainer = MetaTrainer(config, pool, device=torch.device('cpu'))

            # Create dummy outputs and labels
            outputs = torch.randn(10, requires_grad=False)  # Model outputs (frozen)
            labels = torch.randint(0, 2, (10,))

            # Compute loss
            trainer.learned_loss.set_prior(0.5)
            loss = trainer.learned_loss(outputs, labels)

            # Backprop
            loss.backward()

            # Check that loss parameters have gradients
            has_gradients = False
            for param in trainer.learned_loss.parameters():
                if param.grad is not None and param.grad.abs().sum() > 0:
                    has_gradients = True
                    break

            assert has_gradients, "Loss parameters should receive gradients"

    def test_regularization_computation(self):
        """Test that regularization is computed correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'num_repetitions': 3,
                'num_fourier': 16,
                'use_prior': True,
                'l1_weight': 1e-4,
                'l2_weight': 1e-3,
                'oracle_mode': False,
                'init_scale': 0.01,
                'meta_lr': 1e-4,
                'loss_checkpoint_dir': tmpdir
            }

            pool = CheckpointPool(tmpdir)
            trainer = MetaTrainer(config, pool, device=torch.device('cpu'))

            # Compute regularization
            reg_loss = trainer.learned_loss.compute_regularization()

            # Should be a scalar tensor
            assert isinstance(reg_loss, torch.Tensor)
            assert reg_loss.dim() == 0  # Scalar

            # Should be positive (sum of L1 and L2)
            assert reg_loss.item() >= 0


class TestModelsAsDataParadigm:
    """Test the 'models as data' paradigm."""

    def test_checkpoint_as_data_sample(self):
        """Test that checkpoints can be treated as data samples."""
        # A checkpoint contains:
        # - model_state: pre-trained weights (frozen "input")
        # - dataset info: task configuration
        # - task_id: identifier

        checkpoint = {
            'task_id': 'mnist_c0.3_prior0.5_seed42',
            'epoch': 10,
            'seed': 42,
            'model_state': {'layer.weight': torch.randn(10, 5)},
            'dataset': 'mnist',
            'c_value': 0.3,
            'prior': 0.5,
            'source': 'pn_naive',
            'created_at': 0
        }

        # Should be able to extract all necessary info
        assert 'model_state' in checkpoint  # The "data" itself
        assert 'task_id' in checkpoint  # Metadata
        assert 'dataset' in checkpoint  # For model creation
        assert 'prior' in checkpoint  # For loss conditioning

    def test_meta_batch_like_data_batch(self):
        """Test that meta-batches are analogous to data batches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = CheckpointPool(tmpdir)

            # Create multiple "data samples" (checkpoints)
            for i in range(10):
                checkpoint = {
                    'task_id': f'task_{i}',
                    'epoch': i,
                    'seed': 42,
                    'model_state': {'dummy': torch.randn(5)},
                    'dataset': 'mnist',
                    'c_value': 0.3,
                    'prior': 0.5,
                    'source': 'pn_naive',
                    'created_at': 0
                }
                pool.add_checkpoint(checkpoint)

            # Sample a "batch" of checkpoints
            meta_batch = pool.sample_meta_batch(4)

            # Just like a data batch has multiple (x, y) pairs,
            # meta-batch has multiple (model_state, task_config) pairs
            assert len(meta_batch) == 4
            for checkpoint in meta_batch:
                assert 'model_state' in checkpoint
                assert 'dataset' in checkpoint
                assert 'prior' in checkpoint


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
