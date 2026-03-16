#!/usr/bin/env python3
"""Tests for Neural PU Loss."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
from loss.neural_pu_loss import NeuralPULoss


def test_forward_pass():
    """Test basic forward pass with known inputs."""
    loss_fn = NeuralPULoss(hidden_dim=100)
    outputs = torch.randn(32, 1)  # Logits
    labels = torch.tensor([1] * 16 + [-1] * 16)  # Half positive, half unlabeled

    loss = loss_fn(outputs, labels, mode='pu')

    assert loss.shape == ()  # Scalar
    assert loss.requires_grad
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    print(f"✓ Forward pass: loss = {loss.item():.4f}")


def test_edge_cases():
    """Test edge cases: empty batches, only positives, only unlabeled."""
    loss_fn = NeuralPULoss(hidden_dim=100)

    # Only positives
    outputs = torch.randn(16, 1)
    labels = torch.ones(16)
    loss = loss_fn(outputs, labels, mode='pu')
    assert not torch.isnan(loss)
    assert loss.item() != 0.0  # Should compute actual loss
    print(f"✓ Only positives: loss = {loss.item():.4f}")

    # Only unlabeled (should handle gracefully)
    labels = -torch.ones(16)
    loss = loss_fn(outputs, labels, mode='pu')
    assert not torch.isnan(loss)
    print(f"✓ Only unlabeled: loss = {loss.item():.4f}")

    # Empty batch
    outputs = torch.randn(0, 1)
    labels = torch.tensor([])
    loss = loss_fn(outputs, labels, mode='pu')
    assert loss.item() == 0.0
    assert loss.requires_grad
    print(f"✓ Empty batch: loss = {loss.item():.4f}")


def test_safe_log():
    """Test safe_log edge cases."""
    loss_fn = NeuralPULoss()

    # Zero input
    result = loss_fn.safe_log(torch.tensor(0.0))
    assert torch.isfinite(result)
    print(f"✓ safe_log(0.0) = {result.item():.4f} (finite)")

    # Very small input
    result = loss_fn.safe_log(torch.tensor(1e-10))
    assert torch.isfinite(result)
    print(f"✓ safe_log(1e-10) = {result.item():.4f} (finite)")

    # Normal input
    result = loss_fn.safe_log(torch.tensor(0.5))
    expected = torch.log(torch.tensor(0.5))
    assert torch.allclose(result, expected, atol=1e-6)
    print(f"✓ safe_log(0.5) = {result.item():.4f} (correct)")


def test_l05_regularization():
    """Test L0.5 regularization computation."""
    loss_fn = NeuralPULoss(hidden_dim=100, l05_lambda=0.01)

    # Run forward to populate activations
    outputs = torch.randn(32, 1)
    labels = torch.tensor([1] * 16 + [-1] * 16)
    loss = loss_fn(outputs, labels, mode='pu')

    # Compute regularization
    reg = loss_fn.compute_l05_regularization()
    assert reg > 0  # Should be positive
    assert torch.isfinite(reg)
    print(f"✓ L0.5 regularization: {reg.item():.4f} (positive and finite)")

    # Test with lambda=0 (no regularization)
    loss_fn_no_reg = NeuralPULoss(hidden_dim=100, l05_lambda=0.0)
    outputs = torch.randn(32, 1)
    labels = torch.tensor([1] * 16 + [-1] * 16)
    loss = loss_fn_no_reg(outputs, labels, mode='pu')
    reg_zero = loss_fn_no_reg.compute_l05_regularization()
    assert reg_zero.item() == 0.0
    print(f"✓ L0.5 with lambda=0: {reg_zero.item():.4f} (zero)")


def test_initialization_modes():
    """Test all initialization modes."""
    for mode in ['xavier_uniform', 'kaiming_normal', 'bce_equivalent', 'random_normal']:
        loss_fn = NeuralPULoss(init_mode=mode)
        assert loss_fn.linear.weight.shape == (64, 13)
        assert loss_fn.linear.bias.shape == (64,)
        print(f"✓ Initialization mode '{mode}': weights shape correct")


def test_hidden_dim_validation():
    """Test hidden_dim must be divisible by 4."""
    # Should raise ValueError
    with pytest.raises(ValueError, match="divisible by 4"):
        NeuralPULoss(hidden_dim=99)
    print("✓ hidden_dim=99 raises ValueError")

    with pytest.raises(ValueError, match="divisible by 4"):
        NeuralPULoss(hidden_dim=102)
    print("✓ hidden_dim=102 raises ValueError")

    # Should work
    NeuralPULoss(hidden_dim=100)
    print("✓ hidden_dim=100 works")

    NeuralPULoss(hidden_dim=200)
    print("✓ hidden_dim=200 works")

    NeuralPULoss(hidden_dim=4)
    print("✓ hidden_dim=4 works")


def test_gradient_flow():
    """Test that gradients flow through the loss."""
    loss_fn = NeuralPULoss(hidden_dim=100)

    # Create inputs with requires_grad
    outputs = torch.randn(32, 1, requires_grad=True)
    labels = torch.tensor([1] * 16 + [-1] * 16)

    # Forward pass
    loss = loss_fn(outputs, labels, mode='pu')

    # Backward pass
    loss.backward()

    # Check gradients
    assert outputs.grad is not None
    assert not torch.isnan(outputs.grad).any()
    assert loss_fn.linear.weight.grad is not None
    assert not torch.isnan(loss_fn.linear.weight.grad).any()

    print(f"✓ Gradients flow: output grad norm = {outputs.grad.norm().item():.4f}")
    print(f"✓ Gradients flow: weight grad norm = {loss_fn.linear.weight.grad.norm().item():.4f}")


def test_parameter_count():
    """Test get_num_parameters returns correct count."""
    # hidden_dim=64: weights=13*64, bias=64, total=896
    loss_fn = NeuralPULoss(hidden_dim=64)
    assert loss_fn.get_num_parameters() == 896
    print(f"✓ Parameter count (hidden_dim=64): {loss_fn.get_num_parameters()} (correct)")

    # hidden_dim=200: weights=13*200, bias=200, total=2800
    loss_fn = NeuralPULoss(hidden_dim=200)
    assert loss_fn.get_num_parameters() == 2800
    print(f"✓ Parameter count (hidden_dim=200): {loss_fn.get_num_parameters()} (correct)")


def test_repr():
    """Test string representation."""
    loss_fn = NeuralPULoss(hidden_dim=100, l05_lambda=0.01)
    repr_str = repr(loss_fn)

    assert 'NeuralPULoss' in repr_str
    assert 'hidden_dim=100' in repr_str
    assert 'num_parameters=1400' in repr_str  # 13*100 + 100 = 1400
    assert 'l05_lambda=0.01' in repr_str

    print(f"✓ __repr__ contains expected fields")
    print(f"Repr:\n{repr_str}")


def test_device_compatibility():
    """Test loss works on CPU and respects device."""
    loss_fn = NeuralPULoss(hidden_dim=100)

    # CPU
    outputs_cpu = torch.randn(32, 1)
    labels_cpu = torch.tensor([1] * 16 + [-1] * 16)
    loss_cpu = loss_fn(outputs_cpu, labels_cpu, mode='pu')
    assert loss_cpu.device.type == 'cpu'
    print(f"✓ CPU device: loss = {loss_cpu.item():.4f}")

    # MPS (if available)
    if torch.backends.mps.is_available():
        loss_fn_mps = loss_fn.to('mps')
        outputs_mps = torch.randn(32, 1, device='mps')
        labels_mps = torch.tensor([1] * 16 + [-1] * 16, device='mps')
        loss_mps = loss_fn_mps(outputs_mps, labels_mps, mode='pu')
        assert loss_mps.device.type == 'mps'
        print(f"✓ MPS device: loss = {loss_mps.item():.4f}")


def test_different_batch_sizes():
    """Test loss with various batch sizes."""
    loss_fn = NeuralPULoss(hidden_dim=100)

    for batch_size in [4, 16, 64, 128]:
        outputs = torch.randn(batch_size, 1)
        labels = torch.tensor([1] * (batch_size // 2) + [-1] * (batch_size // 2))
        loss = loss_fn(outputs, labels, mode='pu')

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        print(f"✓ Batch size {batch_size}: loss = {loss.item():.4f}")


if __name__ == '__main__':
    print("="*70)
    print("Testing NeuralPULoss")
    print("="*70)
    print()

    test_forward_pass()
    print()

    test_edge_cases()
    print()

    test_safe_log()
    print()

    test_l05_regularization()
    print()

    test_initialization_modes()
    print()

    test_hidden_dim_validation()
    print()

    test_gradient_flow()
    print()

    test_parameter_count()
    print()

    test_repr()
    print()

    test_device_compatibility()
    print()

    test_different_batch_sizes()
    print()

    print("="*70)
    print("All tests passed! ✓")
    print("="*70)
