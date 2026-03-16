"""Test chunked integration with 100 points for improved BCE accuracy."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
from loss.loss_monotonic_basis import MonotonicBasisLoss


def compute_analytical_bce(logits, labels):
    """Compute analytical BCE for comparison."""
    pos_mask = labels == 1
    neg_mask = labels == 0
    p = torch.sigmoid(logits)
    p_pos = p[pos_mask]
    p_neg = p[neg_mask]

    # BCE for PU: -log(p_pos).mean() - log(1-p_neg).mean()
    bce = -(torch.log(p_pos + 1e-8).mean() + torch.log(1 - p_neg + 1e-8).mean())
    return bce


def test_100_points_chunked_achieves_low_error():
    """Test that 100 points with chunking achieves <0.5% error."""
    torch.manual_seed(42)
    logits = torch.randn(1000) * 2
    labels = (torch.rand(1000) < 0.3).float()
    pu_labels = labels.clone()
    pu_labels[labels == 0] = -1

    bce_true = compute_analytical_bce(logits, labels)

    # 100 points with chunking
    loss = MonotonicBasisLoss(
        num_repetitions=1,
        num_fourier=16,
        use_prior=True,
        prior=0.5,
        init_mode='bce_equivalent',
        num_integration_points=100,
        integration_chunk_size=256,
    )

    with torch.no_grad():
        monotonic_loss = loss(logits, pu_labels)

    error_pct = abs(monotonic_loss - bce_true) / bce_true * 100

    # Should be <0.5% with 100 points
    assert error_pct < 0.5, \
        f"Expected <0.5% error with 100 points, got {error_pct:.3f}%"

    print(f"\n100 points (chunked): {error_pct:.3f}% error")


def test_chunked_matches_unchunked():
    """Test that chunked integration produces same result as unchunked."""
    torch.manual_seed(42)
    logits = torch.randn(1000) * 2
    labels = (torch.rand(1000) < 0.3).float()
    pu_labels = labels.clone()
    pu_labels[labels == 0] = -1

    # Unchunked (chunk_size=None)
    loss_unchunked = MonotonicBasisLoss(
        num_repetitions=1,
        num_fourier=16,
        use_prior=True,
        prior=0.5,
        init_mode='bce_equivalent',
        num_integration_points=50,
        integration_chunk_size=None,
    )

    # Chunked (chunk_size=128)
    loss_chunked = MonotonicBasisLoss(
        num_repetitions=1,
        num_fourier=16,
        use_prior=True,
        prior=0.5,
        init_mode='bce_equivalent',
        num_integration_points=50,
        integration_chunk_size=128,
    )

    with torch.no_grad():
        result_unchunked = loss_unchunked(logits, pu_labels)
        result_chunked = loss_chunked(logits, pu_labels)

    # Should be identical (or within numerical precision)
    assert torch.allclose(result_unchunked, result_chunked, rtol=1e-5), \
        f"Chunked and unchunked results differ: {result_unchunked:.6f} vs {result_chunked:.6f}"

    print(f"\nUnchunked: {result_unchunked:.6f}")
    print(f"Chunked:   {result_chunked:.6f}")
    print("✓ Results match")


def test_gradients_flow_through_chunked_integration():
    """Test that gradients flow correctly through chunked integration."""
    torch.manual_seed(42)
    # Create a leaf tensor with requires_grad=True
    logits = torch.randn(100) * 2
    logits.requires_grad_(True)
    labels = (torch.rand(100) < 0.3).float()
    pu_labels = labels.clone()
    pu_labels[labels == 0] = -1

    # Chunked integration
    loss = MonotonicBasisLoss(
        num_repetitions=1,
        num_fourier=16,
        use_prior=True,
        prior=0.5,
        init_mode='bce_equivalent',
        num_integration_points=100,
        integration_chunk_size=32,
    )

    # Compute loss
    result = loss(logits, pu_labels)

    # Backward pass
    result.backward()

    # Verify gradients exist and are non-zero
    assert logits.grad is not None, "Gradients should flow to input"
    assert not torch.allclose(logits.grad, torch.zeros_like(logits.grad)), \
        "Gradients should be non-zero"

    print(f"\nLoss: {result.item():.6f}")
    print(f"Gradient norm: {logits.grad.norm().item():.6f}")
    print("✓ Gradients flow correctly")


def test_chunking_with_different_chunk_sizes():
    """Test that different chunk sizes produce consistent results."""
    torch.manual_seed(42)
    logits = torch.randn(500) * 2
    labels = (torch.rand(500) < 0.3).float()
    pu_labels = labels.clone()
    pu_labels[labels == 0] = -1

    chunk_sizes = [None, 64, 128, 256, 512]
    results = []

    for chunk_size in chunk_sizes:
        loss = MonotonicBasisLoss(
            num_repetitions=1,
            num_fourier=16,
            use_prior=True,
            prior=0.5,
            init_mode='bce_equivalent',
            num_integration_points=50,
            integration_chunk_size=chunk_size,
        )

        with torch.no_grad():
            result = loss(logits, pu_labels)
        results.append(result.item())

    # All results should be very close
    for i in range(len(results) - 1):
        rel_diff = abs(results[i] - results[i+1]) / (abs(results[i]) + 1e-8)
        assert rel_diff < 1e-5, \
            f"Chunk sizes {chunk_sizes[i]} and {chunk_sizes[i+1]} differ: {rel_diff:.6e}"

    print(f"\nChunk size consistency:")
    for chunk_size, result in zip(chunk_sizes, results):
        print(f"  {str(chunk_size):>4s}: {result:.6f}")
    print("✓ All chunk sizes produce consistent results")


def test_memory_efficiency_with_large_batch():
    """Test that chunking handles large batches without excessive memory."""
    torch.manual_seed(42)
    # Large batch size that would cause memory issues without chunking
    logits = torch.randn(5000) * 2
    labels = (torch.rand(5000) < 0.3).float()
    pu_labels = labels.clone()
    pu_labels[labels == 0] = -1

    # Should work with chunking
    loss = MonotonicBasisLoss(
        num_repetitions=1,
        num_fourier=16,
        use_prior=True,
        prior=0.5,
        init_mode='bce_equivalent',
        num_integration_points=100,
        integration_chunk_size=256,
    )

    try:
        with torch.no_grad():
            result = loss(logits, pu_labels)
        print(f"\nLarge batch (5000 samples): {result.item():.6f}")
        print("✓ Chunking handles large batches successfully")
    except RuntimeError as e:
        pytest.fail(f"Chunking failed with large batch: {e}")


if __name__ == '__main__':
    print("Testing chunked integration with 100 points...")
    test_100_points_chunked_achieves_low_error()
    test_chunked_matches_unchunked()
    test_gradients_flow_through_chunked_integration()
    test_chunking_with_different_chunk_sizes()
    test_memory_efficiency_with_large_batch()
    print("\n✓ All chunked integration tests passed!")
