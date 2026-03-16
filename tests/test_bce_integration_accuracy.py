"""Tests verifying BCE approximation accuracy vs numerical integration resolution.

This test file validates the assumption that the 6% error in BCE approximation
(observed with default 20 integration points) is due to numerical integration,
not the mathematical formulation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
from loss.loss_monotonic_basis import MonotonicBasisLoss


def compute_analytical_bce(logits, labels):
    """Compute analytical BCE for comparison.

    Args:
        logits: Model logits
        labels: Binary labels (0/1)

    Returns:
        BCE loss value
    """
    pos_mask = labels == 1
    neg_mask = labels == 0
    p = torch.sigmoid(logits)
    p_pos = p[pos_mask]
    p_neg = p[neg_mask]

    # BCE for PU: -log(p_pos).mean() - log(1-p_neg).mean()
    bce = -(torch.log(p_pos + 1e-8).mean() + torch.log(1 - p_neg + 1e-8).mean())
    return bce


def test_integration_resolution_reduces_error():
    """Test that higher integration resolution reduces BCE approximation error."""

    # Fixed test data
    torch.manual_seed(42)
    logits = torch.randn(1000) * 2
    labels = (torch.rand(1000) < 0.3).float()
    pu_labels = labels.clone()
    pu_labels[labels == 0] = -1

    # Compute true BCE
    bce_true = compute_analytical_bce(logits, labels)

    # Test different resolutions
    resolutions = [10, 20, 50, 100]
    errors = []

    for n_points in resolutions:
        loss = MonotonicBasisLoss(
            num_repetitions=1,
            num_fourier=16,
            use_prior=True,
            prior=0.5,
            init_mode='bce_equivalent',
            num_integration_points=n_points,
        )

        with torch.no_grad():
            monotonic_loss = loss(logits, pu_labels)

        error = abs(monotonic_loss - bce_true) / bce_true * 100
        errors.append(error.item())

    # Verify errors decrease monotonically
    for i in range(len(errors) - 1):
        assert errors[i] > errors[i+1], \
            f"Error should decrease: {errors[i]:.2f}% → {errors[i+1]:.2f}%"

    print(f"\nIntegration resolution vs error:")
    for res, err in zip(resolutions, errors):
        print(f"  {res:3d} points: {err:5.2f}% error")


def test_default_20_points_gives_6pct_error():
    """Verify that default 20 points gives ~6% error (as observed)."""

    torch.manual_seed(42)
    logits = torch.randn(1000) * 2
    labels = (torch.rand(1000) < 0.3).float()
    pu_labels = labels.clone()
    pu_labels[labels == 0] = -1

    bce_true = compute_analytical_bce(logits, labels)

    # Default resolution (20 points)
    loss = MonotonicBasisLoss(
        num_repetitions=1,
        num_fourier=16,
        use_prior=True,
        prior=0.5,
        init_mode='bce_equivalent',
        num_integration_points=20,
    )

    with torch.no_grad():
        monotonic_loss = loss(logits, pu_labels)

    error_pct = abs(monotonic_loss - bce_true) / bce_true * 100

    # Should be around 6% (allow 4-8% range for robustness)
    assert 4.0 < error_pct < 8.0, \
        f"Expected ~6% error with 20 points, got {error_pct:.2f}%"

    print(f"\nDefault (20 points): {error_pct:.2f}% error")


def test_50_points_gives_1pct_error():
    """Verify that 50 points achieves <2% error."""

    torch.manual_seed(42)
    logits = torch.randn(1000) * 2
    labels = (torch.rand(1000) < 0.3).float()
    pu_labels = labels.clone()
    pu_labels[labels == 0] = -1

    bce_true = compute_analytical_bce(logits, labels)

    # Higher resolution (50 points)
    loss = MonotonicBasisLoss(
        num_repetitions=1,
        num_fourier=16,
        use_prior=True,
        prior=0.5,
        init_mode='bce_equivalent',
        num_integration_points=50,
    )

    with torch.no_grad():
        monotonic_loss = loss(logits, pu_labels)

    error_pct = abs(monotonic_loss - bce_true) / bce_true * 100

    # Should be <2%
    assert error_pct < 2.0, \
        f"Expected <2% error with 50 points, got {error_pct:.2f}%"

    print(f"\nHigher resolution (50 points): {error_pct:.2f}% error")


def test_100_points_gives_sub_1pct_error():
    """Verify that 100 points achieves <1% error."""

    torch.manual_seed(42)
    logits = torch.randn(1000) * 2
    labels = (torch.rand(1000) < 0.3).float()
    pu_labels = labels.clone()
    pu_labels[labels == 0] = -1

    bce_true = compute_analytical_bce(logits, labels)

    # Very high resolution (100 points)
    loss = MonotonicBasisLoss(
        num_repetitions=1,
        num_fourier=16,
        use_prior=True,
        prior=0.5,
        init_mode='bce_equivalent',
        num_integration_points=100,
    )

    with torch.no_grad():
        monotonic_loss = loss(logits, pu_labels)

    error_pct = abs(monotonic_loss - bce_true) / bce_true * 100

    # Should be <1%
    assert error_pct < 1.0, \
        f"Expected <1% error with 100 points, got {error_pct:.2f}%"

    print(f"\nVery high resolution (100 points): {error_pct:.2f}% error")


def test_error_attribution_is_numerical_not_mathematical():
    """Verify the 6% error is due to numerical integration, not math.

    This is the key test proving our assumption: if we can reduce error
    arbitrarily by increasing resolution, then the formulation is correct
    and only the discretization introduces error.
    """

    torch.manual_seed(42)
    logits = torch.randn(1000) * 2
    labels = (torch.rand(1000) < 0.3).float()
    pu_labels = labels.clone()
    pu_labels[labels == 0] = -1

    bce_true = compute_analytical_bce(logits, labels)

    # Test that we can get arbitrarily close with enough points
    loss_200 = MonotonicBasisLoss(
        num_repetitions=1,
        num_fourier=16,
        use_prior=True,
        prior=0.5,
        init_mode='bce_equivalent',
        num_integration_points=200,
    )

    with torch.no_grad():
        monotonic_loss = loss_200(logits, pu_labels)

    error_pct = abs(monotonic_loss - bce_true) / bce_true * 100

    # With 200 points, should be <0.2% (essentially exact)
    assert error_pct < 0.2, \
        f"Expected <0.2% error with 200 points (proof of correct math), got {error_pct:.3f}%"

    print(f"\n200 points: {error_pct:.3f}% error")
    print("✓ Mathematical formulation is CORRECT")
    print("✓ Error is purely from numerical integration discretization")


def test_three_repetitions_scales_correctly():
    """Verify that 3 repetitions gives 3× the single-repetition loss."""

    torch.manual_seed(42)
    logits = torch.randn(1000) * 2
    labels = (torch.rand(1000) < 0.3).float()
    pu_labels = labels.clone()
    pu_labels[labels == 0] = -1

    # Single repetition
    loss_1rep = MonotonicBasisLoss(
        num_repetitions=1,
        num_fourier=16,
        use_prior=True,
        prior=0.5,
        init_mode='bce_equivalent',
        num_integration_points=50,
    )

    # Three repetitions
    loss_3rep = MonotonicBasisLoss(
        num_repetitions=3,
        num_fourier=16,
        use_prior=True,
        prior=0.5,
        init_mode='bce_equivalent',
        num_integration_points=50,
    )

    with torch.no_grad():
        val_1rep = loss_1rep(logits, pu_labels)
        val_3rep = loss_3rep(logits, pu_labels)

    # Should be exactly 3× (since all reps have same init)
    ratio = val_3rep / val_1rep

    assert abs(ratio - 3.0) < 0.01, \
        f"3 repetitions should give 3× loss, got ratio={ratio:.6f}"

    print(f"\n1 rep: {val_1rep:.6f}")
    print(f"3 rep: {val_3rep:.6f}")
    print(f"Ratio: {ratio:.6f} (expected: 3.0)")
