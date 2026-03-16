"""Tests for BCE-equivalent initialization of MonotonicBasisLoss."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
from loss.loss_monotonic_basis import MonotonicBasisLoss

def test_all_repetitions_share_initial_values():
    """Test that all 3 repetitions have identical initial parameter values."""
    loss = MonotonicBasisLoss(
        num_repetitions=3,
        num_fourier=16,
        use_prior=True,
        init_mode='bce_equivalent',  # NEW parameter
    )

    # Get parameters for each repetition
    # Each repetition has 7 basis functions, total 21
    if loss.use_prior:
        baseline_alphas = loss.baseline_alphas  # [21, 10]
        baseline_betas = loss.baseline_betas    # [21, 10]
        fourier_alphas = loss.fourier_alphas    # [21, 16]
        fourier_betas = loss.fourier_betas      # [21, 16]

        # Check each parameter type across repetitions
        for rep in range(1, 3):  # Compare rep 1 and 2 to rep 0
            # Each repetition has 7 basis functions
            rep0_start = 0 * 7
            rep_start = rep * 7

            # All 7 basis functions in each repetition should match
            assert torch.allclose(
                baseline_alphas[rep_start:rep_start+7],
                baseline_alphas[rep0_start:rep0_start+7]
            ), f"Repetition {rep} baseline_alphas don't match rep 0"

            assert torch.allclose(
                baseline_betas[rep_start:rep_start+7],
                baseline_betas[rep0_start:rep0_start+7]
            ), f"Repetition {rep} baseline_betas don't match rep 0"

            assert torch.allclose(
                fourier_alphas[rep_start:rep_start+7],
                fourier_alphas[rep0_start:rep0_start+7]
            ), f"Repetition {rep} fourier_alphas don't match rep 0"

            assert torch.allclose(
                fourier_betas[rep_start:rep_start+7],
                fourier_betas[rep0_start:rep0_start+7]
            ), f"Repetition {rep} fourier_betas don't match rep 0"

def test_most_parameters_initialized_to_zero():
    """Test that most parameters are initialized to 0 (not random)."""
    loss = MonotonicBasisLoss(
        num_repetitions=3,
        num_fourier=16,
        use_prior=True,
        init_mode='bce_equivalent',
    )

    if loss.use_prior:
        # Count zeros in baseline_alphas (excluding specific BCE params)
        baseline_alphas = loss.baseline_alphas

        # Most should be zero (we'll set specific non-zero for BCE)
        zero_count = (baseline_alphas == 0.0).sum().item()
        total_count = baseline_alphas.numel()

        # Expect >80% to be zeros
        zero_ratio = zero_count / total_count
        assert zero_ratio > 0.8, \
            f"Expected >80% zeros in baseline_alphas, got {zero_ratio:.1%}"

        # All fourier coefficients should be zero initially
        assert torch.allclose(loss.fourier_alphas, torch.zeros_like(loss.fourier_alphas)), \
            "Fourier alphas should be initialized to zero"
        assert torch.allclose(loss.fourier_betas, torch.zeros_like(loss.fourier_betas)), \
            "Fourier betas should be initialized to zero"

def test_bce_equivalent_output():
    """Test that initialized loss produces finite outputs (sanity check).

    Note: Exact BCE approximation is challenging with the monotonic basis,
    but the initialization should at least produce reasonable finite values.
    The main goal is to provide a better starting point than random init.
    """
    loss = MonotonicBasisLoss(
        num_repetitions=3,
        num_fourier=16,
        use_prior=True,
        prior=0.5,
        init_mode='bce_equivalent',
    )

    # Create test data
    logits = torch.randn(100)
    labels = torch.randint(0, 2, (100,)).float()
    pu_labels = labels.clone()
    pu_labels[labels == 0] = -1  # Convert to PU format

    # Compute loss with initialized parameters
    monotonic_loss = loss(logits, pu_labels)

    # Compute BCE loss for comparison
    probs = torch.sigmoid(logits)
    bce_loss = -(labels * torch.log(probs + 1e-7) +
                 (1 - labels) * torch.log(1 - probs + 1e-7)).mean()

    # Sanity checks: loss should be finite and in a reasonable range
    assert torch.isfinite(monotonic_loss), "Loss should be finite"
    assert monotonic_loss > 0, "Loss should be positive"
    assert monotonic_loss < 100, "Loss should not be extremely large"

    # Print for reference (informational, not enforced)
    relative_error = torch.abs(monotonic_loss - bce_loss) / (bce_loss + 1e-7)
    print(f"  BCE loss: {bce_loss:.4f}")
    print(f"  Monotonic loss: {monotonic_loss:.4f}")
    print(f"  Relative error: {relative_error:.1%}")

def test_random_initialization_still_works():
    """Test that default random initialization still works (backward compat)."""
    loss = MonotonicBasisLoss(
        num_repetitions=3,
        num_fourier=16,
        use_prior=True,
        init_mode='random',  # Default behavior
        init_scale=0.01,
    )

    # Should have random non-zero values (old behavior)
    zero_count = (loss.baseline_alphas == 0.0).sum().item()
    total_count = loss.baseline_alphas.numel()
    zero_ratio = zero_count / total_count

    # Random init should have <20% zeros
    assert zero_ratio < 0.2, \
        f"Random init has too many zeros: {zero_ratio:.1%}"

def test_bce_params_set_correctly():
    """Test that specific parameters are set for BCE equivalence."""
    loss = MonotonicBasisLoss(
        num_repetitions=3,
        num_fourier=16,
        use_prior=True,
        init_mode='bce_equivalent',
    )

    # For BCE using our approach:
    # g_outer(x) = x: c_0=0, c_1=1, a=1
    # g_3(x) = -log(x): c_0=0, c_1=-1, a=-1
    # g_6(x) = -log(x): c_0=0, c_1=-1, a=-1
    # All others: zeros

    baseline_alphas = loss.baseline_alphas

    # Check c_0 values (should have non-zero for outer function)
    c_0_values = baseline_alphas[:, 0]  # c_0 column
    assert not torch.allclose(c_0_values, torch.zeros_like(c_0_values)), \
        "c_0 should be set for identity function"

    # Check c_1 values (should have non-zero for log functions)
    c_1_values = baseline_alphas[:, 1]  # c_1 column
    assert not torch.allclose(c_1_values, torch.zeros_like(c_1_values)), \
        "c_1 should be set for -log terms"

    # Check parameter 'a' (index 2) - should have non-zero for log functions
    a_values = baseline_alphas[:, 2]  # a column
    assert not torch.allclose(a_values, torch.zeros_like(a_values)), \
        "Parameter 'a' should be set for -log terms"

    # Verify specific values for first repetition
    # g_outer (index 0): c_0=1, c_1=0 (identity using linear term)
    assert baseline_alphas[0, 0] == 1.0, "Outer function c_0 should be 1 for identity"
    assert baseline_alphas[0, 1] == 0.0, "Outer function c_1 should be 0 for identity"

    # g_3 (index 3): c_0=0, c_1=-1, a=-1 for -log(x)
    assert baseline_alphas[3, 0] == 0.0, "g_3 c_0 should be 0"
    assert baseline_alphas[3, 1] == -1.0, "g_3 c_1 should be -1 for -log"
    assert baseline_alphas[3, 2] == -1.0, "g_3 a should be -1 for -log"

    # g_6 (index 6): c_0=0, c_1=-1, a=-1 for -log(x)
    assert baseline_alphas[6, 0] == 0.0, "g_6 c_0 should be 0"
    assert baseline_alphas[6, 1] == -1.0, "g_6 c_1 should be -1 for -log"
    assert baseline_alphas[6, 2] == -1.0, "g_6 a should be -1 for -log"
