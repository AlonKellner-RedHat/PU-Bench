"""Tests for diverse baseline initialization (uPU + PUDRa + VPU)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
from loss.loss_monotonic_basis import MonotonicBasisLoss


def test_repetitions_are_different():
    """Test that the 4 repetitions have different initializations."""
    loss = MonotonicBasisLoss(
        num_repetitions=4,
        num_fourier=16,
        use_prior=True,
        init_mode='diverse_baselines',
        init_noise_scale=0.0,  # No noise for exact comparison
    )

    # Compare baseline_alphas across repetitions
    rep0 = loss.baseline_alphas[0:7]
    rep1 = loss.baseline_alphas[7:14]
    rep2 = loss.baseline_alphas[14:21]
    rep3 = loss.baseline_alphas[21:28]

    # Should NOT be identical (unlike BCE init)
    assert not torch.allclose(rep0, rep1), "Rep 0 and 1 should differ"
    assert not torch.allclose(rep1, rep2), "Rep 1 and 2 should differ"
    assert not torch.allclose(rep2, rep3), "Rep 2 and 3 should differ"
    assert not torch.allclose(rep0, rep2), "Rep 0 and 2 should differ"


def test_upu_approximation():
    """Test that uPU baseline approximates uPU loss."""
    torch.manual_seed(42)

    # Create loss with uPU baseline
    loss = MonotonicBasisLoss(
        num_repetitions=1,  # Single repetition
        num_fourier=16,
        use_prior=True,
        prior=0.5,
        init_mode='upu_baseline',  # Initialize as uPU
        init_noise_scale=0.0,
    )

    # Test data
    logits = torch.randn(100)
    labels = (torch.rand(100) < 0.3).float()
    pu_labels = labels.clone()
    pu_labels[labels == 0] = -1

    # Compute monotonic basis loss (should approximate uPU)
    mono_loss = loss(logits, pu_labels)

    # Compute true uPU
    from loss.loss_nnpu import PULoss
    upu_loss_fn = PULoss(prior=0.5, loss='logistic', nnpu=False)
    upu_loss = upu_loss_fn(logits, pu_labels)

    # Should be reasonably close
    relative_error = abs(mono_loss - upu_loss) / (upu_loss + 1e-7)
    assert relative_error < 0.2, \
        f"uPU approximation error too high: {relative_error:.1%}"


def test_pudra_approximation():
    """Test that PUDRa baseline has correct structure."""
    torch.manual_seed(42)

    # Create loss with PUDRa baseline
    loss = MonotonicBasisLoss(
        num_repetitions=1,  # Single repetition
        num_fourier=16,
        use_prior=True,
        prior=0.5,
        init_mode='pudra_baseline',  # Initialize as PUDRa
        init_noise_scale=0.0,
    )

    # Check rep 0 (indices 0-6) has expected structure
    # f_outer: identity (alpha[0,0]=1)
    assert torch.isclose(loss.baseline_alphas[0, 0], torch.tensor(1.0))

    # f_3: -π·log(x) (beta[3,1]=-1, alpha[3,2]=-1)
    assert torch.isclose(loss.baseline_betas[3, 1], torch.tensor(-1.0))
    assert torch.isclose(loss.baseline_alphas[3, 2], torch.tensor(-1.0))

    # f_5: x (alpha[5,0]=1)
    assert torch.isclose(loss.baseline_alphas[5, 0], torch.tensor(1.0))


def test_vpu_part1_structure():
    """Test that VPU part 1 has correct structure for log(E_all[p])."""
    torch.manual_seed(42)

    loss = MonotonicBasisLoss(
        num_repetitions=2,  # Two repetitions for VPU
        num_fourier=16,
        use_prior=True,
        prior=0.5,
        init_mode='vpu_baseline',  # Initialize as VPU
        init_noise_scale=0.0,
    )

    # Check rep 0 (indices 0-6) structure - VPU part 1
    # f_outer: log(x) (alpha[0,0]=0, alpha[0,1]=1, alpha[0,2]=-1)
    assert torch.isclose(loss.baseline_alphas[0, 0], torch.tensor(0.0))
    assert torch.isclose(loss.baseline_alphas[0, 1], torch.tensor(1.0))
    assert torch.isclose(loss.baseline_alphas[0, 2], torch.tensor(-1.0))

    # f_1: x (alpha[1,0]=1)
    assert torch.isclose(loss.baseline_alphas[1, 0], torch.tensor(1.0))


def test_vpu_part2_structure():
    """Test that VPU part 2 has correct structure for -E_P[log(p)]."""
    torch.manual_seed(42)

    loss = MonotonicBasisLoss(
        num_repetitions=2,  # Two repetitions for VPU
        num_fourier=16,
        use_prior=True,
        prior=0.5,
        init_mode='vpu_baseline',  # Initialize as VPU
        init_noise_scale=0.0,
    )

    # Check rep 1 (indices 7-13) structure - VPU part 2
    # f_outer: -x (alpha[7,0]=-1)
    assert torch.isclose(loss.baseline_alphas[7, 0], torch.tensor(-1.0))

    # f_3: log(x) (alpha[10,0]=0, alpha[10,1]=1, alpha[10,2]=-1)
    assert torch.isclose(loss.baseline_alphas[10, 0], torch.tensor(0.0))
    assert torch.isclose(loss.baseline_alphas[10, 1], torch.tensor(1.0))
    assert torch.isclose(loss.baseline_alphas[10, 2], torch.tensor(-1.0))


def test_noise_is_applied():
    """Test that initialization noise is added."""
    loss_no_noise = MonotonicBasisLoss(
        num_repetitions=4,
        num_fourier=16,
        use_prior=True,
        init_mode='diverse_baselines',
        init_noise_scale=0.0,
    )

    torch.manual_seed(42)
    loss_with_noise = MonotonicBasisLoss(
        num_repetitions=4,
        num_fourier=16,
        use_prior=True,
        init_mode='diverse_baselines',
        init_noise_scale=0.01,
    )

    # Should differ due to noise
    assert not torch.allclose(
        loss_no_noise.baseline_alphas,
        loss_with_noise.baseline_alphas
    ), "Noise should make parameters different"
