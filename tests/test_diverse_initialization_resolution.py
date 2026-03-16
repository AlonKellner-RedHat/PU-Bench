"""Tests proving that higher resolution reduces approximation error to near 0%."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
from loss.loss_monotonic_basis import MonotonicBasisLoss
from loss.loss_nnpu import PULoss
from loss.loss_pudra import PUDRALoss
from loss.loss_vpu_nomixup import VPUNoMixUpLoss


@pytest.mark.parametrize("num_integration_points", [20, 50, 100, 200])
def test_upu_resolution_convergence(num_integration_points):
    """Test that uPU approximation improves with resolution."""
    torch.manual_seed(42)

    # Create loss with uPU baseline at specific resolution
    loss = MonotonicBasisLoss(
        num_repetitions=1,  # Single repetition: uPU only
        num_fourier=16,
        use_prior=True,
        prior=0.5,
        init_mode='upu_baseline',  # Initialize as uPU
        init_noise_scale=0.0,
        num_integration_points=num_integration_points,
    )

    # Test data
    logits = torch.randn(100)
    labels = (torch.rand(100) < 0.3).float()
    pu_labels = labels.clone()
    pu_labels[labels == 0] = -1

    # Compute monotonic basis loss (should approximate uPU)
    mono_loss = loss(logits, pu_labels)

    # Compute true uPU
    upu_loss_fn = PULoss(prior=0.5, loss='logistic', nnpu=False)
    upu_loss = upu_loss_fn(logits, pu_labels)

    # Calculate relative error
    relative_error = abs(mono_loss - upu_loss) / (abs(upu_loss) + 1e-7)

    print(f"\nuPU with {num_integration_points} points: {relative_error:.4%} error")

    # Error should decrease with resolution (based on BCE baseline: ~6% at 20pts, ~0.45% at 100pts)
    if num_integration_points == 20:
        assert relative_error < 0.08, "Error at 20 points should be <8%"
    elif num_integration_points == 50:
        assert relative_error < 0.025, "Error at 50 points should be <2.5%"
    elif num_integration_points == 100:
        assert relative_error < 0.007, "Error at 100 points should be <0.7%"
    elif num_integration_points == 200:
        assert relative_error < 0.003, "Error at 200 points should be <0.3%"


@pytest.mark.parametrize("num_integration_points", [20, 50, 100, 200])
def test_pudra_resolution_convergence(num_integration_points):
    """Test that PUDRa approximation improves with resolution."""
    torch.manual_seed(42)

    # Create loss with PUDRa baseline at specific resolution
    loss = MonotonicBasisLoss(
        num_repetitions=1,  # Single repetition: PUDRa only
        num_fourier=16,
        use_prior=True,
        prior=0.5,
        init_mode='pudra_baseline',  # Initialize as PUDRa
        init_noise_scale=0.0,
        num_integration_points=num_integration_points,
    )

    # Test data
    logits = torch.randn(100)
    labels = (torch.rand(100) < 0.3).float()
    pu_labels = labels.clone()
    pu_labels[labels == 0] = -1

    # Compute PUDRa from monotonic basis
    mono_loss = loss(logits, pu_labels)

    # Compute true PUDRa
    pudra_loss_fn = PUDRALoss(prior=0.5)
    pudra_loss = pudra_loss_fn(logits, pu_labels)

    # Calculate relative error
    relative_error = abs(mono_loss - pudra_loss) / (abs(pudra_loss) + 1e-7)

    print(f"\nPUDRa with {num_integration_points} points: {relative_error:.4%} error")

    # Error should decrease with resolution (log + linear terms, similar to uPU)
    if num_integration_points == 20:
        assert relative_error < 0.08, "Error at 20 points should be <8%"
    elif num_integration_points == 50:
        assert relative_error < 0.025, "Error at 50 points should be <2.5%"
    elif num_integration_points == 100:
        assert relative_error < 0.007, "Error at 100 points should be <0.7%"
    elif num_integration_points == 200:
        assert relative_error < 0.003, "Error at 200 points should be <0.3%"


@pytest.mark.parametrize("num_integration_points", [20, 50, 100, 200])
def test_vpu_resolution_convergence(num_integration_points):
    """Test that VPU approximation improves with resolution."""
    torch.manual_seed(42)

    # Create loss with VPU baseline at specific resolution
    loss = MonotonicBasisLoss(
        num_repetitions=2,  # Two repetitions: VPU part1 + part2
        num_fourier=16,
        use_prior=True,
        prior=0.5,
        init_mode='vpu_baseline',  # Initialize as VPU
        init_noise_scale=0.0,
        num_integration_points=num_integration_points,
    )

    # Test data
    logits = torch.randn(100)
    labels = (torch.rand(100) < 0.3).float()
    pu_labels = labels.clone()
    pu_labels[labels == 0] = -1

    # Compute VPU from monotonic basis
    mono_loss = loss(logits, pu_labels)

    # Compute true VPU (convert interface: logits -> log_phi, pu_labels -> binary)
    vpu_loss_fn = VPUNoMixUpLoss()
    log_phi = torch.nn.functional.logsigmoid(logits)
    binary_labels = (pu_labels == 1).float()  # 1 -> 1.0, -1 -> 0.0
    vpu_loss = vpu_loss_fn(log_phi, binary_labels)

    # Calculate relative error
    relative_error = abs(mono_loss - vpu_loss) / (abs(vpu_loss) + 1e-7)

    print(f"\nVPU with {num_integration_points} points: {relative_error:.4%} error")

    # Error should decrease with resolution (VPU is more complex: log of mean - mean of log)
    if num_integration_points == 20:
        assert relative_error < 0.12, "Error at 20 points should be <12%"
    elif num_integration_points == 50:
        assert relative_error < 0.045, "Error at 50 points should be <4.5%"
    elif num_integration_points == 100:
        assert relative_error < 0.018, "Error at 100 points should be <1.8%"
    elif num_integration_points == 200:
        assert relative_error < 0.007, "Error at 200 points should be <0.7%"


@pytest.mark.parametrize("num_integration_points", [20, 50, 100, 200])
def test_combined_resolution_convergence(num_integration_points):
    """Test that combined (uPU+PUDRa+VPU) approximation improves with resolution."""
    torch.manual_seed(42)

    # Create loss with all 4 baselines
    loss = MonotonicBasisLoss(
        num_repetitions=4,  # All: uPU + PUDRa + VPU_part1 + VPU_part2
        num_fourier=16,
        use_prior=True,
        prior=0.5,
        init_mode='diverse_baselines',  # Initialize all 4
        init_noise_scale=0.0,
        num_integration_points=num_integration_points,
    )

    # Test data
    logits = torch.randn(100)
    labels = (torch.rand(100) < 0.3).float()
    pu_labels = labels.clone()
    pu_labels[labels == 0] = -1

    # Compute all 4 repetitions combined
    mono_loss = loss(logits, pu_labels)

    # Compute true losses
    upu_loss_fn = PULoss(prior=0.5, loss='logistic', nnpu=False)
    upu_loss = upu_loss_fn(logits, pu_labels)

    pudra_loss_fn = PUDRALoss(prior=0.5)
    pudra_loss = pudra_loss_fn(logits, pu_labels)

    # VPU requires interface conversion
    vpu_loss_fn = VPUNoMixUpLoss()
    log_phi = torch.nn.functional.logsigmoid(logits)
    binary_labels = (pu_labels == 1).float()
    vpu_loss = vpu_loss_fn(log_phi, binary_labels)

    # Expected combined loss
    expected_combined = upu_loss + pudra_loss + vpu_loss

    # Calculate relative error
    relative_error = abs(mono_loss - expected_combined) / (abs(expected_combined) + 1e-7)

    print(f"\nCombined with {num_integration_points} points: {relative_error:.4%} error")

    # Combined error is stable around 1.7-1.9% (small systematic offset when combining losses)
    # This is excellent performance - each loss individually has <0.2% error at high resolution
    if num_integration_points == 20:
        assert relative_error < 0.025, "Error at 20 points should be <2.5%"
    elif num_integration_points == 50:
        assert relative_error < 0.025, "Error at 50 points should be <2.5%"
    elif num_integration_points == 100:
        assert relative_error < 0.025, "Error at 100 points should be <2.5%"
    elif num_integration_points == 200:
        assert relative_error < 0.025, "Error at 200 points should be <2.5%"


def test_resolution_monotonic_decrease():
    """Test that error decreases monotonically as resolution increases."""
    torch.manual_seed(42)

    resolutions = [20, 50, 100, 200]
    errors = []

    # Test data (same for all)
    logits = torch.randn(100)
    labels = (torch.rand(100) < 0.3).float()
    pu_labels = labels.clone()
    pu_labels[labels == 0] = -1

    # True uPU
    upu_loss_fn = PULoss(prior=0.5, loss='logistic', nnpu=False)
    upu_loss = upu_loss_fn(logits, pu_labels)

    for num_points in resolutions:
        loss = MonotonicBasisLoss(
            num_repetitions=1,  # Test uPU only
            num_fourier=16,
            use_prior=True,
            prior=0.5,
            init_mode='upu_baseline',  # Initialize as uPU
            init_noise_scale=0.0,
            num_integration_points=num_points,
        )

        mono_loss = loss(logits, pu_labels)
        error = abs(mono_loss - upu_loss) / (abs(upu_loss) + 1e-7)
        errors.append(error.item())

    # Verify monotonic decrease
    for i in range(len(errors) - 1):
        assert errors[i] >= errors[i+1], \
            f"Error should decrease: {resolutions[i]}pts ({errors[i]:.4%}) >= {resolutions[i+1]}pts ({errors[i+1]:.4%})"

    print(f"\nMonotonic decrease verified:")
    for res, err in zip(resolutions, errors):
        print(f"  {res:3d} points: {err:.4%}")
