"""Tests for PolynomialBasisLoss."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
from loss.loss_polynomial_basis import PolynomialBasisLoss


class TestPolynomialBasisInitialization:
    """Test initialization modes."""

    def test_bce_initialization(self):
        """Test BCE-equivalent initialization."""
        loss = PolynomialBasisLoss(
            num_repetitions=3,
            use_prior=True,
            prior=0.5,
            init_mode='bce_equivalent',
        )

        # Check specific coefficients are set
        # f_outer: linear term (index 1) should be 1.0
        assert torch.isclose(loss.coefficient_alphas[0, 1], torch.tensor(1.0))
        # f_3: log term (index 5) should be -1.0
        assert torch.isclose(loss.coefficient_alphas[3, 5], torch.tensor(-1.0))
        # f_6: log term (index 5) should be -1.0
        assert torch.isclose(loss.coefficient_alphas[6, 5], torch.tensor(-1.0))

        # All other coefficients should be zero
        for rep in range(3):
            base_idx = rep * 7
            # f_outer: only index 1 should be non-zero
            assert torch.allclose(
                loss.coefficient_alphas[base_idx, [0, 2, 3, 4, 5, 6]],
                torch.zeros(6)
            )

    def test_bce_approximation_accuracy(self):
        """Test that BCE initialization gives close-to-PN-Naive loss.

        Note: The 'bce_equivalent' initialization actually implements PN-Naive loss:
        L = E_pos[-log(p)] + E_neg[-log(1-p)]
        This equals BCE only when classes are balanced (50/50 split).
        """
        torch.manual_seed(42)

        loss = PolynomialBasisLoss(
            num_repetitions=1,
            use_prior=True,
            prior=0.5,
            init_mode='bce_equivalent',
        )

        # Use balanced classes so PN-Naive ≈ BCE
        logits = torch.randn(1000) * 2
        labels = (torch.rand(1000) < 0.5).float()  # 50/50 split
        pu_labels = labels.clone()
        pu_labels[labels == 0] = -1

        poly_loss = loss(logits, pu_labels)

        # Compute PN-Naive loss (what bce_equivalent actually implements)
        p = torch.sigmoid(logits)
        pos_mask = labels == 1
        neg_mask = labels == 0

        pn_naive = -torch.log(p[pos_mask] + 1e-8).mean() - torch.log(1 - p[neg_mask] + 1e-8).mean()

        # Should be very close (within 2% - only difference is log approximation)
        error = abs(poly_loss - pn_naive) / abs(pn_naive)
        assert error < 0.02, f"Error: {error:.2%}, poly={poly_loss:.4f}, pn_naive={pn_naive:.4f}"

    def test_random_initialization(self):
        """Test random initialization."""
        loss = PolynomialBasisLoss(
            num_repetitions=3,
            use_prior=True,
            init_mode='random',
            init_scale=0.01,
        )

        # Should have diverse non-zero values
        assert not torch.allclose(
            loss.coefficient_alphas,
            torch.zeros_like(loss.coefficient_alphas)
        )

    def test_zeros_initialization(self):
        """Test zeros initialization."""
        loss = PolynomialBasisLoss(
            num_repetitions=3,
            use_prior=True,
            init_mode='zeros',
        )

        # All parameters should be zero
        assert torch.allclose(
            loss.coefficient_alphas,
            torch.zeros_like(loss.coefficient_alphas)
        )
        assert torch.allclose(
            loss.coefficient_betas,
            torch.zeros_like(loss.coefficient_betas)
        )


class TestPolynomialBasisForward:
    """Test forward pass and numerical stability."""

    def test_forward_produces_finite_output(self):
        """Test that forward pass produces finite output."""
        loss = PolynomialBasisLoss(
            num_repetitions=3,
            use_prior=True,
            init_mode='bce_equivalent',
        )

        logits = torch.randn(100)
        labels = torch.randint(0, 2, (100,)).float()
        pu_labels = labels.clone()
        pu_labels[labels == 0] = -1

        output = loss(logits, pu_labels)

        assert torch.isfinite(output), f"Output is not finite: {output}"
        assert output.requires_grad, "Output should require gradients"

    def test_edge_case_no_positive(self):
        """Test handling of no positive samples."""
        loss = PolynomialBasisLoss(num_repetitions=1)

        logits = torch.randn(100)
        pu_labels = torch.full((100,), -1.0)  # All unlabeled

        output = loss(logits, pu_labels)
        assert output == 0.0

    def test_edge_case_no_unlabeled(self):
        """Test handling of no unlabeled samples."""
        loss = PolynomialBasisLoss(num_repetitions=1)

        logits = torch.randn(100)
        pu_labels = torch.full((100,), 1.0)  # All positive

        output = loss(logits, pu_labels)
        assert torch.isfinite(output)

    def test_edge_case_extreme_logits(self):
        """Test stability with extreme logits."""
        loss = PolynomialBasisLoss(
            num_repetitions=1,
            init_mode='bce_equivalent',
        )

        # Extreme positive and negative logits
        logits = torch.tensor([-20.0, 20.0, -10.0, 10.0, 0.0])
        pu_labels = torch.tensor([1.0, 1.0, -1.0, -1.0, 1.0])

        output = loss(logits, pu_labels)

        assert torch.isfinite(output), f"Output not finite with extreme logits: {output}"

    def test_oracle_mode(self):
        """Test oracle mode with binary labels."""
        loss = PolynomialBasisLoss(
            num_repetitions=1,
            oracle_mode=True,
            init_mode='bce_equivalent',
        )

        logits = torch.randn(100)
        binary_labels = torch.randint(0, 2, (100,)).float()  # 0/1 labels

        output = loss(logits, binary_labels)

        assert torch.isfinite(output)


class TestPolynomialBasisGradients:
    """Test gradient flow."""

    def test_gradients_flow_through_all_parameters(self):
        """Test that gradients flow to all parameters."""
        loss = PolynomialBasisLoss(
            num_repetitions=1,
            use_prior=True,
            init_mode='random',
        )

        logits = torch.randn(100, requires_grad=True)
        labels = torch.randint(0, 2, (100,)).float()
        pu_labels = labels.clone()
        pu_labels[labels == 0] = -1

        output = loss(logits, pu_labels)
        output.backward()

        # Check gradients exist
        assert loss.coefficient_alphas.grad is not None
        assert loss.coefficient_betas.grad is not None
        assert torch.any(loss.coefficient_alphas.grad != 0)

    def test_backward_is_stable(self):
        """Test that backward pass doesn't explode."""
        loss = PolynomialBasisLoss(
            num_repetitions=3,
            init_mode='bce_equivalent',
        )

        logits = torch.randn(100, requires_grad=True)
        labels = torch.randint(0, 2, (100,)).float()
        pu_labels = labels.clone()
        pu_labels[labels == 0] = -1

        output = loss(logits, pu_labels)
        output.backward()

        # Check gradients are finite
        assert torch.all(torch.isfinite(loss.coefficient_alphas.grad))
        assert torch.all(torch.isfinite(loss.coefficient_betas.grad))

    def test_second_order_gradients(self):
        """Test that second-order gradients work (needed for meta-learning)."""
        loss = PolynomialBasisLoss(
            num_repetitions=1,
            use_prior=True,
            init_mode='random',
        )

        logits = torch.randn(10, requires_grad=True)
        labels = torch.randint(0, 2, (10,)).float()
        pu_labels = labels.clone()
        pu_labels[labels == 0] = -1

        # Compute loss
        output = loss(logits, pu_labels)

        # First-order gradients
        grad_logits = torch.autograd.grad(
            output,
            logits,
            create_graph=True,
            retain_graph=True
        )[0]

        # Second-order gradients (w.r.t. loss parameters)
        grad_loss_params = torch.autograd.grad(
            grad_logits.sum(),
            loss.coefficient_alphas,
        )[0]

        # Should be finite
        assert torch.all(torch.isfinite(grad_loss_params))


class TestPolynomialBasisRegularization:
    """Test regularization computation."""

    def test_regularization_is_nonnegative(self):
        """Test regularization is non-negative."""
        loss = PolynomialBasisLoss(
            num_repetitions=3,
            l1_weight=1e-4,
            l2_weight=1e-3,
        )

        reg = loss.compute_regularization()
        assert reg >= 0

    def test_regularization_increases_with_params(self):
        """Test regularization increases with parameter magnitude."""
        loss1 = PolynomialBasisLoss(num_repetitions=1, init_scale=0.01)
        loss2 = PolynomialBasisLoss(num_repetitions=1, init_scale=0.1)

        reg1 = loss1.compute_regularization()
        reg2 = loss2.compute_regularization()

        assert reg2 > reg1

    def test_regularization_with_zeros(self):
        """Test that zero parameters give zero regularization."""
        loss = PolynomialBasisLoss(
            num_repetitions=1,
            init_mode='zeros',
            l1_weight=1e-4,
            l2_weight=1e-3,
        )

        reg = loss.compute_regularization()
        assert torch.isclose(reg, torch.tensor(0.0))


class TestPolynomialBasisPrior:
    """Test prior conditioning."""

    def test_set_prior_updates_effectively(self):
        """Test that set_prior changes output."""
        torch.manual_seed(42)
        loss = PolynomialBasisLoss(
            num_repetitions=1,
            use_prior=True,
            prior=0.3,
            init_mode='random',
        )

        # Set non-zero betas
        with torch.no_grad():
            loss.coefficient_betas.fill_(0.1)

        logits = torch.randn(100)
        pu_labels = torch.randint(0, 2, (100,)).float()
        pu_labels[pu_labels == 0] = -1

        output1 = loss(logits, pu_labels)

        loss.set_prior(0.7)
        output2 = loss(logits, pu_labels)

        # Outputs should differ
        assert output1 != output2, "Setting prior should change output"

    def test_set_prior_without_use_prior_raises(self):
        """Test that set_prior raises error when use_prior=False."""
        loss = PolynomialBasisLoss(use_prior=False)

        with pytest.raises(ValueError, match="Cannot set prior"):
            loss.set_prior(0.5)

    def test_prior_conditioning_formula(self):
        """Test that prior conditioning follows formula: a = α + β·π."""
        loss = PolynomialBasisLoss(
            num_repetitions=1,
            use_prior=True,
            prior=0.3,
        )

        # Set known values
        with torch.no_grad():
            loss.coefficient_alphas[0, 0] = 1.0
            loss.coefficient_betas[0, 0] = 2.0

        coeffs = loss.get_coefficients(0)

        # a = α + β·π = 1.0 + 2.0·0.3 = 1.6
        expected = 1.0 + 2.0 * 0.3
        assert torch.isclose(coeffs[0], torch.tensor(expected))


class TestPolynomialBasisSpeed:
    """Test computational speed vs MonotonicBasisLoss."""

    def test_speed_comparison(self):
        """Compare speed with MonotonicBasisLoss."""
        import time
        from loss.loss_monotonic_basis import MonotonicBasisLoss

        # Polynomial loss
        poly_loss = PolynomialBasisLoss(
            num_repetitions=3,
            use_prior=True,
            init_mode='bce_equivalent',
        )

        # Monotonic loss
        mono_loss = MonotonicBasisLoss(
            num_repetitions=3,
            num_fourier=16,
            use_prior=True,
            init_mode='bce_equivalent',
            num_integration_points=20,
        )

        # Test data
        logits = torch.randn(1000)
        labels = torch.randint(0, 2, (1000,)).float()
        pu_labels = labels.clone()
        pu_labels[labels == 0] = -1

        # Warm up
        _ = poly_loss(logits, pu_labels)
        _ = mono_loss(logits, pu_labels)

        # Time polynomial
        n_iters = 100
        start = time.time()
        for _ in range(n_iters):
            _ = poly_loss(logits, pu_labels)
        poly_time = time.time() - start

        # Time monotonic
        start = time.time()
        for _ in range(n_iters):
            _ = mono_loss(logits, pu_labels)
        mono_time = time.time() - start

        speedup = mono_time / poly_time
        print(f"\nPolynomial: {poly_time:.4f}s")
        print(f"Monotonic:  {mono_time:.4f}s")
        print(f"Speedup:    {speedup:.2f}×")

        # Polynomial should be faster
        assert poly_time < mono_time, f"Polynomial ({poly_time:.4f}s) should be faster than Monotonic ({mono_time:.4f}s)"
        assert speedup >= 2.0, f"Expected at least 2× speedup, got {speedup:.2f}×"


class TestPolynomialBasisParameterCount:
    """Test parameter counting."""

    def test_parameter_count_with_prior(self):
        """Test parameter count with prior conditioning."""
        loss = PolynomialBasisLoss(
            num_repetitions=3,
            use_prior=True,
        )

        # 3 reps × 7 funcs × 7 coeffs × 2 (alpha + beta) = 294
        expected = 3 * 7 * 7 * 2
        assert loss.get_num_parameters() == expected

    def test_parameter_count_without_prior(self):
        """Test parameter count without prior conditioning."""
        loss = PolynomialBasisLoss(
            num_repetitions=3,
            use_prior=False,
        )

        # 3 reps × 7 funcs × 7 coeffs = 147
        expected = 3 * 7 * 7
        assert loss.get_num_parameters() == expected

    def test_parameter_summary(self):
        """Test parameter summary."""
        loss = PolynomialBasisLoss(
            num_repetitions=3,
            use_prior=True,
        )

        summary = loss.get_parameter_summary()

        assert summary['total_params'] == 294
        assert summary['num_repetitions'] == 3
        assert summary['num_basis_functions'] == 21
        assert summary['coeffs_per_function'] == 7
        assert summary['use_prior'] is True
