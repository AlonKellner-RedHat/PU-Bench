"""Learnable Monotonic Basis Loss for PU Learning.

This module implements a novel learnable loss function that uses the Universal
Monotonic Spectral Basis as its core transformation mechanism. Unlike traditional
PU losses with fixed forms, this loss has learnable parameters that can be
optimized through meta-learning.

Mathematical Structure:
    L = Σ_{rep} g_outer(
        (g_1(p_all) + g_2(1-p_all)).mean() +
        (g_3(p_pos) + g_4(1-p_pos)).mean() +
        (g_5(p_oth) + g_6(1-p_oth)).mean()
    )

    where g are monotonic basis integrands with learnable parameters.

Key Features:
    - First learnable loss in PU-Bench (all others use fixed hyperparameters)
    - Meta-learning ready (designed for learning good losses across tasks)
    - Prior-adaptive (can transfer between different class priors)
    - Transportable (state_dict preserves internal coefficients)

For standard PU learning, only the model parameters are optimized (not loss parameters).
For meta-learning, both model and loss parameters would be optimized.
"""

import torch
import torch.nn as nn
from typing import Optional
import sys
import os

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.monotonic_basis_torch import monotonic_basis_integrand


class MonotonicBasisLoss(nn.Module):
    """Learnable loss function using monotonic basis transformations.

    This loss has a hierarchical structure with multiple repetitions:
    - Each repetition has 7 basis functions (1 outer + 6 inner)
    - Inner functions transform p_all, p_pos, and p_oth
    - Outer function transforms the sum of inner means

    Attributes:
        num_repetitions: Number of repetition blocks (default 3)
        num_fourier: Number of Fourier coefficients per basis (default 5)
        use_prior: Whether to condition parameters on class prior
        prior: Class prior π (only used if use_prior=True)
        oracle_mode: If True, expects binary labels (0/1); if False, PU labels (±1)

    Parameters:
        If use_prior=False:
            - baseline_params: shape [num_repetitions * 7, 9]
            - fourier_params: shape [num_repetitions * 7, num_fourier]

        If use_prior=True:
            - baseline_alphas: shape [num_repetitions * 7, 9]
            - baseline_betas: shape [num_repetitions * 7, 9]
            - fourier_alphas: shape [num_repetitions * 7, num_fourier]
            - fourier_betas: shape [num_repetitions * 7, num_fourier]
            Then: param = alpha + beta * prior

    Examples:
        >>> # Standard PU mode without prior
        >>> loss = MonotonicBasisLoss(num_repetitions=3, use_prior=False, oracle_mode=False)
        >>> outputs = torch.randn(100)  # Model logits
        >>> labels = torch.randint(0, 2, (100,)) * 2 - 1  # PU labels: {-1, 1}
        >>> loss_val = loss(outputs, labels)

        >>> # Oracle mode with prior conditioning
        >>> loss = MonotonicBasisLoss(
        >>>     num_repetitions=3, use_prior=True, prior=0.3, oracle_mode=True
        >>> )
        >>> outputs = torch.randn(100)
        >>> labels = torch.randint(0, 2, (100,))  # Binary labels: {0, 1}
        >>> loss_val = loss(outputs, labels)

        >>> # Transfer to different prior
        >>> loss.set_prior(0.5)  # Changes effective parameters without retraining
    """

    def __init__(
        self,
        num_repetitions: int = 3,
        num_fourier: int = 5,
        use_prior: bool = True,
        prior: float = 0.5,
        oracle_mode: bool = False,
        init_scale: float = 0.01,
    ):
        """Initialize learnable monotonic basis loss.

        Args:
            num_repetitions: Number of repetition blocks
            num_fourier: Number of Fourier coefficients per basis function
            use_prior: If True, parameters are linear functions of prior
            prior: Class prior value (only used if use_prior=True)
            oracle_mode: If True, expects binary labels; if False, PU labels
            init_scale: Initialization scale for random parameters
        """
        super().__init__()

        self.num_repetitions = num_repetitions
        self.num_fourier = num_fourier
        self.use_prior = use_prior
        self.oracle_mode = oracle_mode
        self.init_scale = init_scale
        self.name = "monotonic_basis"

        # Total number of basis functions
        # Each repetition has 7 functions: 1 outer + 6 inner
        self.total_basis_funcs = num_repetitions * 7

        # Baseline parameters: c_0, a, b, c, d, e, g, h, t_0 (9 total)
        num_baseline = 9

        if use_prior:
            # Prior-conditioned: param = alpha + beta * prior
            self.baseline_alphas = nn.Parameter(
                torch.randn(self.total_basis_funcs, num_baseline) * init_scale
            )
            self.baseline_betas = nn.Parameter(
                torch.randn(self.total_basis_funcs, num_baseline) * init_scale
            )
            self.fourier_alphas = nn.Parameter(
                torch.randn(self.total_basis_funcs, num_fourier) * init_scale
            )
            self.fourier_betas = nn.Parameter(
                torch.randn(self.total_basis_funcs, num_fourier) * init_scale
            )

            # Store prior as buffer (not trainable, but saved in state_dict)
            self.register_buffer(
                "prior_tensor", torch.tensor(prior, dtype=torch.float32)
            )
        else:
            # Direct parameters
            self.baseline_params = nn.Parameter(
                torch.randn(self.total_basis_funcs, num_baseline) * init_scale
            )
            self.fourier_params = nn.Parameter(
                torch.randn(self.total_basis_funcs, num_fourier) * init_scale
            )

            self.register_buffer("prior_tensor", torch.tensor(0.0, dtype=torch.float32))

        # Initialize some parameters to reasonable values
        self._initialize_params()

    def _initialize_params(self):
        """Initialize parameters to reasonable values for better training."""
        with torch.no_grad():
            if self.use_prior:
                # Initialize baseline_alphas for better starting point
                # c_0 (index 0): small positive for reasonable output scale
                self.baseline_alphas[:, 0].fill_(0.1)
                # h (index 7): around 1.0 for reasonable sigmoid steepness
                self.baseline_alphas[:, 7].fill_(1.0)
                # t_0 (index 8): around 0.5 for center of domain
                self.baseline_alphas[:, 8].fill_(0.5)
            else:
                # Direct params
                self.baseline_params[:, 0].fill_(0.1)  # c_0
                self.baseline_params[:, 7].fill_(1.0)  # h
                self.baseline_params[:, 8].fill_(0.5)  # t_0

    def get_basis_params(self, idx: int) -> tuple:
        """Get effective parameters for basis function at index idx.

        Args:
            idx: Basis function index (0 to total_basis_funcs-1)

        Returns:
            (baseline, fourier) where:
                baseline: shape [9] - (c_0, a, b, c, d, e, g, h, t_0)
                fourier: shape [num_fourier] - (d_1, ..., d_K)
        """
        if self.use_prior:
            baseline = (
                self.baseline_alphas[idx] + self.baseline_betas[idx] * self.prior_tensor
            )
            fourier = (
                self.fourier_alphas[idx] + self.fourier_betas[idx] * self.prior_tensor
            )
        else:
            baseline = self.baseline_params[idx]
            fourier = self.fourier_params[idx]

        return baseline, fourier

    def apply_basis(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        """Apply basis function at index idx to input x.

        Args:
            x: Input tensor, shape [N] or [N, ...] or scalar
            idx: Basis function index

        Returns:
            Output tensor, same shape as x
        """
        baseline, fourier = self.get_basis_params(idx)

        # Extract individual baseline parameters
        c_0 = baseline[0]
        a = baseline[1]
        b = baseline[2]
        c = baseline[3]
        d = baseline[4]
        e = baseline[5]
        g = baseline[6]
        h = baseline[7]
        t_0 = baseline[8]

        return monotonic_basis_integrand(x, c_0, a, b, c, d, e, g, h, t_0, fourier)

    def forward(
        self,
        outputs: torch.Tensor,
        pu_labels: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute monotonic basis loss.

        Args:
            outputs: Model logits, shape [N]
            pu_labels: Labels, shape [N]
                If oracle_mode=True: binary labels (0=negative, 1=positive)
                If oracle_mode=False: PU labels (-1=unlabeled, 1=positive)
            weights: Optional sample weights (unused, for API compatibility)

        Returns:
            Scalar loss tensor

        Note:
            For standard training, only model parameters are optimized.
            For meta-learning, both model and loss parameters would be optimized.
        """
        # Flatten inputs
        outputs = outputs.view(-1)
        pu_labels = pu_labels.view(-1)

        # Apply sigmoid to get probabilities
        p = torch.sigmoid(outputs)

        # Separate into groups based on mode
        if self.oracle_mode:
            # Oracle: 1=positive, 0=negative
            pos_mask = pu_labels == 1
            oth_mask = pu_labels == 0
        else:
            # PU: 1=positive, -1=unlabeled
            pos_mask = pu_labels == 1
            oth_mask = pu_labels == -1

        # Get group probabilities
        p_all = p  # All samples
        p_pos = p[pos_mask]  # Positive samples
        p_oth = p[oth_mask]  # Other samples (negative or unlabeled)

        # Handle edge case: no positive samples
        if len(p_pos) == 0:
            return torch.tensor(0.0, device=p.device, requires_grad=True)

        # Handle edge case: no other samples
        if len(p_oth) == 0:
            # In this case, create dummy tensor for computation
            p_oth = torch.zeros(1, device=p.device)

        # Compute loss across all repetitions
        total_loss = torch.tensor(0.0, device=p.device)

        for rep in range(self.num_repetitions):
            # Base index for this repetition
            # Each repetition has 7 basis functions: indices [rep*7, rep*7+6]
            base_idx = rep * 7

            # ===== Inner terms =====

            # Term 1: (f_1(p_all) + f_2(1-p_all)).mean()
            term1_a = self.apply_basis(p_all, base_idx + 1)
            term1_b = self.apply_basis(1.0 - p_all, base_idx + 2)
            term1 = (term1_a + term1_b).mean()

            # Term 2: (f_3(p_pos) + f_4(1-p_pos)).mean()
            term2_a = self.apply_basis(p_pos, base_idx + 3)
            term2_b = self.apply_basis(1.0 - p_pos, base_idx + 4)
            term2 = (term2_a + term2_b).mean()

            # Term 3: (f_5(p_oth) + f_6(1-p_oth)).mean()
            term3_a = self.apply_basis(p_oth, base_idx + 5)
            term3_b = self.apply_basis(1.0 - p_oth, base_idx + 6)
            term3 = (term3_a + term3_b).mean()

            # Sum of inner terms (scalar)
            inner_sum = term1 + term2 + term3

            # ===== Outer term =====
            # Apply outer basis function to the scalar sum
            # Since inner_sum is scalar, we need to make it a tensor for apply_basis
            inner_sum_tensor = inner_sum.view(1)  # Shape [1]
            outer_result = self.apply_basis(inner_sum_tensor, base_idx)

            # Add to total loss
            total_loss = total_loss + outer_result.squeeze()

        return total_loss

    def set_prior(self, new_prior: float):
        """Update prior without changing learned (alpha, beta) coefficients.

        This enables transfer to new datasets with different priors.

        Args:
            new_prior: New class prior value

        Raises:
            ValueError: If use_prior=False

        Examples:
            >>> loss = MonotonicBasisLoss(use_prior=True, prior=0.3)
            >>> # Train on dataset A...
            >>> # Now transfer to dataset B with prior=0.5
            >>> loss.set_prior(0.5)
        """
        if not self.use_prior:
            raise ValueError("Cannot set prior when use_prior=False")

        self.prior_tensor.fill_(new_prior)

    def get_num_parameters(self) -> int:
        """Get total number of learnable parameters.

        Returns:
            Total number of parameters

        Examples:
            >>> loss = MonotonicBasisLoss(num_repetitions=3, num_fourier=5, use_prior=True)
            >>> num_params = loss.get_num_parameters()
            >>> print(num_params)  # 588 for default settings
        """
        return sum(p.numel() for p in self.parameters())

    def get_parameter_summary(self) -> dict:
        """Get summary of parameter counts and configuration.

        Returns:
            Dictionary with parameter counts and configuration

        Examples:
            >>> loss = MonotonicBasisLoss(num_repetitions=3, num_fourier=5, use_prior=True)
            >>> summary = loss.get_parameter_summary()
            >>> print(summary)
            {
                'total_params': 588,
                'baseline_params': 378,
                'fourier_params': 210,
                'num_basis_functions': 21,
                'num_repetitions': 3,
                'use_prior': True,
            }
        """
        if self.use_prior:
            baseline_count = (
                self.baseline_alphas.numel() + self.baseline_betas.numel()
            )
            fourier_count = self.fourier_alphas.numel() + self.fourier_betas.numel()
        else:
            baseline_count = self.baseline_params.numel()
            fourier_count = self.fourier_params.numel()

        return {
            "total_params": self.get_num_parameters(),
            "baseline_params": baseline_count,
            "fourier_params": fourier_count,
            "num_basis_functions": self.total_basis_funcs,
            "num_repetitions": self.num_repetitions,
            "use_prior": self.use_prior,
        }
