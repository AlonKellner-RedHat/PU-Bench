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

from utils.monotonic_basis_torch import monotonic_basis_full


class MonotonicBasisLoss(nn.Module):
    """Learnable loss function using monotonic basis transformations.

    This loss has a hierarchical structure with multiple repetitions:
    - Each repetition has 7 basis functions (1 outer + 6 inner)
    - Inner functions transform p_all, p_pos, and p_oth
    - Outer function transforms the sum of inner means

    Attributes:
        num_repetitions: Number of repetition blocks (default 3)
        num_fourier: Number of Fourier coefficients per basis (default 16, vectorized)
        use_prior: Whether to condition parameters on class prior
        prior: Class prior π (only used if use_prior=True)
        oracle_mode: If True, expects binary labels (0/1); if False, PU labels (±1)

    Parameters:
        If use_prior=False:
            - baseline_params: shape [num_repetitions * 7, 10]  (c_0, c_1, a, b, c, d, e, g, h, t_0)
            - fourier_params: shape [num_repetitions * 7, num_fourier]

        If use_prior=True:
            - baseline_alphas: shape [num_repetitions * 7, 10]  (c_0, c_1, a, b, c, d, e, g, h, t_0)
            - baseline_betas: shape [num_repetitions * 7, 10]
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
        num_fourier: int = 16,
        use_prior: bool = True,
        prior: float = 0.5,
        oracle_mode: bool = False,
        init_scale: float = 0.01,
        init_mode: str = 'random',
        init_noise_scale: float = 0.0,
        l1_weight: float = 1e-4,
        l2_weight: float = 1e-3,
        num_integration_points: int = 20,
        integration_chunk_size: int = None,
    ):
        """Initialize learnable monotonic basis loss.

        Args:
            num_repetitions: Number of repetition blocks
            num_fourier: Number of Fourier coefficients per basis function (default 16)
            use_prior: If True, parameters are linear functions of prior
            prior: Class prior value (only used if use_prior=True)
            oracle_mode: If True, expects binary labels; if False, PU labels
            init_scale: Initialization scale for random parameters
            init_mode: Initialization mode ('random', 'bce_equivalent', 'upu_baseline', 'pudra_baseline', 'vpu_baseline', 'diverse_baselines')
            init_noise_scale: Additional Gaussian noise scale for baseline initializations (encourages exploration)
            l1_weight: L1 regularization weight for baseline parameters (sparsity)
            l2_weight: L2 regularization weight for Fourier parameters (stability)
            num_integration_points: Number of points for numerical integration (100=0.45% error, 20=fast, 50=1.4% error)
            integration_chunk_size: Process inputs in chunks during integration (default: None=no chunking, 256=recommended for 100 points)
        """
        super().__init__()

        self.num_repetitions = num_repetitions
        self.num_fourier = num_fourier
        self.use_prior = use_prior
        self.oracle_mode = oracle_mode
        self.init_scale = init_scale
        self.init_mode = init_mode
        self.init_noise_scale = init_noise_scale
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.num_integration_points = num_integration_points
        self.integration_chunk_size = integration_chunk_size
        self.name = "monotonic_basis"

        # Total number of basis functions
        # Each repetition has 7 functions: 1 outer + 6 inner
        self.total_basis_funcs = num_repetitions * 7

        # Baseline parameters: c_0, c_1, a, b, c, d, e, g, h, t_0 (10 total)
        num_baseline = 10

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

        # Initialize parameters based on init_mode
        if init_mode == 'diverse_baselines':
            self._initialize_params_diverse_baselines()
        elif init_mode == 'upu_baseline':
            self._initialize_params_upu_baseline()
        elif init_mode == 'pudra_baseline':
            self._initialize_params_pudra_baseline()
        elif init_mode == 'vpu_baseline':
            self._initialize_params_vpu_baseline()
        elif init_mode == 'bce_equivalent':
            self._initialize_params_bce_equivalent()
        else:
            self._initialize_params()  # Random initialization

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

    def _initialize_params_bce_equivalent(self):
        """Initialize parameters to approximate BCE/PN-Naive.

        For PN-Naive (treating unlabeled as negative):
            L = E_pos[-log(p)] + E_unlabeled[-log(1-p)]

        Using monotonic basis structure:
            L = g_outer(g_3(p_pos).mean() + g_6(1-p_oth).mean())

        Where:
            g_outer(x) = x (identity)
            g_1(x) = 0, g_2(x) = 0 (no contribution from p_all)
            g_3(x) = -log(x) (for positive samples)
            g_4(x) = 0
            g_5(x) = 0
            g_6(x) = -log(x) (for unlabeled/negative samples)

        Key insight: -log(x) = -∫[1,x] 1/t dt
        So integrand f(t) = -1/t = -t^(-1)
        """
        with torch.no_grad():
            if self.use_prior:
                # Initialize all to zeros
                self.baseline_alphas.zero_()
                self.baseline_betas.zero_()
                self.fourier_alphas.zero_()
                self.fourier_betas.zero_()

                # Basis function indices in first repetition:
                # 0: g_outer
                # 1, 2: g_1, g_2 (for p_all, 1-p_all)
                # 3, 4: g_3, g_4 (for p_pos, 1-p_pos)
                # 5, 6: g_5, g_6 (for p_oth, 1-p_oth)

                # g_outer(x) = x (identity)
                # Formula: f(x) = c₀·x + c₁·∫[1,x] g(t) dt
                # For identity: use linear term only
                self.baseline_alphas[0, 0] = 1.0   # c_0 = 1 (gives x)
                self.baseline_alphas[0, 1] = 0.0   # c_1 = 0 (no integral)

                # g_1, g_2: all zeros (no contribution from p_all terms)
                # Already zeroed above

                # g_3(x) = -log(x)
                # Formula: f(x) = c₀·x + c₁·∫[1,x] g(t) dt
                # where g(t) = exp(a·log(t) + b + ...)
                # For -log(x): need integrand g(t) = 1/t
                # Since g(t) = exp(a·log(t)) = t^a, we need t^(-1) → a = -1
                # Then: f(x) = 0 + (-1)·∫[1,x] t^(-1) dt = -log(x) ✓
                self.baseline_alphas[3, 0] = 0.0   # c_0 = 0
                self.baseline_alphas[3, 1] = -1.0  # c_1 = -1
                self.baseline_alphas[3, 2] = -1.0  # a = -1
                self.baseline_alphas[3, 3] = 0.0   # b = 0

                # g_4: all zeros
                # Already zeroed above

                # g_5: all zeros
                # Already zeroed above

                # g_6(x) = -log(x)
                # Same parameters as g_3
                self.baseline_alphas[6, 0] = 0.0   # c_0 = 0
                self.baseline_alphas[6, 1] = -1.0  # c_1 = -1
                self.baseline_alphas[6, 2] = -1.0  # a = -1
                self.baseline_alphas[6, 3] = 0.0   # b = 0

                # Copy first repetition to all others (shared initialization)
                for rep in range(1, self.num_repetitions):
                    start_idx = rep * 7
                    self.baseline_alphas[start_idx:start_idx+7] = \
                        self.baseline_alphas[0:7].clone()
                    self.baseline_betas[start_idx:start_idx+7] = \
                        self.baseline_betas[0:7].clone()
                    self.fourier_alphas[start_idx:start_idx+7] = \
                        self.fourier_alphas[0:7].clone()
                    self.fourier_betas[start_idx:start_idx+7] = \
                        self.fourier_betas[0:7].clone()

                print("✓ Initialized MonotonicBasisLoss with BCE-equivalent parameters")
                print(f"  - g_outer(x) = x, g_3(x) = g_6(x) = -log(x), others = 0")
                print(f"  - All {self.num_repetitions} repetitions share same initial values")

            else:
                # Non-prior mode initialization
                self.baseline_params.zero_()
                self.fourier_params.zero_()

                # g_outer: identity
                self.baseline_params[0, 0] = 1.0   # c_0 = 1 (gives x)
                self.baseline_params[0, 1] = 0.0   # c_1 = 0 (no integral)

                # g_3: -log(x)
                self.baseline_params[3, 0] = 0.0   # c_0 = 0
                self.baseline_params[3, 1] = -1.0  # c_1 = -1
                self.baseline_params[3, 2] = -1.0  # a = -1
                self.baseline_params[3, 3] = 0.0   # b = 0

                # g_6: -log(x)
                self.baseline_params[6, 0] = 0.0   # c_0 = 0
                self.baseline_params[6, 1] = -1.0  # c_1 = -1
                self.baseline_params[6, 2] = -1.0  # a = -1
                self.baseline_params[6, 3] = 0.0   # b = 0

                # Copy to other repetitions
                for rep in range(1, self.num_repetitions):
                    start_idx = rep * 7
                    self.baseline_params[start_idx:start_idx+7] = \
                        self.baseline_params[0:7].clone()
                    self.fourier_params[start_idx:start_idx+7] = \
                        self.fourier_params[0:7].clone()

                print("✓ Initialized MonotonicBasisLoss with BCE-equivalent parameters")
                print(f"  - g_outer(x) = x, g_3(x) = g_6(x) = -log(x), others = 0")
                print(f"  - All {self.num_repetitions} repetitions share same initial values")

    def _initialize_params_upu_baseline(self):
        """Initialize single repetition as uPU loss.

        L_uPU = π·E_P[-log(p)] + E_U[-log(1-p)] - π·E_P[-log(1-p)]

        Requires: num_repetitions=1, use_prior=True
        """
        with torch.no_grad():
            if not self.use_prior:
                raise ValueError("upu_baseline requires use_prior=True")
            if self.num_repetitions != 1:
                raise ValueError(f"upu_baseline requires num_repetitions=1, got {self.num_repetitions}")

            # Zero all parameters
            self.baseline_alphas.zero_()
            self.baseline_betas.zero_()
            self.fourier_alphas.zero_()
            self.fourier_betas.zero_()

            # Repetition 0: uPU
            base_idx = 0

            # f_outer: identity
            self.baseline_alphas[base_idx + 0, 0] = 1.0

            # f_3: -π·log(x)
            self.baseline_betas[base_idx + 3, 1] = -1.0  # c₁ = -π
            self.baseline_alphas[base_idx + 3, 2] = -1.0  # a = -1

            # f_4: π·log(x)
            self.baseline_betas[base_idx + 4, 1] = 1.0  # c₁ = π
            self.baseline_alphas[base_idx + 4, 2] = -1.0  # a = -1

            # f_6: -log(x)
            self.baseline_alphas[base_idx + 6, 1] = -1.0  # c₁ = -1
            self.baseline_alphas[base_idx + 6, 2] = -1.0  # a = -1

            # Add noise if specified
            if self.init_noise_scale > 0:
                self.baseline_alphas.add_(torch.randn_like(self.baseline_alphas) * self.init_noise_scale)
                self.baseline_betas.add_(torch.randn_like(self.baseline_betas) * self.init_noise_scale)
                self.fourier_alphas.add_(torch.randn_like(self.fourier_alphas) * self.init_noise_scale)
                self.fourier_betas.add_(torch.randn_like(self.fourier_betas) * self.init_noise_scale)

    def _initialize_params_pudra_baseline(self):
        """Initialize single repetition as PUDRa loss.

        L_PUDRa = π·E_P[-log(p)] + E_U[p]

        Requires: num_repetitions=1, use_prior=True
        """
        with torch.no_grad():
            if not self.use_prior:
                raise ValueError("pudra_baseline requires use_prior=True")
            if self.num_repetitions != 1:
                raise ValueError(f"pudra_baseline requires num_repetitions=1, got {self.num_repetitions}")

            # Zero all parameters
            self.baseline_alphas.zero_()
            self.baseline_betas.zero_()
            self.fourier_alphas.zero_()
            self.fourier_betas.zero_()

            # Repetition 0: PUDRa
            base_idx = 0

            # f_outer: identity
            self.baseline_alphas[base_idx + 0, 0] = 1.0

            # f_3: -π·log(x)
            self.baseline_betas[base_idx + 3, 1] = -1.0  # c₁ = -π
            self.baseline_alphas[base_idx + 3, 2] = -1.0  # a = -1

            # f_5: x (identity on p_oth)
            self.baseline_alphas[base_idx + 5, 0] = 1.0

            # Add noise if specified
            if self.init_noise_scale > 0:
                self.baseline_alphas.add_(torch.randn_like(self.baseline_alphas) * self.init_noise_scale)
                self.baseline_betas.add_(torch.randn_like(self.baseline_betas) * self.init_noise_scale)
                self.fourier_alphas.add_(torch.randn_like(self.fourier_alphas) * self.init_noise_scale)
                self.fourier_betas.add_(torch.randn_like(self.fourier_betas) * self.init_noise_scale)

    def _initialize_params_vpu_baseline(self):
        """Initialize 2 repetitions as VPU loss (part1 + part2).

        L_VPU = log(E_all[φ(x)]) - E_P[log(φ(x))]

        Requires: num_repetitions=2, use_prior=True
        """
        with torch.no_grad():
            if not self.use_prior:
                raise ValueError("vpu_baseline requires use_prior=True")
            if self.num_repetitions != 2:
                raise ValueError(f"vpu_baseline requires num_repetitions=2, got {self.num_repetitions}")

            # Zero all parameters
            self.baseline_alphas.zero_()
            self.baseline_betas.zero_()
            self.fourier_alphas.zero_()
            self.fourier_betas.zero_()

            # Repetition 0: VPU First Factor (log(E_all[p]))
            base_idx = 0

            # f_outer: log(x)
            self.baseline_alphas[base_idx + 0, 0] = 0.0  # c₀ = 0
            self.baseline_alphas[base_idx + 0, 1] = 1.0  # c₁ = 1
            self.baseline_alphas[base_idx + 0, 2] = -1.0  # a = -1

            # f_1: x (identity, so E_all[f_1(p)] = E_all[p])
            self.baseline_alphas[base_idx + 1, 0] = 1.0

            # Repetition 1: VPU Second Factor (-E_P[log(p)])
            base_idx = 7

            # f_outer: -x
            self.baseline_alphas[base_idx + 0, 0] = -1.0

            # f_3: log(x)
            self.baseline_alphas[base_idx + 3, 0] = 0.0  # c₀ = 0
            self.baseline_alphas[base_idx + 3, 1] = 1.0  # c₁ = 1
            self.baseline_alphas[base_idx + 3, 2] = -1.0  # a = -1

            # Add noise if specified
            if self.init_noise_scale > 0:
                self.baseline_alphas.add_(torch.randn_like(self.baseline_alphas) * self.init_noise_scale)
                self.baseline_betas.add_(torch.randn_like(self.baseline_betas) * self.init_noise_scale)
                self.fourier_alphas.add_(torch.randn_like(self.fourier_alphas) * self.init_noise_scale)
                self.fourier_betas.add_(torch.randn_like(self.fourier_betas) * self.init_noise_scale)

    def _initialize_params_diverse_baselines(self):
        """Initialize 4 repetitions with different strong baselines.

        Repetition 0: uPU (unbiased PU with prior factor)
        Repetition 1: PUDRa (density ratio)
        Repetition 2: VPU first factor - log(E_all[φ(x)])
        Repetition 3: VPU second factor - E_P[log(φ(x))] (negated)

        Then add Gaussian noise N(0, init_noise_scale²) to all parameters.

        Requires: num_repetitions=4, use_prior=True
        """
        with torch.no_grad():
            if not self.use_prior:
                raise ValueError("diverse_baselines requires use_prior=True")
            if self.num_repetitions != 4:
                raise ValueError(f"diverse_baselines requires num_repetitions=4, got {self.num_repetitions}")

            # Zero all parameters
            self.baseline_alphas.zero_()
            self.baseline_betas.zero_()
            self.fourier_alphas.zero_()
            self.fourier_betas.zero_()

            # ===== Repetition 0: uPU =====
            base_idx = 0

            # f_outer: identity
            self.baseline_alphas[base_idx + 0, 0] = 1.0

            # f_3: -π·log(x)
            self.baseline_betas[base_idx + 3, 1] = -1.0  # c₁ = -π
            self.baseline_alphas[base_idx + 3, 2] = -1.0  # a = -1

            # f_4: π·log(x)
            self.baseline_betas[base_idx + 4, 1] = 1.0  # c₁ = π
            self.baseline_alphas[base_idx + 4, 2] = -1.0  # a = -1

            # f_6: -log(x)
            self.baseline_alphas[base_idx + 6, 1] = -1.0  # c₁ = -1
            self.baseline_alphas[base_idx + 6, 2] = -1.0  # a = -1

            # ===== Repetition 1: PUDRa =====
            base_idx = 7

            # f_outer: identity
            self.baseline_alphas[base_idx + 0, 0] = 1.0

            # f_3: -π·log(x)
            self.baseline_betas[base_idx + 3, 1] = -1.0  # c₁ = -π
            self.baseline_alphas[base_idx + 3, 2] = -1.0  # a = -1

            # f_5: x (identity on p_oth)
            self.baseline_alphas[base_idx + 5, 0] = 1.0

            # ===== Repetition 2: VPU First Factor (log(E_all[p])) =====
            base_idx = 14

            # f_outer: log(x)
            self.baseline_alphas[base_idx + 0, 0] = 0.0  # c₀ = 0
            self.baseline_alphas[base_idx + 0, 1] = 1.0  # c₁ = 1
            self.baseline_alphas[base_idx + 0, 2] = -1.0  # a = -1

            # f_1: x (identity, so E_all[f_1(p)] = E_all[p])
            self.baseline_alphas[base_idx + 1, 0] = 1.0

            # ===== Repetition 3: VPU Second Factor (-E_P[log(p)]) =====
            base_idx = 21

            # f_outer: -x
            self.baseline_alphas[base_idx + 0, 0] = -1.0

            # f_3: log(x)
            self.baseline_alphas[base_idx + 3, 0] = 0.0  # c₀ = 0
            self.baseline_alphas[base_idx + 3, 1] = 1.0  # c₁ = 1
            self.baseline_alphas[base_idx + 3, 2] = -1.0  # a = -1

            # ===== Add Noise for Exploration =====
            if self.init_noise_scale > 0:
                self.baseline_alphas.add_(torch.randn_like(self.baseline_alphas) * self.init_noise_scale)
                self.baseline_betas.add_(torch.randn_like(self.baseline_betas) * self.init_noise_scale)
                self.fourier_alphas.add_(torch.randn_like(self.fourier_alphas) * self.init_noise_scale)
                self.fourier_betas.add_(torch.randn_like(self.fourier_betas) * self.init_noise_scale)

            print("✓ Initialized MonotonicBasisLoss with diverse baselines")
            print(f"  - Repetition 0: uPU (π·E_P[-log(p)] + E_U[-log(1-p)] - π·E_P[-log(1-p)])")
            print(f"  - Repetition 1: PUDRa (π·E_P[-log(p)] + E_U[p])")
            print(f"  - Repetition 2: VPU part 1 (log(E_all[p]))")
            print(f"  - Repetition 3: VPU part 2 (-E_P[log(p)])")
            print(f"  - Combined VPU: log(E_all[p]) - E_P[log(p)])")
            if self.init_noise_scale > 0:
                print(f"  - Added Gaussian noise N(0, {self.init_noise_scale}²) to all parameters")

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
        """Apply full basis function at index idx to input x.

        Computes f(x) = c₀·x + c₁·∫[1,x] g(t) dt where g is the integrand.
        The derivative is f'(x) = c₀ + c₁·g(x), handled automatically by PyTorch autograd.

        Args:
            x: Input tensor, shape [N] or [N, ...] or scalar
            idx: Basis function index

        Returns:
            Output tensor, same shape as x
        """
        baseline, fourier = self.get_basis_params(idx)

        # Extract individual baseline parameters
        c_0 = baseline[0]
        c_1 = baseline[1]
        a = baseline[2]
        b = baseline[3]
        c = baseline[4]
        d = baseline[5]
        e = baseline[6]
        g = baseline[7]
        h = baseline[8]
        t_0 = baseline[9]

        return monotonic_basis_full(
            x, c_0, c_1, a, b, c, d, e, g, h, t_0, fourier,
            num_integration_points=self.num_integration_points,
            chunk_size=self.integration_chunk_size
        )

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

    def compute_regularization(self) -> torch.Tensor:
        """Compute L1 (baseline) + L2 (Fourier) regularization.

        L1 on baseline parameters encourages sparsity (many params → 0).
        L2 on Fourier parameters prevents spectral coefficient explosion.

        Returns:
            Scalar regularization loss

        Examples:
            >>> loss_fn = MonotonicBasisLoss(l1_weight=1e-4, l2_weight=1e-3)
            >>> reg_loss = loss_fn.compute_regularization()
            >>> total_loss = task_loss + reg_loss
        """
        # Collect baseline and Fourier parameters
        if self.use_prior:
            # Regularize both alpha and beta for prior conditioning
            baseline_params = torch.cat([
                self.baseline_alphas.view(-1),
                self.baseline_betas.view(-1)
            ])
            fourier_params = torch.cat([
                self.fourier_alphas.view(-1),
                self.fourier_betas.view(-1)
            ])
        else:
            baseline_params = self.baseline_params.view(-1)
            fourier_params = self.fourier_params.view(-1)

        # L1 on baseline (sparsity)
        l1_reg = self.l1_weight * torch.abs(baseline_params).sum()

        # L2 on Fourier (stability)
        l2_reg = self.l2_weight * (fourier_params ** 2).sum()

        return l1_reg + l2_reg
