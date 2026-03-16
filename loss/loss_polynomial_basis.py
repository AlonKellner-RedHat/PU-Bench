"""Learnable Polynomial Basis Loss for PU Learning.

This module implements a simplified learnable loss function using elementary polynomials
instead of numerical integration. This provides faster computation with analytical gradients
while maintaining expressivity for common PU loss functions (BCE, uPU, PUDRa, VPU).

Mathematical Structure:
    L = Σ_{rep} f_outer(
        (f_1(p_all) + f_2(1-p_all)).mean() +
        (f_3(p_pos) + f_4(1-p_pos)).mean() +
        (f_5(p_oth) + f_6(1-p_oth)).mean()
    )

    where each f is a polynomial: f(x) = a₁ + a₂·x + a₃/x + a₄·x² + a₅/x² + a₆·log(x) + a₇·exp(x)

Key Features:
    - Faster than MonotonicBasisLoss (5-10× speedup, no numerical integration)
    - Fewer parameters (294 vs 1,456 for 3 repetitions)
    - Analytical gradients (pure autograd, no integration grids)
    - Same interface as MonotonicBasisLoss (drop-in replacement)
    - Still expressive enough for standard PU losses

For meta-learning, loss parameters are optimized across tasks.
For standard PU learning, only model parameters are optimized (loss parameters frozen).
"""

import torch
import torch.nn as nn
from typing import Optional


class PolynomialBasisLoss(nn.Module):
    """Learnable loss function using polynomial basis transformations.

    This loss has a hierarchical structure with multiple repetitions:
    - Each repetition has 7 polynomial functions (1 outer + 6 inner)
    - Inner functions transform p_all, p_pos, and p_oth
    - Outer function transforms the sum of inner means

    Attributes:
        num_repetitions: Number of repetition blocks (default 3)
        use_prior: Whether to condition parameters on class prior
        prior: Class prior π (only used if use_prior=True)
        oracle_mode: If True, expects binary labels (0/1); if False, PU labels (±1)

    Parameters:
        If use_prior=False:
            - coefficients: shape [num_repetitions * 7, 7]  (a₁, a₂, a₃, a₄, a₅, a₆, a₇)

        If use_prior=True:
            - coefficient_alphas: shape [num_repetitions * 7, 7]
            - coefficient_betas: shape [num_repetitions * 7, 7]
            Then: aᵢ = αᵢ + βᵢ * prior

    Examples:
        >>> # Standard PU mode without prior
        >>> loss = PolynomialBasisLoss(num_repetitions=3, use_prior=False, oracle_mode=False)
        >>> outputs = torch.randn(100)  # Model logits
        >>> labels = torch.randint(0, 2, (100,)) * 2 - 1  # PU labels: {-1, 1}
        >>> loss_val = loss(outputs, labels)

        >>> # Oracle mode with prior conditioning
        >>> loss = PolynomialBasisLoss(
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
        use_prior: bool = True,
        prior: float = 0.5,
        oracle_mode: bool = False,
        init_scale: float = 0.01,
        init_mode: str = 'random',
        init_noise_scale: float = 0.0,
        l1_weight: float = 1e-4,
        l2_weight: float = 1e-3,
    ):
        """Initialize learnable polynomial basis loss.

        Args:
            num_repetitions: Number of repetition blocks
            use_prior: If True, parameters are linear functions of prior
            prior: Class prior value (only used if use_prior=True)
            oracle_mode: If True, expects binary labels; if False, PU labels
            init_scale: Initialization scale for random parameters
            init_mode: Initialization mode ('random', 'bce_equivalent', 'zeros', 'diverse_baselines')
            init_noise_scale: Noise added to diverse_baselines initialization
            l1_weight: L1 regularization weight (for constant, inverse, log terms)
            l2_weight: L2 regularization weight (for polynomial terms, 10× for exp)
        """
        super().__init__()

        self.num_repetitions = num_repetitions
        self.use_prior = use_prior
        self.oracle_mode = oracle_mode
        self.init_scale = init_scale
        self.init_mode = init_mode
        self.init_noise_scale = init_noise_scale
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.name = "polynomial_basis"

        # Total number of basis functions
        # Each repetition has 7 functions: 1 outer + 6 inner
        self.total_basis_funcs = num_repetitions * 7

        # Number of coefficients per function (7-term polynomial)
        num_coeffs = 7

        if use_prior:
            # Prior-conditioned: aᵢ = αᵢ + βᵢ * prior
            self.coefficient_alphas = nn.Parameter(
                torch.randn(self.total_basis_funcs, num_coeffs) * init_scale
            )
            self.coefficient_betas = nn.Parameter(
                torch.randn(self.total_basis_funcs, num_coeffs) * init_scale
            )

            # Store prior as buffer (not trainable, but saved in state_dict)
            self.register_buffer(
                "prior_tensor", torch.tensor(prior, dtype=torch.float32)
            )
        else:
            # Direct parameters
            self.coefficients = nn.Parameter(
                torch.randn(self.total_basis_funcs, num_coeffs) * init_scale
            )

            self.register_buffer("prior_tensor", torch.tensor(0.0, dtype=torch.float32))

        # Initialize parameters based on init_mode
        if init_mode == 'bce_equivalent':
            self._initialize_params_bce_equivalent()
        elif init_mode == 'zeros':
            self._initialize_params_zeros()
        elif init_mode == 'diverse_baselines':
            self._initialize_params_diverse_baselines()
        elif init_mode == 'bce_plus_trinary':
            self._initialize_params_bce_plus_trinary()
        else:
            # Random initialization already done above
            pass

    def get_coefficients(self, idx: int) -> torch.Tensor:
        """Get effective coefficients for a specific basis function.

        Args:
            idx: Index of the basis function (0 to total_basis_funcs-1)

        Returns:
            Tensor of shape [7] with effective coefficients
        """
        if self.use_prior:
            return self.coefficient_alphas[idx] + self.coefficient_betas[idx] * self.prior_tensor
        else:
            return self.coefficients[idx]

    def _apply_inner_polynomial(self, x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        """Apply full 7-term polynomial for inner functions.

        f(x) = a₁ + a₂·x + a₃/x + a₄·x² + a₅/x² + a₆·log(x) + a₇·exp(x)

        Args:
            x: Input tensor (probabilities in [0, 1])
            coeffs: Coefficient tensor [7]

        Returns:
            Transformed values
        """
        a1, a2, a3, a4, a5, a6, a7 = coeffs

        # Epsilon for numerical stability
        eps = 1e-6
        max_value = 1e3  # Maximum allowed intermediate value

        # Clamp x for numerical stability
        x_safe = torch.clamp(x, min=eps, max=1.0 - eps)

        # Compute polynomial terms with gradient-safe operations
        result = torch.clamp(a1, min=-max_value, max=max_value)  # Constant

        # Linear term
        linear = a2 * x_safe
        result = result + torch.clamp(linear, min=-max_value, max=max_value)

        # Inverse term (clamp coefficient to prevent explosion)
        a3_safe = torch.clamp(a3, min=-max_value, max=max_value)
        inverse = a3_safe / x_safe
        result = result + torch.clamp(inverse, min=-max_value, max=max_value)

        # Quadratic term
        quadratic = a4 * (x_safe ** 2)
        result = result + torch.clamp(quadratic, min=-max_value, max=max_value)

        # Inverse quadratic term (clamp coefficient)
        a5_safe = torch.clamp(a5, min=-max_value, max=max_value)
        inv_quadratic = a5_safe / (x_safe ** 2)
        result = result + torch.clamp(inv_quadratic, min=-max_value, max=max_value)

        # Logarithm term (use max(eps, x) to ensure valid input)
        a6_safe = torch.clamp(a6, min=-max_value, max=max_value)
        log_input = torch.maximum(x_safe, torch.tensor(eps, device=x.device))
        log_term = a6_safe * torch.log(log_input)
        result = result + torch.clamp(log_term, min=-max_value, max=max_value)

        # Exponential with strict clamping (reduced range for safety)
        a7_safe = torch.clamp(a7, min=-10, max=10)  # Clamp coefficient
        x_exp = torch.clamp(x_safe, min=eps, max=5.0)  # Reduced exp range
        exp_term = a7_safe * torch.exp(x_exp - 3.0)  # Shift to prevent explosion
        result = result + torch.clamp(exp_term, min=-max_value, max=max_value)

        # Final result clamping
        result = torch.clamp(result, min=-max_value, max=max_value)

        return result

    def _apply_outer_polynomial(self, x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        """Apply restricted polynomial for outer function (only const + linear + log).

        f(x) = a₁ + a₂·x + a₆·log(x)

        Restricted form prevents numerical instability from nested inverse/exp terms.

        Args:
            x: Input tensor (aggregated inner values)
            coeffs: Coefficient tensor [7]

        Returns:
            Transformed values
        """
        a1, a2, _, _, _, a6, _ = coeffs

        # Epsilon for numerical stability
        eps = 1e-6
        max_value = 1e3

        # Clamp for log stability and prevent extreme values
        x_safe = torch.clamp(x, min=eps, max=max_value)

        # Compute restricted polynomial with gradient safety
        result = torch.clamp(a1, min=-max_value, max=max_value)  # Constant

        # Linear term
        a2_safe = torch.clamp(a2, min=-max_value, max=max_value)
        linear = a2_safe * x_safe
        result = result + torch.clamp(linear, min=-max_value, max=max_value)

        # Logarithm term (use max(eps, x) to ensure valid input)
        a6_safe = torch.clamp(a6, min=-max_value, max=max_value)
        log_input = torch.maximum(x_safe, torch.tensor(eps, device=x.device))
        log_term = a6_safe * torch.log(log_input)
        result = result + torch.clamp(log_term, min=-max_value, max=max_value)

        # Final clamping
        result = torch.clamp(result, min=-max_value, max=max_value)

        return result

    def apply_polynomial_basis(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        """Apply polynomial basis function.

        Outer functions (idx % 7 == 0) use restricted form.
        Inner functions use full 7-term polynomial.

        Args:
            x: Input tensor
            idx: Function index

        Returns:
            Transformed tensor
        """
        coeffs = self.get_coefficients(idx)

        # Outer functions at indices 0, 7, 14, 21, ...
        if idx % 7 == 0:
            return self._apply_outer_polynomial(x, coeffs)
        else:
            return self._apply_inner_polynomial(x, coeffs)

    def forward(
        self,
        outputs: torch.Tensor,
        pu_labels: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute loss value.

        Args:
            outputs: Model logits, shape [batch_size]
            pu_labels: PU labels (1=positive, -1=unlabeled) or binary (1/0 if oracle_mode)
            weights: Optional sample weights (not used)

        Returns:
            Scalar loss value
        """
        # Apply sigmoid to get probabilities
        p = torch.sigmoid(outputs.view(-1))
        pu_labels = pu_labels.view(-1)

        # Separate samples into groups
        if self.oracle_mode:
            # Binary labels: 1 = positive, 0 = negative
            pos_mask = pu_labels == 1
            oth_mask = pu_labels == 0
        else:
            # PU labels: 1 = positive, -1 = unlabeled
            pos_mask = pu_labels == 1
            oth_mask = pu_labels == -1

        # Get probability groups
        p_all = p  # All samples
        p_pos = p[pos_mask]  # Positive samples
        p_oth = p[oth_mask]  # Other samples (unlabeled or negative)

        # Handle edge case: no positive samples
        if len(p_pos) == 0:
            return torch.tensor(0.0, device=p.device, requires_grad=True)

        # Handle edge case: no other samples
        if len(p_oth) == 0:
            p_oth = torch.zeros(1, device=p.device)

        # Compute loss across all repetitions
        total_loss = torch.tensor(0.0, device=p.device)

        for rep in range(self.num_repetitions):
            base_idx = rep * 7

            # Compute inner terms
            # Term 1: E_all[f_1(p) + f_2(1-p)]
            term1_vals = (
                self.apply_polynomial_basis(p_all, base_idx + 1) +
                self.apply_polynomial_basis(1.0 - p_all, base_idx + 2)
            )
            term1 = term1_vals.mean()

            # Term 2: E_pos[f_3(p) + f_4(1-p)]
            term2_vals = (
                self.apply_polynomial_basis(p_pos, base_idx + 3) +
                self.apply_polynomial_basis(1.0 - p_pos, base_idx + 4)
            )
            term2 = term2_vals.mean()

            # Term 3: E_oth[f_5(p) + f_6(1-p)]
            term3_vals = (
                self.apply_polynomial_basis(p_oth, base_idx + 5) +
                self.apply_polynomial_basis(1.0 - p_oth, base_idx + 6)
            )
            term3 = term3_vals.mean()

            # Sum inner terms with safety clamping
            inner_sum = term1 + term2 + term3
            inner_sum = torch.clamp(inner_sum, min=-1e3, max=1e3)

            # Apply outer function
            inner_sum_tensor = inner_sum.view(1)
            outer_result = self.apply_polynomial_basis(inner_sum_tensor, base_idx)

            # Add to total loss with safety clamping
            outer_result_safe = torch.clamp(outer_result.squeeze(), min=-1e3, max=1e3)
            total_loss = total_loss + outer_result_safe

        # Final safety check for NaN/inf
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            # Return a large but finite loss to allow gradient computation
            return torch.tensor(1e3, device=total_loss.device, requires_grad=True)

        # Final clamping
        total_loss = torch.clamp(total_loss, min=-1e3, max=1e3)

        return total_loss

    def set_prior(self, new_prior: float):
        """Update prior without changing learned (alpha, beta) coefficients.

        This enables transfer to new datasets with different priors.

        Args:
            new_prior: New class prior value

        Raises:
            ValueError: If use_prior=False
        """
        if not self.use_prior:
            raise ValueError("Cannot set prior when use_prior=False")

        self.prior_tensor.fill_(new_prior)

    def get_num_parameters(self) -> int:
        """Get total number of learnable parameters.

        Returns:
            Total number of parameters
        """
        return sum(p.numel() for p in self.parameters())

    def get_parameter_summary(self) -> dict:
        """Get summary of parameter counts and configuration.

        Returns:
            Dictionary with parameter counts and configuration
        """
        if self.use_prior:
            coeff_count = self.coefficient_alphas.numel() + self.coefficient_betas.numel()
        else:
            coeff_count = self.coefficients.numel()

        return {
            "total_params": self.get_num_parameters(),
            "coefficient_params": coeff_count,
            "num_basis_functions": self.total_basis_funcs,
            "num_repetitions": self.num_repetitions,
            "use_prior": self.use_prior,
            "coeffs_per_function": 7,
        }

    def compute_regularization(self) -> torch.Tensor:
        """Compute differentiated regularization by coefficient type.

        L1 on indices [0, 2, 4, 5]: constant, inverse terms, log
        L2 on indices [1, 3]: polynomial terms (linear, quadratic)
        Strong L2 (10×) on index 6: exponential term

        Returns:
            Scalar regularization loss
        """
        # Get all coefficients
        if self.use_prior:
            coeffs = torch.cat([
                self.coefficient_alphas.view(-1),
                self.coefficient_betas.view(-1)
            ])
        else:
            coeffs = self.coefficients.view(-1)

        # Reshape to [num_funcs, 7] to separate by coefficient type
        num_funcs = self.total_basis_funcs
        if self.use_prior:
            # Stack alpha and beta
            coeffs_reshaped = torch.cat([
                self.coefficient_alphas,  # [num_funcs, 7]
                self.coefficient_betas,   # [num_funcs, 7]
            ], dim=0)  # [2*num_funcs, 7]
        else:
            coeffs_reshaped = self.coefficients  # [num_funcs, 7]

        # L1 regularization on indices [0, 2, 4, 5] (constant, inverses, log)
        l1_indices = [0, 2, 4, 5]
        l1_reg = self.l1_weight * torch.abs(
            coeffs_reshaped[:, l1_indices]
        ).sum()

        # L2 regularization on indices [1, 3] (linear, quadratic)
        l2_indices = [1, 3]
        l2_reg = self.l2_weight * (
            coeffs_reshaped[:, l2_indices] ** 2
        ).sum()

        # Strong L2 on index 6 (exponential) - 10× weight
        exp_reg = (10 * self.l2_weight) * (
            coeffs_reshaped[:, 6] ** 2
        ).sum()

        return l1_reg + l2_reg + exp_reg

    def _initialize_params_bce_equivalent(self):
        """Initialize to approximate BCE.

        BCE: L = -E_pos[log(p)] - E_neg[log(1-p)]

        Polynomial approximation:
        - f_outer(x) = x  → α₂ = 1 (linear term)
        - f_3(p_pos) = -log(p) → α₆ = -1 (log term)
        - f_6(p_oth) = -log(p) → α₆ = -1 (log term)
        - All other functions = 0
        """
        with torch.no_grad():
            if self.use_prior:
                # Zero all parameters
                self.coefficient_alphas.zero_()
                self.coefficient_betas.zero_()

                # For each repetition
                for rep in range(self.num_repetitions):
                    base_idx = rep * 7

                    # f_outer: identity (a₂ = 1)
                    # Index 1 is the linear term
                    self.coefficient_alphas[base_idx, 1] = 1.0

                    # f_3: -log(p) for positive samples
                    # Index 5 is the log term (a₆)
                    self.coefficient_alphas[base_idx + 3, 5] = -1.0

                    # f_6: -log(p) for other/unlabeled samples
                    self.coefficient_alphas[base_idx + 6, 5] = -1.0

            else:
                # Zero all parameters
                self.coefficients.zero_()

                # For each repetition
                for rep in range(self.num_repetitions):
                    base_idx = rep * 7

                    # f_outer: identity
                    self.coefficients[base_idx, 1] = 1.0

                    # f_3: -log(p)
                    self.coefficients[base_idx + 3, 5] = -1.0

                    # f_6: -log(p)
                    self.coefficients[base_idx + 6, 5] = -1.0

            # Add noise for exploration if specified (positive noise only via abs)
            if self.init_noise_scale > 0:
                if self.use_prior:
                    noise_alphas = torch.randn_like(self.coefficient_alphas).abs() * self.init_noise_scale
                    noise_betas = torch.randn_like(self.coefficient_betas).abs() * self.init_noise_scale
                    self.coefficient_alphas.add_(noise_alphas)
                    self.coefficient_betas.add_(noise_betas)
                else:
                    noise = torch.randn_like(self.coefficients).abs() * self.init_noise_scale
                    self.coefficients.add_(noise)
                print(f"✓ Added positive noise |N(0, {self.init_noise_scale}²)| to BCE initialization")

    def _initialize_params_zeros(self):
        """Initialize all parameters to zeros (for testing)."""
        with torch.no_grad():
            if self.use_prior:
                self.coefficient_alphas.zero_()
                self.coefficient_betas.zero_()
            else:
                self.coefficients.zero_()

    def _initialize_params_diverse_baselines(self):
        """Initialize with uPU + PUDRa + VPU baselines + noise.

        Requires: num_repetitions=4, use_prior=True

        Repetition 0: uPU (unbiased PU with prior factor)
        Repetition 1: PUDRa (density ratio)
        Repetition 2: VPU first factor - log(E_all[p])
        Repetition 3: VPU second factor - E_P[log(p)] (negated)

        Then adds Gaussian noise N(0, init_noise_scale²) to all parameters.

        Polynomial basis: f(x) = a₁ + a₂·x + a₃/x + a₄·x² + a₅/x² + a₆·log(x) + a₇·exp(x)
        Indices: [0=const, 1=linear, 2=inv, 3=quad, 4=inv_quad, 5=log, 6=exp]
        """
        with torch.no_grad():
            if not self.use_prior:
                raise ValueError("diverse_baselines requires use_prior=True")
            if self.num_repetitions != 4:
                raise ValueError(f"diverse_baselines requires num_repetitions=4, got {self.num_repetitions}")

            # Zero all parameters
            self.coefficient_alphas.zero_()
            self.coefficient_betas.zero_()

            # ===== Repetition 0: uPU =====
            # L_uPU = π·E_P[-log(p)] + E_U[-log(1-p)] - π·E_P[-log(1-p)]
            base_idx = 0

            # f_outer: identity (a₂ = 1)
            self.coefficient_alphas[base_idx + 0, 1] = 1.0

            # f_3: -π·log(p) on positive samples
            # With prior: a₆ = α₆ + β₆·π = 0 + (-1)·π = -π
            self.coefficient_betas[base_idx + 3, 5] = -1.0

            # f_4: π·log(1-p) on positive samples
            # Approximation: log(1-p) ≈ -p for small p, but we use log directly
            # Actually for (1-p), we need -log(1-p) ≈ -log(1) + p = p for small p
            # But in structure: f_4 processes (1-p), so we want π·log(1-p)
            # Since the structure computes f_4(1-p), we set: a₆ = π for log term
            self.coefficient_betas[base_idx + 4, 5] = 1.0

            # f_6: -log(p) on unlabeled samples
            # This is the E_U[-log(1-p)] term, but f_6 operates on p values
            # We want -log(1-p) which is processed as f_6(1-p) in the structure
            # So we set: a₆ = -1 for log term
            self.coefficient_alphas[base_idx + 6, 5] = -1.0

            # ===== Repetition 1: PUDRa =====
            # L_PUDRa = π·E_P[-log(p)] + E_U[p]
            base_idx = 7

            # f_outer: identity
            self.coefficient_alphas[base_idx + 0, 1] = 1.0

            # f_3: -π·log(p)
            self.coefficient_betas[base_idx + 3, 5] = -1.0

            # f_5: p (identity on p_oth/unlabeled)
            self.coefficient_alphas[base_idx + 5, 1] = 1.0

            # ===== Repetition 2: VPU First Factor (log(E_all[p])) =====
            base_idx = 14

            # f_outer: log(x) - we use a₆ = 1 for log term
            self.coefficient_alphas[base_idx + 0, 5] = 1.0

            # f_1: p (identity on E_all)
            self.coefficient_alphas[base_idx + 1, 1] = 1.0

            # ===== Repetition 3: VPU Second Factor (-E_P[log(p)]) =====
            base_idx = 21

            # f_outer: -x (negative identity)
            self.coefficient_alphas[base_idx + 0, 1] = -1.0

            # f_3: log(p)
            self.coefficient_alphas[base_idx + 3, 5] = 1.0

            # ===== Add Noise for Exploration =====
            if self.init_noise_scale > 0:
                self.coefficient_alphas.add_(torch.randn_like(self.coefficient_alphas) * self.init_noise_scale)
                self.coefficient_betas.add_(torch.randn_like(self.coefficient_betas) * self.init_noise_scale)

            print("✓ Initialized PolynomialBasisLoss with diverse baselines")
            print(f"  - Repetition 0: uPU (π·E_P[-log(p)] + E_U[-log(1-p)] - π·E_P[-log(1-p)])")
            print(f"  - Repetition 1: PUDRa (π·E_P[-log(p)] + E_U[p])")
            print(f"  - Repetition 2: VPU part 1 (log(E_all[p]))")
            print(f"  - Repetition 3: VPU part 2 (-E_P[log(p)])")
            print(f"  - Combined VPU: log(E_all[p]) - E_P[log(p)])")
            if self.init_noise_scale > 0:
                print(f"  - Added Gaussian noise N(0, {self.init_noise_scale}²) to all parameters")

    def _initialize_params_bce_plus_trinary(self):
        """Initialize first repetition as BCE, remaining as random trinary {-1, 0, 1}.

        Repetition 0: Exact BCE (like bce_equivalent)
        Repetitions 1-3: Random trinary values {-1, 0, 1}

        This provides a strong BCE baseline plus random exploration.
        """
        with torch.no_grad():
            if self.use_prior:
                # Zero all parameters
                self.coefficient_alphas.zero_()
                self.coefficient_betas.zero_()

                # Repetition 0: BCE initialization
                base_idx = 0
                # f_outer: identity (a₂ = 1)
                self.coefficient_alphas[base_idx, 1] = 1.0
                # f_3: -log(p) for positive samples
                self.coefficient_alphas[base_idx + 3, 5] = -1.0
                # f_6: -log(p) for other/unlabeled samples
                self.coefficient_alphas[base_idx + 6, 5] = -1.0

                # Repetitions 1-3: Random trinary {-1, 0, 1}
                for rep in range(1, self.num_repetitions):
                    base_idx = rep * 7
                    # Sample trinary values for all 7 functions × 7 coefficients
                    for func_offset in range(7):
                        func_idx = base_idx + func_offset
                        # Random choice from {-1, 0, 1} for alphas
                        self.coefficient_alphas[func_idx] = torch.randint(
                            -1, 2, (7,), dtype=torch.float32
                        )
                        # Random choice from {-1, 0, 1} for betas
                        self.coefficient_betas[func_idx] = torch.randint(
                            -1, 2, (7,), dtype=torch.float32
                        )

            else:
                # Zero all parameters
                self.coefficients.zero_()

                # Repetition 0: BCE initialization
                self.coefficients[0, 1] = 1.0  # f_outer: identity
                self.coefficients[3, 5] = -1.0  # f_3: -log(p)
                self.coefficients[6, 5] = -1.0  # f_6: -log(p)

                # Repetitions 1-3: Random trinary {-1, 0, 1}
                for rep in range(1, self.num_repetitions):
                    base_idx = rep * 7
                    for func_offset in range(7):
                        func_idx = base_idx + func_offset
                        self.coefficients[func_idx] = torch.randint(
                            -1, 2, (7,), dtype=torch.float32
                        )

            print("✓ Initialized PolynomialBasisLoss with BCE + random trinary")
            print(f"  - Repetition 0: Exact BCE")
            print(f"  - Repetitions 1-{self.num_repetitions-1}: Random trinary values {{-1, 0, 1}}")
