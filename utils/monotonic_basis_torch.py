"""PyTorch implementation of Universal Monotonic Basis integrand.

This module provides a differentiable version of the monotonic basis integrand
for use in learnable loss functions. Unlike the NumPy version in monotonic_basis.py
which uses scipy integration, this version computes only the integrand:

    g(x; θ) = exp(a·log(x) + b + c·x + d·x² + e·exp(x) +
                  g·σ'(h·(x-t₀)) + Σ dₖ·cos(2πk·log(x)))

This is fully differentiable and can be used with PyTorch autograd.

Key differences from monotonic_basis.py:
- No integration, just the integrand (exp of the exponent)
- PyTorch tensors instead of NumPy arrays
- Fully differentiable for autograd
- Vectorized for batch processing
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


def monotonic_basis_integrand(
    x: torch.Tensor,
    c_0: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    e: torch.Tensor,
    g: torch.Tensor,
    h: torch.Tensor,
    t_0: torch.Tensor,
    d_k: torch.Tensor,
) -> torch.Tensor:
    """Compute monotonic basis integrand (vectorized, differentiable).

    Args:
        x: Input values, shape [...] (any shape)
        c_0: Linear coefficient (not used in integrand, only in full basis)
        a: Log-power coefficient
        b: Constant offset
        c: Linear coefficient
        d: Quadratic coefficient
        e: Exponential coefficient
        g: Sigmoid derivative amplitude
        h: Sigmoid steepness
        t_0: Sigmoid center
        d_k: Fourier coefficients, shape [K]

    Returns:
        Integrand values, shape [...] (same as x)

    Note:
        All inputs are tensors to support autograd.
        Returns exp(exponent) which is always positive, guaranteeing monotonicity.
    """
    # Clamp x to avoid log(0) and exp overflow
    x_safe = torch.clamp(x, min=1e-8, max=1e8)

    # Log-power term: a·log(x)
    log_term = a * torch.log(x_safe)

    # Polynomial terms: b + c·x + d·x²
    poly_term = b + c * x + d * x**2

    # Exponential term: e·exp(x)
    # Clamp x for exp to avoid overflow
    x_exp_safe = torch.clamp(x, min=-10, max=10)
    exp_term = e * torch.exp(x_exp_safe)

    # Sigmoid derivative: g·σ'(h·(x-t₀))
    # where σ'(z) = σ(z)·(1-σ(z))
    z = h * (x - t_0)
    sigmoid_val = torch.sigmoid(z)
    sigmoid_deriv = sigmoid_val * (1 - sigmoid_val)
    sigmoid_term = g * sigmoid_deriv

    # Spectral (Fourier) terms: Σ dₖ·cos(2πk·log(x))
    # d_k shape: [K]
    # Need to compute for k=1..K
    spectral_term = torch.zeros_like(x)
    K = d_k.shape[0]
    log_x_safe = torch.log(x_safe)

    for k in range(K):
        k_val = k + 1  # k starts from 1
        spectral_term = spectral_term + d_k[k] * torch.cos(
            2 * np.pi * k_val * log_x_safe
        )

    # Sum all terms in exponent
    exponent = log_term + poly_term + exp_term + sigmoid_term + spectral_term

    # Clamp exponent to prevent overflow/underflow
    exponent_safe = torch.clamp(exponent, min=-50, max=50)

    # Return exp(exponent) - always positive
    return torch.exp(exponent_safe)


class MonotonicBasisIntegrand(nn.Module):
    """Wrapper module for monotonic basis integrand with parameter storage.

    This is a utility class that stores parameters and applies the integrand.
    Useful for debugging and testing.

    Examples:
        >>> # Non-learnable version
        >>> integrand = MonotonicBasisIntegrand(a=1.0, b=0.5, learnable=False)
        >>> x = torch.tensor([0.3, 0.5, 0.7])
        >>> y = integrand(x)
        >>> assert torch.all(y > 0)  # Always positive

        >>> # Learnable version
        >>> integrand = MonotonicBasisIntegrand(learnable=True)
        >>> optimizer = torch.optim.Adam(integrand.parameters(), lr=0.01)
        >>> # Train integrand...
    """

    def __init__(
        self,
        c_0: float = 0.0,
        a: float = 0.0,
        b: float = 0.0,
        c: float = 0.0,
        d: float = 0.0,
        e: float = 0.0,
        g: float = 0.0,
        h: float = 1.0,
        t_0: float = 0.5,
        d_k: Optional[list] = None,
        learnable: bool = False,
    ):
        super().__init__()

        if d_k is None:
            d_k = [0.0] * 5

        # Store as parameters or buffers
        if learnable:
            self.c_0 = nn.Parameter(torch.tensor(c_0, dtype=torch.float32))
            self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
            self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))
            self.c = nn.Parameter(torch.tensor(c, dtype=torch.float32))
            self.d = nn.Parameter(torch.tensor(d, dtype=torch.float32))
            self.e = nn.Parameter(torch.tensor(e, dtype=torch.float32))
            self.g = nn.Parameter(torch.tensor(g, dtype=torch.float32))
            self.h = nn.Parameter(torch.tensor(h, dtype=torch.float32))
            self.t_0 = nn.Parameter(torch.tensor(t_0, dtype=torch.float32))
            self.d_k = nn.Parameter(torch.tensor(d_k, dtype=torch.float32))
        else:
            self.register_buffer("c_0", torch.tensor(c_0, dtype=torch.float32))
            self.register_buffer("a", torch.tensor(a, dtype=torch.float32))
            self.register_buffer("b", torch.tensor(b, dtype=torch.float32))
            self.register_buffer("c", torch.tensor(c, dtype=torch.float32))
            self.register_buffer("d", torch.tensor(d, dtype=torch.float32))
            self.register_buffer("e", torch.tensor(e, dtype=torch.float32))
            self.register_buffer("g", torch.tensor(g, dtype=torch.float32))
            self.register_buffer("h", torch.tensor(h, dtype=torch.float32))
            self.register_buffer("t_0", torch.tensor(t_0, dtype=torch.float32))
            self.register_buffer("d_k", torch.tensor(d_k, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply integrand to input tensor.

        Args:
            x: Input tensor, any shape

        Returns:
            Output tensor, same shape as x
        """
        return monotonic_basis_integrand(
            x,
            self.c_0,
            self.a,
            self.b,
            self.c,
            self.d,
            self.e,
            self.g,
            self.h,
            self.t_0,
            self.d_k,
        )
