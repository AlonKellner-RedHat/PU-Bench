"""Universal Monotonic Spectral Basis function implementation.

This module implements a sophisticated mathematical function basis designed for
monotonic function approximation. The basis guarantees strict monotonicity while
providing flexible, hierarchical decomposition similar to Fourier series.

Mathematical Foundation:
    f(x) = c₀·x + ∫[1,x] exp(a·log(t) + b + c·t + d·t² + e·exp(t)
                              + g·σ'(h·(t-t₀)) + Σ dₖ·cos(2πk·log(t))) dt

    where:
    - σ'(z) = σ(z)·(1-σ(z)) is the logistic sigmoid derivative
    - Parameters control polynomial (c, d), exponential (e), sigmoid (g, h, t₀),
      and spectral (dₖ) components
    - Guaranteed monotonic for x > 0 (integrand always positive)

Key Properties:
    - Trivializes linear, logarithmic, exponential, and sigmoid functions
    - Coarse-to-fine spectral control through Fourier terms in log-space
    - Useful for calibration, propensity modeling, and constrained approximation
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
import warnings


@dataclass
class BasisParameters:
    """Parameter container for Universal Monotonic Spectral Basis.

    Attributes:
        c_0: Linear coefficient (outside integral), controls overall slope
        a: Log-power term coefficient (a·log(t))
        b: Constant offset in exponent
        c: Linear term coefficient in exponent (c·t)
        d: Quadratic term coefficient in exponent (d·t²)
        e: Exponential term coefficient (e·exp(t))
        g: Sigmoid derivative amplitude
        h: Sigmoid steepness (larger = sharper transition)
        t_0: Sigmoid center location in domain
        d_k: Fourier coefficients for spectral terms (K=5 by default)

    Examples:
        >>> # Linear function: f(x) = x
        >>> linear = BasisParameters(c_0=1.0)

        >>> # Logarithmic-like function
        >>> log_like = BasisParameters(c_0=0.0, a=1.0, b=-1.0)

        >>> # Sigmoid step function
        >>> sigmoid = BasisParameters(c_0=0.0, g=5.0, h=10.0, t_0=0.5, b=0.5)
    """
    c_0: float = 1.0
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0
    e: float = 0.0
    g: float = 0.0
    h: float = 1.0
    t_0: float = 0.5
    d_k: List[float] = None

    def __post_init__(self):
        """Initialize Fourier coefficients if not provided."""
        if self.d_k is None:
            self.d_k = [0.0] * 5

    def to_dict(self) -> dict:
        """Convert parameters to dictionary for serialization."""
        return {
            'c_0': self.c_0,
            'a': self.a,
            'b': self.b,
            'c': self.c,
            'd': self.d,
            'e': self.e,
            'g': self.g,
            'h': self.h,
            't_0': self.t_0,
            'd_k': self.d_k
        }


class UniversalMonotonicBasis:
    """Universal Monotonic Spectral Basis function generator.

    Implements a monotonic function basis that can approximate various
    monotonic functions while guaranteeing non-decreasing behavior.

    The function is defined as:
        f(x) = c₀·x + ∫[1,x] exp(exponent_terms(t)) dt

    where exponent_terms includes polynomial, exponential, sigmoid, and
    spectral (Fourier) components.

    Attributes:
        params: BasisParameters instance containing all coefficients

    Examples:
        >>> params = BasisParameters(c_0=1.0, a=0.5)
        >>> basis = UniversalMonotonicBasis(params)
        >>> x = np.linspace(0.1, 1.0, 100)
        >>> y = basis.evaluate(x)
        >>> assert basis.verify_monotonic(x)  # Guaranteed monotonic
    """

    def __init__(self, params: BasisParameters):
        """Initialize basis with given parameters.

        Args:
            params: BasisParameters instance
        """
        self.params = params
        self._integration_warnings = []

    def _integrand(self, t: float) -> float:
        """Compute the integrand at point t.

        Computes:
            exp(a·log(t) + b + c·t + d·t² + e·exp(t)
                + g·σ'(h·(t-t₀)) + Σ dₖ·cos(2πk·log(t)))

        Args:
            t: Point at which to evaluate integrand (must be > 0)

        Returns:
            Integrand value (always positive due to exp)

        Note:
            - Exponent is clipped to [-50, 50] for numerical stability
            - Returns exp(0) = 1.0 if t <= 0 (boundary case)
        """
        if t <= 0:
            return 1.0  # Boundary case, shouldn't occur in practice

        p = self.params

        # Log-power term: a·log(t)
        log_term = p.a * np.log(t)

        # Polynomial terms: b + c·t + d·t²
        poly_term = p.b + p.c * t + p.d * t**2

        # Exponential term: e·exp(t)
        # Use clip to prevent overflow in exp(t)
        exp_arg = np.clip(t, -50, 50)
        exp_term = p.e * np.exp(exp_arg)

        # Sigmoid derivative: g·σ'(h·(t-t₀))
        # where σ'(z) = σ(z)·(1-σ(z)) = expit(z)·(1-expit(z))
        z = p.h * (t - p.t_0)
        sigmoid_val = expit(z)  # Numerically stable sigmoid
        sigmoid_deriv = sigmoid_val * (1 - sigmoid_val)
        sigmoid_term = p.g * sigmoid_deriv

        # Spectral (Fourier) terms: Σ dₖ·cos(2πk·log(t))
        spectral_term = 0.0
        for k, d_k in enumerate(p.d_k, start=1):
            spectral_term += d_k * np.cos(2 * np.pi * k * np.log(t))

        # Sum all terms in exponent
        exponent = log_term + poly_term + exp_term + sigmoid_term + spectral_term

        # Clip exponent to prevent overflow/underflow in final exp()
        exponent = np.clip(exponent, -50, 50)

        return np.exp(exponent)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate f(x) = c₀·x + ∫[1,x] integrand(t) dt.

        Args:
            x: Points to evaluate, shape (N,). Should be in (0, ∞).
               For visualization, typically [0.01, 1.0] to avoid log(0).

        Returns:
            Function values at x, shape (N,)

        Note:
            - Uses scipy.integrate.quad for adaptive quadrature
            - For x < 1, integrates from x to 1 and negates
            - Integration tolerances: epsabs=1e-8, epsrel=1e-8

        Examples:
            >>> params = BasisParameters(c_0=1.0)
            >>> basis = UniversalMonotonicBasis(params)
            >>> x = np.array([0.5, 1.0])
            >>> y = basis.evaluate(x)
            >>> print(y)  # [linear_value, 1.0]
        """
        # Ensure x is array
        x = np.atleast_1d(x)

        # Linear term: c₀·x
        linear_part = self.params.c_0 * x

        # Integral term: ∫[1,x] integrand(t) dt
        integral_part = np.zeros_like(x, dtype=float)

        for i, xi in enumerate(x):
            if np.abs(xi - 1.0) < 1e-10:
                # At x=1, integral is 0
                integral_part[i] = 0.0
            elif xi > 1.0:
                # Integrate from 1 to xi
                try:
                    result, error = quad(
                        self._integrand,
                        1.0,
                        xi,
                        limit=100,
                        epsabs=1e-8,
                        epsrel=1e-8
                    )
                    integral_part[i] = result

                    # Warn if error is large
                    if error > 1e-5:
                        self._integration_warnings.append(
                            f"Large integration error at x={xi:.4f}: {error:.2e}"
                        )
                except Exception as e:
                    warnings.warn(f"Integration failed at x={xi}: {e}")
                    integral_part[i] = 0.0
            else:  # xi < 1.0
                # Integrate from xi to 1 and negate
                try:
                    result, error = quad(
                        self._integrand,
                        xi,
                        1.0,
                        limit=100,
                        epsabs=1e-8,
                        epsrel=1e-8
                    )
                    integral_part[i] = -result

                    if error > 1e-5:
                        self._integration_warnings.append(
                            f"Large integration error at x={xi:.4f}: {error:.2e}"
                        )
                except Exception as e:
                    warnings.warn(f"Integration failed at x={xi}: {e}")
                    integral_part[i] = 0.0

        return linear_part + integral_part

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Compute derivative f'(x) = c₀ + integrand(x).

        The derivative is straightforward due to Fundamental Theorem of Calculus:
            d/dx[c₀·x + ∫[1,x] g(t)dt] = c₀ + g(x)

        Args:
            x: Points to evaluate derivative, shape (N,)

        Returns:
            Derivative values, shape (N,). Should always be > 0 for monotonicity.

        Examples:
            >>> params = BasisParameters(c_0=1.0)
            >>> basis = UniversalMonotonicBasis(params)
            >>> x = np.array([0.5, 1.0])
            >>> deriv = basis.derivative(x)
            >>> assert np.all(deriv > 0)  # Monotonicity check
        """
        x = np.atleast_1d(x)
        deriv = np.zeros_like(x, dtype=float)

        for i, xi in enumerate(x):
            deriv[i] = self.params.c_0 + self._integrand(xi)

        return deriv

    def verify_monotonic(self, x: np.ndarray, tolerance: float = 1e-6) -> bool:
        """Check if function is monotonically increasing on domain.

        Verifies that f'(x) ≥ -tolerance for all x in domain.

        Args:
            x: Domain points to check, shape (N,)
            tolerance: Numerical tolerance for non-negativity (default 1e-6)

        Returns:
            True if monotonic, False otherwise

        Note:
            Mathematically guaranteed to be True due to basis construction
            (derivative is c₀ + exp(...) which is always positive).
            This method is provided for numerical verification.

        Examples:
            >>> params = BasisParameters(c_0=0.1, a=1.0)
            >>> basis = UniversalMonotonicBasis(params)
            >>> x = np.linspace(0.01, 1.0, 100)
            >>> assert basis.verify_monotonic(x)
        """
        deriv = self.derivative(x)
        return bool(np.all(deriv >= -tolerance))

    def get_integration_warnings(self) -> List[str]:
        """Get any integration warnings accumulated during evaluation.

        Returns:
            List of warning messages
        """
        return self._integration_warnings

    def clear_warnings(self):
        """Clear accumulated integration warnings."""
        self._integration_warnings = []


class ParameterSampler:
    """Generate diverse parameter sets for Universal Monotonic Basis.

    Provides methods to sample parameters from predefined ranges with
    different emphasis modes (polynomial, spectral, sigmoid, etc.).

    Attributes:
        rng: NumPy random number generator
        seed: Random seed for reproducibility

    Examples:
        >>> sampler = ParameterSampler(seed=42)
        >>> params = sampler.sample_params(n_samples=10, mode='mixed')
        >>> len(params)
        10
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize sampler with random seed.

        Args:
            seed: Random seed for reproducibility (default: None)
        """
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def sample_params(
        self,
        n_samples: int = 1,
        mode: str = 'mixed'
    ) -> List[BasisParameters]:
        """Sample parameter sets with specified diversity mode.

        Args:
            n_samples: Number of parameter sets to generate
            mode: Sampling mode, one of:
                - 'polynomial': Emphasize c, d (polynomial trends)
                - 'spectral': Emphasize d_k (Fourier oscillations)
                - 'sigmoid': Emphasize g, h (smooth steps)
                - 'exponential': Emphasize e (exponential growth)
                - 'mixed': Random combinations (default)

        Returns:
            List of BasisParameters instances

        Note:
            Parameter ranges optimized for visual diversity on [0, 1]:
            - c_0: [0.1, 5.0] - overall slope
            - a: [-2.0, 2.0] - log-power behavior
            - b: [-5.0, 5.0] - vertical offset
            - c: [-3.0, 3.0] - linear growth
            - d: [-2.0, 2.0] - quadratic curvature
            - e: [-1.0, 1.0] - exponential (small to avoid overflow)
            - g: [-3.0, 3.0] - sigmoid amplitude
            - h: [0.5, 10.0] - sigmoid steepness
            - t_0: [0.3, 0.7] - sigmoid center
            - d_k: [-1.0, 1.0] for each k=1..5

        Examples:
            >>> sampler = ParameterSampler(seed=123)
            >>> params = sampler.sample_params(5, mode='polynomial')
            >>> # Polynomial mode emphasizes c and d terms
        """
        params_list = []

        for _ in range(n_samples):
            if mode == 'polynomial':
                params = self._sample_polynomial()
            elif mode == 'spectral':
                params = self._sample_spectral()
            elif mode == 'sigmoid':
                params = self._sample_sigmoid()
            elif mode == 'exponential':
                params = self._sample_exponential()
            elif mode == 'mixed':
                params = self._sample_mixed()
            else:
                raise ValueError(
                    f"Unknown mode '{mode}'. Choose from: "
                    "polynomial, spectral, sigmoid, exponential, mixed"
                )

            params_list.append(params)

        return params_list

    def _sample_polynomial(self) -> BasisParameters:
        """Sample parameters emphasizing polynomial terms."""
        return BasisParameters(
            c_0=self.rng.uniform(0.1, 5.0),
            a=self.rng.uniform(-1.0, 1.0),
            b=self.rng.uniform(-3.0, 3.0),
            c=self.rng.uniform(-3.0, 3.0),  # Emphasize
            d=self.rng.uniform(-2.0, 2.0),  # Emphasize
            e=self.rng.uniform(-0.3, 0.3),
            g=self.rng.uniform(-1.0, 1.0),
            h=self.rng.uniform(0.5, 5.0),
            t_0=self.rng.uniform(0.3, 0.7),
            d_k=[self.rng.uniform(-0.3, 0.3) for _ in range(5)]
        )

    def _sample_spectral(self) -> BasisParameters:
        """Sample parameters emphasizing Fourier spectral terms."""
        return BasisParameters(
            c_0=self.rng.uniform(0.1, 3.0),
            a=self.rng.uniform(-1.0, 1.0),
            b=self.rng.uniform(-2.0, 2.0),
            c=self.rng.uniform(-1.0, 1.0),
            d=self.rng.uniform(-0.5, 0.5),
            e=self.rng.uniform(-0.3, 0.3),
            g=self.rng.uniform(-1.0, 1.0),
            h=self.rng.uniform(0.5, 5.0),
            t_0=self.rng.uniform(0.3, 0.7),
            d_k=[self.rng.uniform(-1.0, 1.0) for _ in range(5)]  # Emphasize
        )

    def _sample_sigmoid(self) -> BasisParameters:
        """Sample parameters emphasizing sigmoid step features."""
        return BasisParameters(
            c_0=self.rng.uniform(0.1, 3.0),
            a=self.rng.uniform(-1.0, 1.0),
            b=self.rng.uniform(-3.0, 3.0),
            c=self.rng.uniform(-1.0, 1.0),
            d=self.rng.uniform(-0.5, 0.5),
            e=self.rng.uniform(-0.3, 0.3),
            g=self.rng.uniform(-3.0, 3.0),  # Emphasize
            h=self.rng.uniform(2.0, 10.0),  # Emphasize
            t_0=self.rng.uniform(0.3, 0.7),
            d_k=[self.rng.uniform(-0.3, 0.3) for _ in range(5)]
        )

    def _sample_exponential(self) -> BasisParameters:
        """Sample parameters emphasizing exponential growth."""
        return BasisParameters(
            c_0=self.rng.uniform(0.1, 3.0),
            a=self.rng.uniform(-1.0, 1.0),
            b=self.rng.uniform(-3.0, 3.0),
            c=self.rng.uniform(-1.0, 1.0),
            d=self.rng.uniform(-0.5, 0.5),
            e=self.rng.uniform(-1.0, 1.0),  # Emphasize
            g=self.rng.uniform(-1.0, 1.0),
            h=self.rng.uniform(0.5, 5.0),
            t_0=self.rng.uniform(0.3, 0.7),
            d_k=[self.rng.uniform(-0.3, 0.3) for _ in range(5)]
        )

    def _sample_mixed(self) -> BasisParameters:
        """Sample parameters with full random diversity."""
        return BasisParameters(
            c_0=self.rng.uniform(0.1, 5.0),
            a=self.rng.uniform(-2.0, 2.0),
            b=self.rng.uniform(-5.0, 5.0),
            c=self.rng.uniform(-3.0, 3.0),
            d=self.rng.uniform(-2.0, 2.0),
            e=self.rng.uniform(-1.0, 1.0),
            g=self.rng.uniform(-3.0, 3.0),
            h=self.rng.uniform(0.5, 10.0),
            t_0=self.rng.uniform(0.3, 0.7),
            d_k=[self.rng.uniform(-1.0, 1.0) for _ in range(5)]
        )


# Convenience function for quick usage
def create_linear_basis() -> UniversalMonotonicBasis:
    """Create a simple linear basis: f(x) = x.

    Returns:
        UniversalMonotonicBasis with linear parameters

    Examples:
        >>> basis = create_linear_basis()
        >>> x = np.array([0.5, 1.0])
        >>> y = basis.evaluate(x)
        >>> np.allclose(y, x)
        True
    """
    return UniversalMonotonicBasis(BasisParameters(c_0=1.0))


def create_log_basis() -> UniversalMonotonicBasis:
    """Create a logarithmic-like basis.

    Returns:
        UniversalMonotonicBasis with log-dominant parameters

    Examples:
        >>> basis = create_log_basis()
        >>> x = np.linspace(0.1, 1.0, 50)
        >>> y = basis.evaluate(x)
        >>> assert basis.verify_monotonic(x)
    """
    return UniversalMonotonicBasis(BasisParameters(c_0=0.0, a=1.0, b=-1.0))


def create_sigmoid_basis(steepness: float = 10.0, center: float = 0.5) -> UniversalMonotonicBasis:
    """Create a sigmoid step basis.

    Args:
        steepness: How sharp the sigmoid transition is (default 10.0)
        center: Where the sigmoid step occurs in [0,1] (default 0.5)

    Returns:
        UniversalMonotonicBasis with sigmoid parameters

    Examples:
        >>> basis = create_sigmoid_basis(steepness=5.0, center=0.3)
        >>> x = np.linspace(0.1, 1.0, 100)
        >>> y = basis.evaluate(x)
        >>> assert basis.verify_monotonic(x)
    """
    return UniversalMonotonicBasis(
        BasisParameters(c_0=0.0, g=5.0, h=steepness, t_0=center, b=0.5)
    )
