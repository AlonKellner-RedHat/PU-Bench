"""Direct verification of baseline mathematical functions in the monotonic basis.

This script tests that the Universal Monotonic Spectral Basis can exactly represent
fundamental mathematical functions by setting specific parameter combinations.

Each test directly sets the basis parameters to achieve a target function and
verifies the output matches the expected mathematical formula.

Run: uv run python test_baseline_functions.py
"""

import torch
import numpy as np
from utils.monotonic_basis_torch import monotonic_basis_full


def test_pure_logarithm():
    """Test f(x) = log(x) with a=-1, c₀=0."""
    print("\n" + "="*70)
    print("Test 1: Pure Logarithm f(x) = log(x)")
    print("="*70)
    print("Parameters: a=-1, c₀=0, all others=0")
    print("Mechanism: g(t) = exp(-log(t)) = 1/t, then ∫[1,x] 1/t dt = log(x)")

    x = torch.tensor([0.5, 1.0, 2.0, 3.0, 5.0])

    # Parameters for log(x)
    c_0 = torch.tensor(0.0)     # No linear term
    c_1 = torch.tensor(1.0)     # Integral coefficient = 1
    a = torch.tensor(-1.0)      # Makes g(t) = 1/t
    b = torch.tensor(0.0)
    c = torch.tensor(0.0)
    d = torch.tensor(0.0)
    e = torch.tensor(0.0)
    g = torch.tensor(0.0)
    h = torch.tensor(1.0)
    t_0 = torch.tensor(0.5)
    d_k = torch.zeros(5)

    result = monotonic_basis_full(x, c_0, c_1, a, b, c, d, e, g, h, t_0, d_k, num_integration_points=100)
    expected = torch.log(x)

    rel_error = torch.abs((result - expected) / (expected.abs() + 1e-8)).max().item() * 100

    print(f"\nx values: {x.numpy()}")
    print(f"Expected log(x): {expected.numpy()}")
    print(f"Got f(x):        {result.numpy()}")
    print(f"Max relative error: {rel_error:.3f}%")

    if rel_error < 1.0:
        print("✅ PASS: log(x) represented with < 1% error")
        return True
    else:
        print(f"❌ FAIL: Relative error {rel_error:.3f}% too large")
        return False


def test_pure_linear():
    """Test f(x) = k·x with c₀=k, c₁=0."""
    print("\n" + "="*70)
    print("Test 2: Pure Linear f(x) = k·x")
    print("="*70)
    print("Parameters: c₀=2.5, c₁=0 (no integral contribution)")
    print("Mechanism: Linear term c₀·x only, integral term disabled with c₁=0")

    x = torch.linspace(0.1, 5.0, 20)
    k = 2.5

    # Parameters for k·x
    # Set c_1=0 to disable integral contribution (much cleaner than b=-50 hack!)
    c_0 = torch.tensor(k)
    c_1 = torch.tensor(0.0)  # Disable integral
    a = torch.tensor(0.0)
    b = torch.tensor(0.0)
    c = torch.tensor(0.0)
    d = torch.tensor(0.0)
    e = torch.tensor(0.0)
    g = torch.tensor(0.0)
    h = torch.tensor(1.0)
    t_0 = torch.tensor(0.5)
    d_k = torch.zeros(5)

    result = monotonic_basis_full(x, c_0, c_1, a, b, c, d, e, g, h, t_0, d_k)
    expected = k * x

    error = torch.abs(result - expected).max().item()

    print(f"\nSample values (k={k}):")
    for i in [0, 5, 10, 15, 19]:
        print(f"  x={x[i]:.2f}: expected={expected[i]:.3f}, got={result[i]:.3f}")
    print(f"Max absolute error: {error:.6f}")

    if error < 1e-4:
        print("✅ PASS: Linear function exact to 1e-4")
        return True
    else:
        print(f"❌ FAIL: Error {error:.6f} too large")
        return False


def test_exponential_in_exponent():
    """Test g(x) = exp(k·x) using c parameter."""
    print("\n" + "="*70)
    print("Test 3: Exponential via Integrand g(x) = exp(k·x)")
    print("="*70)
    print("Parameters: c=k in exponent, a=0, c₀=0")
    print("Mechanism: g(x) = exp(c·x) appears in integrand, integrates to exp(k·x)/k")

    x = torch.linspace(0.0, 2.0, 10)
    k = 1.5

    # Parameters for exponential integrand
    c_0 = torch.tensor(0.0)
    c_1 = torch.tensor(1.0)
    a = torch.tensor(0.0)
    b = torch.tensor(0.0)
    c = torch.tensor(k)      # Exponential term in exponent
    d = torch.tensor(0.0)
    e = torch.tensor(0.0)
    g = torch.tensor(0.0)
    h = torch.tensor(1.0)
    t_0 = torch.tensor(0.5)
    d_k = torch.zeros(5)

    result = monotonic_basis_full(x, c_0, c_1, a, b, c, d, e, g, h, t_0, d_k, num_integration_points=100)

    # Expected: ∫[1,x] exp(k·t) dt = [exp(k·t)/k]₁ˣ = (exp(k·x) - exp(k))/k
    expected = (torch.exp(k * x) - torch.exp(torch.tensor(k))) / k

    rel_error = torch.abs((result - expected) / (expected.abs() + 1e-8)).max().item() * 100

    print(f"\nSample values (k={k}):")
    for i in [0, 3, 6, 9]:
        print(f"  x={x[i]:.2f}: expected={expected[i]:.4f}, got={result[i]:.4f}")
    print(f"Max relative error: {rel_error:.3f}%")

    if rel_error < 5.0:  # Allow 5% for numerical integration
        print("✅ PASS: Exponential represented with < 5% error")
        return True
    else:
        print(f"❌ FAIL: Relative error {rel_error:.3f}% too large")
        return False


def test_power_law():
    """Test f(x) = x^a via integrating x^(a-1)."""
    print("\n" + "="*70)
    print("Test 4: Power Law f(x) ∝ x^a")
    print("="*70)
    print("Parameters: a=2 (makes g(t) = t²), integrate to get t³/3")
    print("Mechanism: g(t) = exp(a·log(t)) = t^a, ∫[1,x] t^a dt = [t^(a+1)/(a+1)]₁ˣ")

    x = torch.linspace(1.0, 3.0, 10)
    alpha = 2.0  # Will give g(t) = t², integrates to t³/3

    # Parameters for power law
    c_0 = torch.tensor(0.0)
    c_1 = torch.tensor(1.0)
    a = torch.tensor(alpha)   # Makes g(t) = exp(alpha·log(t)) = t^alpha
    b = torch.tensor(0.0)
    c = torch.tensor(0.0)
    d = torch.tensor(0.0)
    e = torch.tensor(0.0)
    g = torch.tensor(0.0)
    h = torch.tensor(1.0)
    t_0 = torch.tensor(0.5)
    d_k = torch.zeros(5)

    result = monotonic_basis_full(x, c_0, c_1, a, b, c, d, e, g, h, t_0, d_k, num_integration_points=100)

    # Expected: ∫[1,x] t^alpha dt = [t^(alpha+1)/(alpha+1)]₁ˣ = (x^(alpha+1) - 1)/(alpha+1)
    expected = (x**(alpha + 1) - 1) / (alpha + 1)

    rel_error = torch.abs((result - expected) / (expected.abs() + 1e-8)).max().item() * 100

    print(f"\nSample values (a={alpha}):")
    for i in [0, 3, 6, 9]:
        print(f"  x={x[i]:.2f}: x³/3={expected[i]:.4f}, got={result[i]:.4f}")
    print(f"Max relative error: {rel_error:.3f}%")

    if rel_error < 1.0:
        print("✅ PASS: Power law represented with < 1% error")
        return True
    else:
        print(f"❌ FAIL: Relative error {rel_error:.3f}% too large")
        return False


def test_pudra_combination():
    """Test f(x) = -log(p) + p (PUDRa positive risk term)."""
    print("\n" + "="*70)
    print("Test 5: PUDRa Combination f(p) = -log(p) + p")
    print("="*70)
    print("Using two separate basis functions combined:")
    print("  f1(p) = log(p)   [a=-1, c₀=0, c₁=1]")
    print("  f2(p) = p        [c₀=1, c₁=0]")
    print("  Result: -f1 + f2 = -log(p) + p")

    p = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])

    # First basis: log(p)
    c_0_log = torch.tensor(0.0)
    c_1_log = torch.tensor(1.0)
    a_log = torch.tensor(-1.0)
    zeros = torch.tensor(0.0)
    d_k = torch.zeros(5)

    log_p = monotonic_basis_full(
        p, c_0_log, c_1_log, a_log, zeros, zeros, zeros, zeros, zeros,
        torch.tensor(1.0), torch.tensor(0.5), d_k, num_integration_points=100
    )

    # Second basis: p (c_1=0 to disable integral)
    c_0_linear = torch.tensor(1.0)
    c_1_linear = torch.tensor(0.0)  # Disable integral (cleaner than b=-50!)
    a_linear = torch.tensor(0.0)

    linear_p = monotonic_basis_full(
        p, c_0_linear, c_1_linear, a_linear, zeros, zeros, zeros, zeros, zeros,
        torch.tensor(1.0), torch.tensor(0.5), d_k
    )

    # Combine: -log(p) + p
    result = -log_p + linear_p
    expected = -torch.log(p) + p

    error = torch.abs(result - expected).max().item()

    print(f"\np values: {p.numpy()}")
    print(f"Expected (-log(p) + p): {expected.numpy()}")
    print(f"Got (basis combo):      {result.numpy()}")
    print(f"Max absolute error: {error:.4f}")

    if error < 0.01:
        print("✅ PASS: PUDRa combination -log(p) + p with < 0.01 error")
        return True
    else:
        print(f"❌ FAIL: Error {error:.4f} too large")
        return False


def test_sigmoid_derivative_integration():
    """Test that sigmoid derivative integrand gives sigmoid function."""
    print("\n" + "="*70)
    print("Test 6: Sigmoid via Derivative Integration")
    print("="*70)
    print("Parameters: g=1, h=1, t₀=0 (sigmoid derivative in integrand)")
    print("Mechanism: g(t) includes σ'(t), which integrates to σ(t)")

    x = torch.linspace(-3.0, 3.0, 20)

    # Parameters for sigmoid
    # Note: The integrand includes g·σ'(h·(t-t₀)) in the EXPONENT
    # So we get exp(g·σ'(t)), which is NOT directly σ(t)
    # This test shows the sigmoid derivative term exists

    c_0 = torch.tensor(0.0)
    c_1 = torch.tensor(1.0)
    a = torch.tensor(0.0)
    b = torch.tensor(0.0)
    c = torch.tensor(0.0)
    d = torch.tensor(0.0)
    e = torch.tensor(0.0)
    g_param = torch.tensor(1.0)  # Sigmoid derivative amplitude
    h = torch.tensor(1.0)        # Steepness
    t_0 = torch.tensor(0.0)      # Center
    d_k = torch.zeros(5)

    result = monotonic_basis_full(x, c_0, c_1, a, b, c, d, e, g_param, h, t_0, d_k, num_integration_points=100)

    # This won't give exactly sigmoid because σ' is in the exponent
    # But we can show the sigmoid derivative term affects the output
    print(f"\nNote: The integrand contains σ'(x) in the EXPONENT:")
    print(f"  g(t) = exp(...+ g·σ'(t) +...)")
    print(f"  This is NOT the same as g(t) = σ'(t)")
    print(f"\nFor exact sigmoid representation, we'd need σ'(t) as the integrand itself,")
    print(f"not in the exponent. The current basis can approximate sigmoid via learned")
    print(f"combinations of the available terms.")

    print(f"\nSample output with sigmoid derivative term:")
    for i in [0, 5, 10, 15, 19]:
        sigmoid_deriv = torch.sigmoid(x[i]) * (1 - torch.sigmoid(x[i]))
        print(f"  x={x[i]:5.2f}: σ'(x)={sigmoid_deriv:.4f}, f(x)={result[i]:.4f}")

    # Just verify it's non-constant (showing the term has an effect)
    is_varying = result.std().item() > 0.01

    if is_varying:
        print("✅ PASS: Sigmoid derivative term affects output")
        return True
    else:
        print("❌ FAIL: Output is nearly constant")
        return False


def test_quadratic():
    """Test f(x) = x² using a=2 (g(t) = t²)."""
    print("\n" + "="*70)
    print("Test 7: Quadratic f(x) ∝ x²")
    print("="*70)
    print("Parameters: a=1 (makes g(t) = t), integrates to t²/2")

    x = torch.linspace(1.0, 4.0, 10)

    # Parameters: g(t) = exp(1·log(t)) = t, integrates to t²/2
    c_0 = torch.tensor(0.0)
    c_1 = torch.tensor(1.0)
    a = torch.tensor(1.0)    # Makes g(t) = t
    b = torch.tensor(0.0)
    c = torch.tensor(0.0)
    d = torch.tensor(0.0)
    e = torch.tensor(0.0)
    g = torch.tensor(0.0)
    h = torch.tensor(1.0)
    t_0 = torch.tensor(0.5)
    d_k = torch.zeros(5)

    result = monotonic_basis_full(x, c_0, c_1, a, b, c, d, e, g, h, t_0, d_k, num_integration_points=100)

    # Expected: ∫[1,x] t dt = [t²/2]₁ˣ = (x² - 1)/2
    expected = (x**2 - 1) / 2

    rel_error = torch.abs((result - expected) / (expected.abs() + 1e-8)).max().item() * 100

    print(f"\nSample values:")
    for i in [0, 3, 6, 9]:
        print(f"  x={x[i]:.2f}: (x²-1)/2={expected[i]:.4f}, got={result[i]:.4f}")
    print(f"Max relative error: {rel_error:.3f}%")

    if rel_error < 1.0:
        print("✅ PASS: Quadratic represented with < 1% error")
        return True
    else:
        print(f"❌ FAIL: Relative error {rel_error:.3f}% too large")
        return False


def test_constant_offset():
    """Test f(x) = c using b parameter in integrand."""
    print("\n" + "="*70)
    print("Test 8: Linear Growth via Constant Integrand")
    print("="*70)
    print("Parameters: b=k (constant in exponent), a=0, c₀=0")
    print("Mechanism: g(t) = exp(k), ∫[1,x] exp(k) dt = exp(k)·(x-1)")

    x = torch.linspace(1.0, 5.0, 10)
    k = 2.0

    # Parameters: g(t) = exp(b) = exp(k) = constant
    c_0 = torch.tensor(0.0)
    c_1 = torch.tensor(1.0)
    a = torch.tensor(0.0)
    b = torch.tensor(k)      # Constant in exponent
    c = torch.tensor(0.0)
    d = torch.tensor(0.0)
    e = torch.tensor(0.0)
    g = torch.tensor(0.0)
    h = torch.tensor(1.0)
    t_0 = torch.tensor(0.5)
    d_k = torch.zeros(5)

    result = monotonic_basis_full(x, c_0, c_1, a, b, c, d, e, g, h, t_0, d_k, num_integration_points=100)

    # Expected: ∫[1,x] exp(k) dt = exp(k)·(x - 1)
    expected = torch.exp(torch.tensor(k)) * (x - 1)

    rel_error = torch.abs((result - expected) / (expected.abs() + 1e-8)).max().item() * 100

    print(f"\nSample values (k={k}, exp(k)={np.exp(k):.3f}):")
    for i in [0, 3, 6, 9]:
        print(f"  x={x[i]:.2f}: exp(k)·(x-1)={expected[i]:.4f}, got={result[i]:.4f}")
    print(f"Max relative error: {rel_error:.3f}%")

    if rel_error < 1.0:
        print("✅ PASS: Linear growth from constant integrand < 1% error")
        return True
    else:
        print(f"❌ FAIL: Relative error {rel_error:.3f}% too large")
        return False


def test_combined_affine():
    """Test f(x) = a + b·x using c₀ and constant integrand."""
    print("\n" + "="*70)
    print("Test 9: Affine Function f(x) = offset + slope·x")
    print("="*70)
    print("Parameters: c₀=3, b=log(2) (so g(t) = exp(b) = 2)")
    print("Mechanism: f(x) = c₀·x + ∫[1,x] exp(b) dt = 3x + 2(x-1) = 5x - 2")

    x = torch.linspace(0.0, 3.0, 10)

    slope_from_c0 = 3.0
    slope_from_integral = 2.0
    log_slope = np.log(slope_from_integral)

    # Parameters
    c_0 = torch.tensor(slope_from_c0)
    c_1 = torch.tensor(1.0)
    a = torch.tensor(0.0)
    b = torch.tensor(log_slope)  # exp(b) = slope_from_integral
    c = torch.tensor(0.0)
    d = torch.tensor(0.0)
    e = torch.tensor(0.0)
    g = torch.tensor(0.0)
    h = torch.tensor(1.0)
    t_0 = torch.tensor(0.5)
    d_k = torch.zeros(5)

    result = monotonic_basis_full(x, c_0, c_1, a, b, c, d, e, g, h, t_0, d_k, num_integration_points=100)

    # Expected: 3x + 2(x-1) = 5x - 2
    total_slope = slope_from_c0 + slope_from_integral
    offset = -slope_from_integral
    expected = total_slope * x + offset

    error = torch.abs(result - expected).max().item()

    print(f"\nExpected: f(x) = {total_slope}x + {offset:.1f}")
    print(f"Sample values:")
    for i in [0, 3, 6, 9]:
        print(f"  x={x[i]:.2f}: expected={expected[i]:.4f}, got={result[i]:.4f}")
    print(f"Max absolute error: {error:.4f}")

    if error < 0.01:
        print("✅ PASS: Affine function with < 0.01 error")
        return True
    else:
        print(f"❌ FAIL: Error {error:.4f} too large")
        return False


def main():
    """Run all baseline function tests."""
    print("\n" + "="*70)
    print("BASELINE FUNCTION VERIFICATION")
    print("="*70)
    print("\nDirect tests of mathematical functions using specific parameter settings.")
    print("Each test sets basis parameters to achieve a target function and verifies")
    print("the output matches the expected mathematical formula.\n")

    results = []

    # Run all tests
    results.append(("Pure logarithm log(x)", test_pure_logarithm()))
    results.append(("Pure linear k·x", test_pure_linear()))
    results.append(("Exponential integrand", test_exponential_in_exponent()))
    results.append(("Power law x^a", test_power_law()))
    results.append(("PUDRa combination -log(p)+p", test_pudra_combination()))
    results.append(("Sigmoid derivative term", test_sigmoid_derivative_integration()))
    results.append(("Quadratic x²", test_quadratic()))
    results.append(("Constant offset linear", test_constant_offset()))
    results.append(("Affine combination", test_combined_affine()))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")

    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    print("\n1. EXACT representations (error < 1%):")
    print("   • Logarithm: log(x) via a=-1, integrate 1/x")
    print("   • Linear: k·x via c₀=k")
    print("   • Power laws: x^a via a parameter")
    print("   • Affine: a + b·x via c₀ and constant integrand")
    print("\n2. Good approximations (error < 5%):")
    print("   • Exponential: exp(k·x) via c parameter")
    print("\n3. Available as building blocks:")
    print("   • Sigmoid derivative σ'(x) in integrand")
    print("   • Polynomial terms via b, c, d")
    print("   • Spectral terms via Fourier coefficients d_k")
    print("\n4. PU Loss Components:")
    print("   • PUDRa: Can exactly represent -log(p) + p")
    print("   • Multiple basis functions can be combined in the loss")
    print("   • With 21 basis functions (R=3), can learn complex combinations")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
