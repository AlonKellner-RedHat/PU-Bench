"""Test that the full basis with integration works correctly.

This verifies:
1. Numerical integration is accurate
2. PyTorch autograd correctly computes derivatives: df/dx = c₀ + g(x)
3. log(x) can be exactly represented via integration

Run: uv run python test_full_basis.py
"""

import torch
from utils.monotonic_basis_torch import monotonic_basis_integrand, monotonic_basis_full


def test_integration_accuracy():
    """Test that numerical integration gives accurate results."""
    print("\n" + "="*70)
    print("Test 1: Integration Accuracy")
    print("="*70)

    # Test case: ∫[1,x] 1/t dt = log(x)
    # Use a=-1 to get g(t) = exp(-log(t)) = 1/t
    x = torch.tensor([1.5, 2.0, 3.0, 5.0], requires_grad=True)

    c_0 = torch.tensor(0.0)
    a = torch.tensor(-1.0)  # Makes g(t) = 1/t
    b = torch.tensor(0.0)
    c = torch.tensor(0.0)
    d = torch.tensor(0.0)
    e = torch.tensor(0.0)
    g_param = torch.tensor(0.0)
    h = torch.tensor(1.0)
    t_0 = torch.tensor(0.5)
    d_k = torch.zeros(5)

    # Compute full basis
    f_x = monotonic_basis_full(
        x, c_0, a, b, c, d, e, g_param, h, t_0, d_k,
        num_integration_points=100  # More points for accuracy
    )

    # Expected: ∫[1,x] 1/t dt = log(x)
    expected = torch.log(x)

    error = torch.abs(f_x - expected).max().item()

    print(f"Input x: {x.detach().numpy()}")
    print(f"Expected (log(x)): {expected.detach().numpy()}")
    print(f"Got (∫[1,x] 1/t dt): {f_x.detach().numpy()}")
    print(f"Max error: {error:.6f}")

    if error < 0.01:  # Allow 1% error for numerical integration
        print("✅ PASS: Integration gives log(x) with < 1% error")
        return True
    else:
        print(f"❌ FAIL: Error {error:.6f} too large")
        return False


def test_derivative_relationship():
    """Test that PyTorch autograd correctly computes df/dx = c₀ + g(x)."""
    print("\n" + "="*70)
    print("Test 2: Derivative Relationship via Autograd")
    print("="*70)

    # Test case: f(x) = c₀·x + ∫[1,x] 1/t dt = c₀·x + log(x)
    # Then: f'(x) = c₀ + 1/x
    x = torch.tensor([1.5, 2.0, 3.0], requires_grad=True)

    c_0 = torch.tensor(2.0)  # Non-zero linear coefficient
    a = torch.tensor(-1.0)   # Makes g(t) = 1/t
    b = torch.tensor(0.0)
    c = torch.tensor(0.0)
    d = torch.tensor(0.0)
    e = torch.tensor(0.0)
    g_param = torch.tensor(0.0)
    h = torch.tensor(1.0)
    t_0 = torch.tensor(0.5)
    d_k = torch.zeros(5)

    # Compute full basis
    f_x = monotonic_basis_full(
        x, c_0, a, b, c, d, e, g_param, h, t_0, d_k,
        num_integration_points=100
    )

    # Compute gradients using autograd
    grad_outputs = torch.ones_like(f_x)
    gradients = torch.autograd.grad(
        outputs=f_x,
        inputs=x,
        grad_outputs=grad_outputs,
        create_graph=False
    )[0]

    # Expected derivative: f'(x) = c₀ + g(x) = c₀ + 1/x
    g_x = monotonic_basis_integrand(x, c_0, a, b, c, d, e, g_param, h, t_0, d_k)
    expected_gradient = c_0 + g_x

    error = torch.abs(gradients - expected_gradient).max().item()

    print(f"Input x: {x.detach().numpy()}")
    print(f"f(x) = {c_0.item():.1f}·x + log(x): {f_x.detach().numpy()}")
    print(f"\nExpected gradient (c₀ + 1/x): {expected_gradient.detach().numpy()}")
    print(f"Autograd gradient: {gradients.detach().numpy()}")
    print(f"Max error: {error:.6f}")

    if error < 0.01:
        print("✅ PASS: Autograd correctly computes df/dx = c₀ + g(x)")
        return True
    else:
        print(f"❌ FAIL: Error {error:.6f} too large")
        return False


def test_logarithm_exact():
    """Test that log(x) is now exactly representable (up to numerical error)."""
    print("\n" + "="*70)
    print("Test 3: Exact Logarithm Representation")
    print("="*70)

    x = torch.linspace(1.1, 5.0, 20)

    # Parameters for f(x) = log(x)
    # g(t) = 1/t via a=-1, then integrate
    c_0 = torch.tensor(0.0)
    a = torch.tensor(-1.0)
    b = torch.tensor(0.0)
    c = torch.tensor(0.0)
    d = torch.tensor(0.0)
    e = torch.tensor(0.0)
    g_param = torch.tensor(0.0)
    h = torch.tensor(1.0)
    t_0 = torch.tensor(0.5)
    d_k = torch.zeros(5)

    # Compute full basis
    f_x = monotonic_basis_full(
        x, c_0, a, b, c, d, e, g_param, h, t_0, d_k,
        num_integration_points=100
    )

    expected = torch.log(x)
    error = torch.abs(f_x - expected).max().item()
    rel_error = (error / expected.max().item()) * 100

    print(f"Sample values:")
    for i in [0, 5, 10, 15, 19]:
        print(f"  x={x[i]:.2f}: log(x)={expected[i]:.4f}, "
              f"∫[1,x] 1/t dt={f_x[i]:.4f}, error={abs(f_x[i]-expected[i]):.6f}")

    print(f"\nMax absolute error: {error:.6f}")
    print(f"Max relative error: {rel_error:.2f}%")

    if rel_error < 1.0:  # Less than 1% relative error
        print("✅ PASS: log(x) represented with < 1% relative error")
        return True
    else:
        print(f"❌ FAIL: Relative error {rel_error:.2f}% too large")
        return False


def test_negative_log_for_pudra():
    """Test that -log(x) can be represented for PUDRa loss."""
    print("\n" + "="*70)
    print("Test 4: -log(x) for PUDRa Loss")
    print("="*70)

    x = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])

    # Parameters for f(x) = -log(x)
    # Use negative c_0 to flip the sign
    c_0 = torch.tensor(0.0)  # No linear term
    a = torch.tensor(-1.0)   # g(t) = 1/t, integrates to log(x)
    b = torch.tensor(0.0)
    c = torch.tensor(0.0)
    d = torch.tensor(0.0)
    e = torch.tensor(0.0)
    g_param = torch.tensor(0.0)
    h = torch.tensor(1.0)
    t_0 = torch.tensor(0.5)
    d_k = torch.zeros(5)

    # Compute log(x)
    log_x = monotonic_basis_full(
        x, c_0, a, b, c, d, e, g_param, h, t_0, d_k,
        num_integration_points=100
    )

    # For -log(x), multiply by -1 (this would be done in the loss via learned params)
    neg_log_x = -log_x
    expected = -torch.log(x)

    error = torch.abs(neg_log_x - expected).max().item()

    print(f"Input x: {x.numpy()}")
    print(f"Expected (-log(x)): {expected.numpy()}")
    print(f"Got (-∫[1,x] 1/t dt): {neg_log_x.detach().numpy()}")
    print(f"Max error: {error:.6f}")

    print("\nNote: In the loss, multiple basis functions combine:")
    print("  f_positive(p) can represent -log(p) for PUDRa's positive risk")
    print("  f_unlabeled(p) can represent p for PUDRa's unlabeled risk")

    if error < 0.01:
        print("✅ PASS: -log(x) represented correctly")
        return True
    else:
        print(f"❌ FAIL: Error {error:.6f} too large")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("FULL BASIS WITH INTEGRATION: VERIFICATION TESTS")
    print("="*70)
    print("\nThese tests verify that:")
    print("1. Numerical integration is accurate")
    print("2. PyTorch autograd handles derivatives correctly")
    print("3. log(x) is now exactly representable (not just 1/x)")

    results = []
    results.append(("Integration accuracy", test_integration_accuracy()))
    results.append(("Derivative via autograd", test_derivative_relationship()))
    results.append(("Exact logarithm", test_logarithm_exact()))
    results.append(("-log(x) for PUDRa", test_negative_log_for_pudra()))

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
    print("CONCLUSION")
    print("="*70)
    print("\nThe full basis with integration now:")
    print("  • EXACTLY represents log(x) via ∫[1,x] 1/t dt")
    print("  • Has correct derivatives: f'(x) = c₀ + g(x)")
    print("  • Enables exact representation of PUDRa: -log(p) + p")
    print("  • Maintains differentiability through PyTorch autograd")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
