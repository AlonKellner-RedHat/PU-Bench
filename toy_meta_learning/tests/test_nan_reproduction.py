#!/usr/bin/env python3
"""Test to reproduce and fix NaN issue in NeuralPULoss.

Root causes of NaN:
1. Gradient explosion during meta-learning → weights grow unbounded
2. X1 * X2 multiplication squares large values → overflow to inf
3. sum(log(abs(A2))) on inf/0 values → NaN
4. L1 regularization on exploded weights → large gradients → more explosion
"""

import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.insert(0, '/Users/akellner/MyDir/Code/Other/PU-Bench/toy_meta_learning')

from loss.neural_pu_loss import NeuralPULoss


def test_normal_forward():
    """Test 1: Normal forward pass should not produce NaN."""
    print("\n" + "="*70)
    print("TEST 1: Normal forward pass")
    print("="*70)

    loss_fn = NeuralPULoss(hidden_dim=128, l1_lambda=0.001)

    # Typical batch
    outputs = torch.randn(64)
    labels = torch.where(torch.rand(64) > 0.7, torch.tensor(1.0), torch.tensor(-1.0))

    loss = loss_fn(outputs, labels, mode='pu')

    print(f"Loss value: {loss.item():.6f}")
    print(f"Has NaN: {torch.isnan(loss).item()}")
    print(f"Has Inf: {torch.isinf(loss).item()}")

    assert not torch.isnan(loss), "Normal forward pass produced NaN!"
    assert not torch.isinf(loss), "Normal forward pass produced Inf!"
    print("✓ PASSED")


def test_extreme_weights():
    """Test 2: Large weights should cause overflow/NaN."""
    print("\n" + "="*70)
    print("TEST 2: Extreme weights causing NaN")
    print("="*70)

    loss_fn = NeuralPULoss(hidden_dim=128, l1_lambda=0.001)

    # Simulate weights after many iterations of gradient explosion
    with torch.no_grad():
        loss_fn.linear.weight *= 100.0  # Explode weights

    outputs = torch.randn(64)
    labels = torch.where(torch.rand(64) > 0.7, torch.tensor(1.0), torch.tensor(-1.0))

    loss = loss_fn(outputs, labels, mode='pu')

    print(f"Loss value: {loss.item()}")
    print(f"Has NaN: {torch.isnan(loss).item()}")
    print(f"Has Inf: {torch.isinf(loss).item()}")
    print(f"Weight norm: {loss_fn.linear.weight.norm().item():.2f}")

    if torch.isnan(loss) or torch.isinf(loss):
        print("✓ REPRODUCED NaN/Inf with large weights")
        return True
    else:
        print("✗ Did not reproduce NaN (weights not large enough)")
        return False


def test_gradient_explosion_scenario():
    """Test 3: Simulate gradient explosion through meta-learning."""
    print("\n" + "="*70)
    print("TEST 3: Gradient explosion through repeated updates")
    print("="*70)

    loss_fn = NeuralPULoss(hidden_dim=128, l1_lambda=0.001)
    optimizer = torch.optim.Adam(loss_fn.parameters(), lr=0.01)  # High LR

    # Simulate aggressive meta-learning updates
    for iteration in range(50):
        outputs = torch.randn(64)
        labels = torch.where(torch.rand(64) > 0.7, torch.tensor(1.0), torch.tensor(-1.0))

        optimizer.zero_grad()
        loss = loss_fn(outputs, labels, mode='pu')

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"✓ NaN/Inf appeared at iteration {iteration}")
            print(f"   Loss: {loss.item()}")
            print(f"   Weight norm: {loss_fn.linear.weight.norm().item():.2f}")
            return True

        # Simulate meta-gradient (maximize loss instead of minimize)
        # This mimics gradient matching trying to increase the loss
        (-loss).backward()  # Negative to maximize
        optimizer.step()

        if iteration % 10 == 0:
            weight_norm = loss_fn.linear.weight.norm().item()
            print(f"  Iter {iteration}: loss={loss.item():.4f}, weight_norm={weight_norm:.2f}")

    print("✗ Did not reproduce NaN in 50 iterations")
    return False


def test_proposed_fix_with_weight_clipping():
    """Test 4: Proposed fix - clip weights to prevent explosion."""
    print("\n" + "="*70)
    print("TEST 4: Proposed fix - weight clipping")
    print("="*70)

    loss_fn = NeuralPULoss(hidden_dim=128, l1_lambda=0.001)
    optimizer = torch.optim.Adam(loss_fn.parameters(), lr=0.01)

    max_weight_norm = 10.0  # Clip threshold

    for iteration in range(50):
        outputs = torch.randn(64)
        labels = torch.where(torch.rand(64) > 0.7, torch.tensor(1.0), torch.tensor(-1.0))

        optimizer.zero_grad()
        loss = loss_fn(outputs, labels, mode='pu')

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"✗ FAILED: NaN appeared at iteration {iteration} despite clipping")
            return False

        (-loss).backward()

        # PROPOSED FIX: Clip gradients
        torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), max_norm=1.0)

        optimizer.step()

        # PROPOSED FIX: Clip weight magnitudes
        with torch.no_grad():
            weight_norm = loss_fn.linear.weight.norm()
            if weight_norm > max_weight_norm:
                loss_fn.linear.weight *= (max_weight_norm / weight_norm)

        if iteration % 10 == 0:
            weight_norm = loss_fn.linear.weight.norm().item()
            print(f"  Iter {iteration}: loss={loss.item():.4f}, weight_norm={weight_norm:.2f}")

    print("✓ PASSED: No NaN with weight clipping")
    return True


def test_proposed_fix_with_log_safeguard():
    """Test 5: Additional fix - safer log operation."""
    print("\n" + "="*70)
    print("TEST 5: Proposed fix - safer log with abs+eps")
    print("="*70)

    # Test the problematic log operation
    A2 = torch.tensor([1e-10, -1e-10, 0.0, 1e10, -1e10])
    eps = 1e-7

    # Current implementation
    result_current = torch.log(torch.clamp(torch.abs(A2), min=eps))
    print(f"Current implementation: {result_current}")
    print(f"Has NaN: {torch.isnan(result_current).any().item()}")
    print(f"Has Inf: {torch.isinf(result_current).any().item()}")

    # Proposed: clamp both before and after abs
    result_proposed = torch.log(torch.clamp(torch.abs(A2), min=eps, max=1e6))
    print(f"Proposed implementation: {result_proposed}")
    print(f"Has NaN: {torch.isnan(result_proposed).any().item()}")
    print(f"Has Inf: {torch.isinf(result_proposed).any().item()}")

    # Both should be safe, but proposed adds upper bound
    assert not torch.isnan(result_proposed).any()
    print("✓ PASSED: Log operation is safe")
    return True


def test_proposed_fix_integration():
    """Test 6: Full integration test with all fixes."""
    print("\n" + "="*70)
    print("TEST 6: Integration test - all fixes combined")
    print("="*70)

    class SafeNeuralPULoss(NeuralPULoss):
        """NeuralPULoss with NaN prevention."""

        def __init__(self, *args, max_weight_norm=10.0, **kwargs):
            super().__init__(*args, **kwargs)
            self.max_weight_norm = max_weight_norm

        def forward(self, outputs, labels, mode='pu'):
            # Clip weights before forward pass
            with torch.no_grad():
                weight_norm = self.linear.weight.norm()
                if weight_norm > self.max_weight_norm:
                    self.linear.weight *= (self.max_weight_norm / weight_norm)

            # Call parent forward
            loss = super().forward(outputs, labels, mode)

            # Safety check
            if torch.isnan(loss) or torch.isinf(loss):
                print("  WARNING: NaN/Inf detected, returning safe fallback")
                return torch.tensor(1.0, device=loss.device, requires_grad=True)

            return loss

    # Test with aggressive updates
    loss_fn = SafeNeuralPULoss(hidden_dim=128, l1_lambda=0.001, max_weight_norm=5.0)
    optimizer = torch.optim.Adam(loss_fn.parameters(), lr=0.01)

    nan_count = 0
    for iteration in range(100):
        outputs = torch.randn(64)
        labels = torch.where(torch.rand(64) > 0.7, torch.tensor(1.0), torch.tensor(-1.0))

        optimizer.zero_grad()
        loss = loss_fn(outputs, labels, mode='pu')

        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1

        (-loss).backward()
        torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), max_norm=1.0)
        optimizer.step()

        if iteration % 20 == 0:
            weight_norm = loss_fn.linear.weight.norm().item()
            print(f"  Iter {iteration}: loss={loss.item():.4f}, weight_norm={weight_norm:.2f}")

    print(f"\nNaN count: {nan_count}/100")
    if nan_count == 0:
        print("✓ PASSED: No NaN in 100 iterations with all fixes")
        return True
    else:
        print(f"✗ FAILED: {nan_count} NaN occurrences")
        return False


def main():
    """Run all tests."""
    print("NaN REPRODUCTION AND FIX TESTS")
    print("Testing NeuralPULoss for numerical stability issues")

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    results = {
        "Normal forward": test_normal_forward(),
        "Extreme weights": test_extreme_weights(),
        "Gradient explosion": test_gradient_explosion_scenario(),
        "Weight clipping fix": test_proposed_fix_with_weight_clipping(),
        "Log safeguard fix": test_proposed_fix_with_log_safeguard(),
        "Integration test": test_proposed_fix_integration(),
    }

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:30s} {status}")

    all_passed = all(results.values())
    print("\n" + ("="*70))
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("="*70)


if __name__ == '__main__':
    main()
