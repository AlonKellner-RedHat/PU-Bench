#!/usr/bin/env python3
"""Test PROPER solutions for differentiable optimization.

Demonstrates:
1. higher library (recommended)
2. torch.func.functional_call()
"""

import torch
import torch.nn as nn
import higher

print("="*80)
print("TESTING PROPER SOLUTIONS FOR META-LEARNING")
print("="*80)

# ==============================================================================
# SOLUTION 1: higher library (RECOMMENDED)
# ==============================================================================
print("\n" + "="*80)
print("SOLUTION 1: higher library (Facebook Research)")
print("="*80)

# Setup
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
)
inner_optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Meta-learnable loss coefficient
loss_coeff = nn.Parameter(torch.tensor([1.0]))
meta_optimizer = torch.optim.Adam([loss_coeff], lr=0.01)

# Data
x_train = torch.randn(10, 2)
y_train = torch.randn(10, 1)
x_val = torch.randn(5, 2)
y_val = torch.randn(5, 1)

print(f"Initial loss coefficient: {loss_coeff.item():.4f}")
print(f"Initial model weight[0,0]: {model[0].weight[0,0].item():.4f}")

# Meta-learning step with higher
with higher.innerloop_ctx(model, inner_optimizer) as (fmodel, diffopt):
    # fmodel: differentiable functional model
    # diffopt: differentiable optimizer (supports SGD, Adam, etc.)

    # Inner loop: train with learnable loss
    for _ in range(3):
        outputs = fmodel(x_train)
        inner_loss = loss_coeff * ((outputs - y_train) ** 2).mean()

        # This step() is DIFFERENTIABLE!
        diffopt.step(inner_loss)

    # Meta-objective: validate
    val_outputs = fmodel(x_val)
    meta_loss = ((val_outputs - y_val) ** 2).mean()

    print(f"\nMeta-loss: {meta_loss.item():.6f}")
    print(f"Meta-loss grad_fn: {meta_loss.grad_fn}")

    # Backward through everything!
    meta_optimizer.zero_grad()
    meta_loss.backward()

    print(f"\nGradient w.r.t. loss_coeff: {loss_coeff.grad}")

    if loss_coeff.grad is not None:
        print(f"✅ SUCCESS! Gradient magnitude: {loss_coeff.grad.abs().item():.6f}")
        print("   - Supports SGD with momentum")
        print("   - Supports Adam, RMSprop, etc.")
        print("   - Optimizer state (momentum) is preserved")
        print("   - Clean, production-ready API")

        # Apply meta-update
        meta_optimizer.step()
        print(f"\nUpdated loss coefficient: {loss_coeff.item():.4f}")
    else:
        print("❌ FAILED")

# ==============================================================================
# SOLUTION 2: torch.func.functional_call()
# ==============================================================================
print("\n" + "="*80)
print("SOLUTION 2: torch.func.functional_call() (Built-in PyTorch)")
print("="*80)

# Fresh model
model2 = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
)

loss_coeff2 = nn.Parameter(torch.tensor([1.0]))
meta_optimizer2 = torch.optim.Adam([loss_coeff2], lr=0.01)

print(f"Initial loss coefficient: {loss_coeff2.item():.4f}")

# Get parameters as dict
params = dict(model2.named_parameters())

# Inner loop with functional_call
for _ in range(3):
    # Functional forward pass
    outputs = torch.func.functional_call(model2, params, x_train)
    inner_loss = loss_coeff2 * ((outputs - y_train) ** 2).mean()

    # Compute gradients (preserves graph)
    grads = torch.autograd.grad(
        inner_loss,
        params.values(),
        create_graph=True,
    )

    # Manual SGD update (graph-preserving)
    lr = 0.01
    params = {
        name: param - lr * grad
        for (name, param), grad in zip(params.items(), grads)
    }

# Meta-objective with updated params
val_outputs = torch.func.functional_call(model2, params, x_val)
meta_loss2 = ((val_outputs - y_val) ** 2).mean()

print(f"\nMeta-loss: {meta_loss2.item():.6f}")
print(f"Meta-loss grad_fn: {meta_loss2.grad_fn}")

# Backward
meta_optimizer2.zero_grad()
meta_loss2.backward()

print(f"\nGradient w.r.t. loss_coeff: {loss_coeff2.grad}")

if loss_coeff2.grad is not None:
    print(f"✅ SUCCESS! Gradient magnitude: {loss_coeff2.grad.abs().item():.6f}")
    print("   - Built into PyTorch (no dependencies)")
    print("   - No optimizer state preservation")
    print("   - Requires manual gradient application")
    print("   - Only supports vanilla SGD")

    meta_optimizer2.step()
    print(f"\nUpdated loss coefficient: {loss_coeff2.item():.4f}")
else:
    print("❌ FAILED")

# ==============================================================================
# COMPARISON
# ==============================================================================
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

comparison = """
| Feature                    | higher library | torch.func     | Manual Updates |
|----------------------------|----------------|----------------|----------------|
| Optimizer support          | All (SGD/Adam) | Vanilla SGD    | Vanilla SGD    |
| Preserves momentum         | ✅ Yes         | ❌ No          | ❌ No          |
| Preserves adaptive LR      | ✅ Yes         | ❌ No          | ❌ No          |
| Clean API                  | ✅ Yes         | ⚠️  Functional | ❌ No          |
| No dependencies            | ❌ No          | ✅ Yes         | ✅ Yes         |
| Production ready           | ✅ Yes         | ⚠️  For simple | ❌ No          |
| Used in research           | ✅ MAML papers | ⚠️  Some       | ❌ Rarely      |

RECOMMENDATION for PU-Bench:
- Use `higher` library for full meta-learning
- It's the industry standard (Facebook Research)
- Supports all optimizers (SGD, Adam, etc.)
- Clean, maintainable code
"""

print(comparison)

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("""
1. Replace manual updates in meta_trainer_fixed.py with higher library
2. Test with SGD (with momentum) - should work perfectly
3. Test with Adam - should also work!
4. Benchmark performance vs manual updates
5. Apply fix to full PU-Bench meta-trainer
""")
