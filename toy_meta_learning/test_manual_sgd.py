#!/usr/bin/env python3
"""Test: Manual SGD updates preserve the computational graph.

Instead of optimizer.step(), manually update: param = param - lr * grad
This keeps the computational graph intact.
"""

import torch
import torch.nn as nn

print("="*80)
print("TEST: Manual SGD updates preserve computational graph")
print("="*80)

# Simple example
a = nn.Parameter(torch.tensor([2.0]))  # Loss coefficient
print(f"\nInitial coefficient a: {a.item():.4f}")

# Create model
model = nn.Linear(1, 1)
model.weight.data = torch.tensor([[0.5]])
model.bias.data = torch.tensor([0.0])

print(f"Initial model weight: {model.weight.item():.4f}")

# Training data
x_train = torch.tensor([[1.0], [2.0]])
y_train = torch.tensor([[1.0], [2.0]])

print("\n--- INNER LOOP (manual SGD with create_graph=True) ---")

# Forward pass
out = model(x_train)
mse = ((out - y_train) ** 2).mean()
inner_loss = a * mse

print(f"Inner loss: {inner_loss.item():.4f}")
print(f"Inner loss grad_fn: {inner_loss.grad_fn}")

# Backward with create_graph
grads = torch.autograd.grad(
    inner_loss,
    model.parameters(),
    create_graph=True,  # KEY: Preserve graph
)

print(f"\nModel weight gradient: {grads[0]}")

# Manual SGD update (preserves graph!)
lr = 0.1
with torch.no_grad():
    # Don't use .data = ... as that breaks graph
    # Instead, use in-place operations or functional updates
    pass

# Actually, we need to do this differently:
# Create new parameters that MAINTAIN the computational graph
new_weight = model.weight - lr * grads[0]
new_bias = model.bias - lr * grads[1]

print(f"\nNew weight (computed): {new_weight.item():.4f}")
print(f"New weight grad_fn: {new_weight.grad_fn}")

# Update model (this is tricky - need to preserve graph)
# Option 1: Don't update in-place, use functional approach
# Option 2: Use higher library like higher or functorch

print("\n❌ PROBLEM: Standard PyTorch doesn't support in-place updates")
print("   that preserve the computational graph!")
print("\nSOLUTIONS:")
print("  1. Use 'higher' library for differentiable optimization")
print("  2. Use 'functorch' for functional parameter updates")
print("  3. Manually apply gradients and re-construct the model")

# Let's test Option 3: Manual parameter reconstruction
print("\n--- OPTION 3: Functional forward pass ---")

def functional_forward(x, weight, bias):
    """Forward pass using explicit parameters (not model.parameters())."""
    return x @ weight.T + bias

# Compute updated parameters (with graph)
updated_weight = model.weight - lr * grads[0]
updated_bias = model.bias - lr * grads[1]

print(f"Updated weight: {updated_weight.item():.4f}")
print(f"Updated weight grad_fn: {updated_weight.grad_fn}")

# Meta-objective with updated parameters
x_val = torch.tensor([[1.5]])
y_val = torch.tensor([[1.5]])

out_val = functional_forward(x_val, updated_weight, updated_bias)
meta_loss = ((out_val - y_val) ** 2).mean()

print(f"\nMeta-loss: {meta_loss.item():.4f}")
print(f"Meta-loss grad_fn: {meta_loss.grad_fn}")

# Test backward
print("\n--- BACKWARD to compute ∂(meta_loss)/∂a ---")
a.grad = None

meta_loss.backward()

print(f"After meta backward - a.grad: {a.grad}")

if a.grad is not None:
    print(f"\n✅ SUCCESS: Graph preserved! ∂(meta_loss)/∂a = {a.grad.item():.6f}")
else:
    print("\n❌ FAILED: Still no gradient")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("""
To preserve the computational graph through optimization:

1. Don't use optimizer.step() - it breaks the graph
2. Instead, use functional parameter updates:
   new_param = old_param - lr * grad
3. Use these updated parameters in subsequent forward passes
4. The graph will be preserved and meta-gradients will flow!

For our meta-trainer, we need to:
- Replace inner_optimizer.step() with manual updates
- Use functional forward passes with explicit parameters
- This is exactly what libraries like 'higher' do internally
""")
