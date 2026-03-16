#!/usr/bin/env python3
"""Test: PyTorch's differentiable=True with non-leaf parameters.

Based on PyTorch issue #150183:
https://github.com/pytorch/pytorch/issues/150183

The key insight: differentiable=True only works with NON-LEAF tensors.
Regular model parameters are leaves, causing in-place errors.

Solution: Convert parameters to non-leaf by doing operations on them.
"""

import torch
import torch.nn as nn

print("="*80)
print("TEST: differentiable=True with NON-LEAF parameters")
print("="*80)

# Meta-parameter (loss coefficient)
a = nn.Parameter(torch.tensor([2.0]))
print(f"\nInitial meta-parameter a: {a.item():.4f}")

# Create model
model = nn.Linear(1, 1)
model.weight.data = torch.tensor([[0.5]])
model.bias.data = torch.tensor([0.0])

print(f"Initial model weight: {model.weight.item():.4f}")

# Training data
x_train = torch.tensor([[1.0], [2.0]])
y_train = torch.tensor([[1.0], [2.0]])

print("\n--- APPROACH 1: Standard parameters (FAILS) ---")

try:
    # This FAILS because model.parameters() are LEAF tensors
    optimizer_fail = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        differentiable=True
    )

    out = model(x_train)
    mse = ((out - y_train) ** 2).mean()
    inner_loss = a * mse

    optimizer_fail.zero_grad()
    inner_loss.backward(create_graph=True)
    optimizer_fail.step()

    print("  ✅ Success (unexpected!)")
except RuntimeError as e:
    print(f"  ❌ Failed (expected): {e}")

print("\n--- APPROACH 2: Non-leaf parameters (SHOULD WORK) ---")

# Reset model
model = nn.Linear(1, 1)
model.weight.data = torch.tensor([[0.5]])
model.bias.data = torch.tensor([0.0])

# KEY: Convert parameters to non-leaf by performing operation
# The * 1 operation makes the result a non-leaf tensor with grad_fn
param_list = []
for p in model.parameters():
    # Clone and multiply by 1 to create non-leaf
    non_leaf_param = p.clone() * 1.0  # This creates grad_fn!
    non_leaf_param.retain_grad()  # Keep gradient
    param_list.append(non_leaf_param)

print(f"Parameter 0 is_leaf: {param_list[0].is_leaf}")  # Should be False
print(f"Parameter 0 grad_fn: {param_list[0].grad_fn}")  # Should have grad_fn

# Create optimizer with non-leaf parameters
optimizer = torch.optim.SGD(
    param_list,
    lr=0.1,
    momentum=0.9,
    differentiable=True  # Now this should work!
)

# Forward pass (need to use param_list, not model.parameters())
def forward_with_params(x, weight, bias):
    return x @ weight.T + bias

out = forward_with_params(x_train, param_list[0], param_list[1])
mse = ((out - y_train) ** 2).mean()
inner_loss = a * mse

print(f"\nInner loss: {inner_loss.item():.4f}")
print(f"Inner loss grad_fn: {inner_loss.grad_fn}")

# Backward with create_graph
optimizer.zero_grad()
inner_loss.backward(create_graph=True)

print(f"\nParameter 0 gradient: {param_list[0].grad}")
print(f"Parameter 0 gradient grad_fn: {param_list[0].grad.grad_fn}")

# Optimizer step (should work with non-leaf parameters!)
try:
    optimizer.step()
    print(f"\n✅ optimizer.step() succeeded!")
    print(f"Updated param 0: {param_list[0].item():.4f}")
    print(f"Updated param 0 grad_fn: {param_list[0].grad_fn}")
except RuntimeError as e:
    print(f"\n❌ optimizer.step() failed: {e}")

# Test meta-objective
print("\n--- META-OBJECTIVE ---")

x_val = torch.tensor([[1.5]])
y_val = torch.tensor([[1.5]])

# Forward with updated parameters
out_val = forward_with_params(x_val, param_list[0], param_list[1])
meta_loss = ((out_val - y_val) ** 2).mean()

print(f"Meta-loss: {meta_loss.item():.4f}")
print(f"Meta-loss grad_fn: {meta_loss.grad_fn}")

# Test backward to compute meta-gradient
print("\n--- BACKWARD to compute ∂(meta_loss)/∂a ---")
a.grad = None

meta_loss.backward()

print(f"After meta backward - a.grad: {a.grad}")

if a.grad is not None and abs(a.grad.item()) > 1e-6:
    print(f"\n✅ SUCCESS: Meta-gradient computed! ∂(meta_loss)/∂a = {a.grad.item():.6f}")
    print(f"\nThis confirms:")
    print(f"  1. Non-leaf parameters work with differentiable=True")
    print(f"  2. Graph is preserved through optimizer.step()")
    print(f"  3. Meta-gradients flow correctly")
else:
    print(f"\n❌ FAILED: No meta-gradient")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("""
To use differentiable=True correctly:

1. Convert leaf parameters to non-leaf:
   non_leaf = param.clone() * 1.0  # Creates grad_fn
   non_leaf.retain_grad()

2. Create optimizer with non-leaf parameters:
   optimizer = torch.optim.SGD([non_leaf], lr=0.1, differentiable=True)

3. Use functional forward passes with these parameters

4. Meta-gradients flow correctly!

However, this approach is MORE COMPLEX than pure functional optimization.

The question is: Does this offer any advantage over functional optimization?
""")
