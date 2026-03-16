#!/usr/bin/env python3
"""Deep dive: Understanding differentiable=True behavior.

Let's trace exactly what happens to the computational graph.
"""

import torch
import torch.nn as nn

print("="*80)
print("DEEP DIVE: Tracing differentiable=True behavior")
print("="*80)

# Meta-parameter
a = nn.Parameter(torch.tensor([2.0]))

# Simple model
model = nn.Linear(1, 1)
model.weight.data = torch.tensor([[0.5]])
model.bias.data = torch.tensor([0.0])

# Data
x = torch.tensor([[1.0]])
y = torch.tensor([[1.0]])

print("\n--- Setup ---")
print(f"Meta-param a: {a.item():.4f}, is_leaf={a.is_leaf}, requires_grad={a.requires_grad}")
print(f"Model weight: {model.weight.item():.4f}, is_leaf={model.weight.is_leaf}")

# Convert to non-leaf
weight_nonleaf = model.weight.clone() * 1.0
weight_nonleaf.retain_grad()
bias_nonleaf = model.bias.clone() * 1.0
bias_nonleaf.retain_grad()

print(f"\nNon-leaf weight: is_leaf={weight_nonleaf.is_leaf}, grad_fn={weight_nonleaf.grad_fn}")

# Create optimizer
optimizer = torch.optim.SGD([weight_nonleaf, bias_nonleaf], lr=0.1, differentiable=True)

print("\n--- Step 1: Forward and backward ---")

# Forward
out = x @ weight_nonleaf.T + bias_nonleaf
loss = a * ((out - y) ** 2).mean()

print(f"Loss: {loss.item():.4f}")
print(f"Loss grad_fn: {loss.grad_fn}")
print(f"Loss requires_grad: {loss.requires_grad}")

# Backward
optimizer.zero_grad()
loss.backward(create_graph=True)

print(f"\nweight_nonleaf.grad: {weight_nonleaf.grad}")
print(f"weight_nonleaf.grad.grad_fn: {weight_nonleaf.grad.grad_fn}")

print("\n--- Step 2: Optimizer step ---")
print(f"Before step - weight_nonleaf: {weight_nonleaf.item():.4f}")
print(f"Before step - weight_nonleaf.grad_fn: {weight_nonleaf.grad_fn}")
print(f"Before step - id(weight_nonleaf): {id(weight_nonleaf)}")

optimizer.step()

print(f"\nAfter step - weight_nonleaf: {weight_nonleaf.item():.4f}")
print(f"After step - weight_nonleaf.grad_fn: {weight_nonleaf.grad_fn}")
print(f"After step - id(weight_nonleaf): {id(weight_nonleaf)}")
print(f"After step - weight_nonleaf.is_leaf: {weight_nonleaf.is_leaf}")

# Check if the parameter object changed
print(f"\nDid optimizer create new tensor? {id(weight_nonleaf) == id(optimizer.param_groups[0]['params'][0])}")

print("\n--- Step 3: Meta-objective ---")

# Use the updated parameter
x_val = torch.tensor([[1.5]])
y_val = torch.tensor([[1.5]])

out_val = x_val @ weight_nonleaf.T + bias_nonleaf
meta_loss = ((out_val - y_val) ** 2).mean()

print(f"Meta-loss: {meta_loss.item():.4f}")
print(f"Meta-loss.grad_fn: {meta_loss.grad_fn}")

# Check grad_fn chain
print(f"\nGrad function chain:")
grad_fn = meta_loss.grad_fn
depth = 0
while grad_fn is not None and depth < 10:
    print(f"  [{depth}] {grad_fn}")
    grad_fn = getattr(grad_fn, 'next_functions', [[None]])[0][0] if hasattr(grad_fn, 'next_functions') else None
    depth += 1

print("\n--- Step 4: Meta-backward ---")

a.grad = None
try:
    meta_loss.backward()
    print(f"✅ Backward succeeded")
    print(f"a.grad: {a.grad}")
except Exception as e:
    print(f"❌ Backward failed: {e}")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

if a.grad is None or abs(a.grad.item()) < 1e-6:
    print("""
The meta-gradient is None/zero, which means the computational graph is broken.

Possible causes:
1. optimizer.step() updates parameters IN-PLACE, breaking the graph
2. The updated weight_nonleaf loses its connection to the original loss
3. differentiable=True doesn't preserve the full graph needed for meta-learning

The graph from 'a' → loss → grad → optimizer.step() → updated_param is broken
somewhere in the optimizer.step() operation.
""")
else:
    print(f"""
✅ SUCCESS! Meta-gradient = {a.grad.item():.6f}

This means differentiable=True DOES preserve the graph when used correctly.
""")
