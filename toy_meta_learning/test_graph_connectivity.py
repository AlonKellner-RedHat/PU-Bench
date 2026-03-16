#!/usr/bin/env python3
"""Test if computational graph connects meta-loss to loss_fn parameters.

The issue: meta_loss (val BCE) should flow back through:
  meta_loss → model params → inner_optimizer.step() → loss_fn params

But optimizer.step() might break the graph!
"""

import torch
import torch.nn as nn

print("="*80)
print("TEST: Does optimizer.step() break the computational graph?")
print("="*80)

# Simple example: learn a loss coefficient that trains a model
a = nn.Parameter(torch.tensor([2.0]))  # Loss coefficient (like our a1, a2, a3)
print(f"\nInitial coefficient a: {a.item():.4f}")

# Create a simple model
model = nn.Linear(1, 1)
model.weight.data = torch.tensor([[0.5]])
model.bias.data = torch.tensor([0.0])

print(f"Initial model weight: {model.weight.item():.4f}")

# Inner loop: train model with learnable loss
inner_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Training data
x_train = torch.tensor([[1.0], [2.0]])
y_train = torch.tensor([[1.0], [2.0]])

print("\n--- INNER LOOP (with create_graph=True) ---")
inner_optimizer.zero_grad()

# Forward pass
out = model(x_train)

# Loss using learnable coefficient: L = a * MSE
mse = ((out - y_train) ** 2).mean()
inner_loss = a * mse

print(f"Inner loss: {inner_loss.item():.4f}")
print(f"Inner loss requires_grad: {inner_loss.requires_grad}")
print(f"Inner loss grad_fn: {inner_loss.grad_fn}")

# Backward with create_graph
inner_loss.backward(create_graph=True)

print(f"\nModel weight gradient: {model.weight.grad}")

# CRITICAL: Apply optimizer step
print("\nApplying inner_optimizer.step()...")
inner_optimizer.step()

print(f"Model weight after step: {model.weight.item():.4f}")
print(f"Model weight requires_grad: {model.weight.requires_grad}")
print(f"Model weight is_leaf: {model.weight.is_leaf}")
print(f"Model weight grad_fn: {model.weight.grad_fn}")

# Meta-objective: evaluate updated model
print("\n--- META-OBJECTIVE (validation loss) ---")
x_val = torch.tensor([[1.5]])
y_val = torch.tensor([[1.5]])

out_val = model(x_val)
meta_loss = ((out_val - y_val) ** 2).mean()

print(f"Meta-loss: {meta_loss.item():.4f}")
print(f"Meta-loss requires_grad: {meta_loss.requires_grad}")
print(f"Meta-loss grad_fn: {meta_loss.grad_fn}")

# Try to compute gradient w.r.t. coefficient 'a'
print("\n--- BACKWARD to compute ∂(meta_loss)/∂a ---")
print(f"Before backward - a.grad: {a.grad}")

# CRITICAL: Zero the gradient first (it has leftovers from inner loop)
a.grad = None
print(f"After zero_grad - a.grad: {a.grad}")

meta_loss.backward()

print(f"After meta backward - a.grad: {a.grad}")

if a.grad is None:
    print("\n❌ GRAPH IS BROKEN: optimizer.step() severed the connection!")
    print("   Model parameters after step() are NEW tensors without grad_fn")
else:
    print(f"\n✅ GRAPH IS CONNECTED: ∂(meta_loss)/∂a = {a.grad.item():.6f}")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)
print("""
The problem: torch.optim.Optimizer.step() creates NEW parameter tensors
that are NOT connected to the computational graph!

After optimizer.step():
- model.weight is a NEW leaf tensor
- model.weight.grad_fn = None (no connection to previous graph)
- meta_loss can't backprop through these new tensors

Solution: Use functional optimization (manual parameter updates) OR
          use higher-order optimization libraries that preserve the graph.
""")
