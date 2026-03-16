#!/usr/bin/env python3
"""Investigate the PROPER solution for differentiable optimization.

The manual parameter update approach is hacky. Let's find the real solution:
1. PyTorch's functional API (torch.func)
2. Stateless optimizers
3. Higher library (Facebook Research)
4. Learn2Learn library
"""

import torch
import torch.nn as nn
import sys

print("="*80)
print("INVESTIGATING PROPER SOLUTIONS FOR DIFFERENTIABLE OPTIMIZATION")
print("="*80)

print("\n1. Checking PyTorch version and available modules...")
print(f"PyTorch version: {torch.__version__}")

# Check for torch.func (new functional API)
if hasattr(torch, 'func'):
    print("✅ torch.func available (functional API)")
    print(f"   Available: {dir(torch.func)[:10]}...")
else:
    print("❌ torch.func not available")

# Check for functorch (older functional API)
try:
    import functorch
    print(f"✅ functorch available (version: {functorch.__version__})")
except ImportError:
    print("❌ functorch not available")

print("\n2. Checking for higher library (Facebook Research meta-learning)...")
try:
    import higher
    print(f"✅ higher available")
    print(f"   This is the PROPER solution for differentiable optimization!")
except ImportError:
    print("❌ higher not available")
    print("   Install with: pip install higher")

print("\n3. Checking for learn2learn library...")
try:
    import learn2learn
    print(f"✅ learn2learn available")
except ImportError:
    print("❌ learn2learn not available")
    print("   Install with: pip install learn2learn")

print("\n" + "="*80)
print("OPTION 1: torch.func.functional_call() [PyTorch Built-in]")
print("="*80)

if hasattr(torch, 'func') and hasattr(torch.func, 'functional_call'):
    print("Testing torch.func.functional_call()...")

    # Create simple model
    model = nn.Linear(1, 1)
    model.weight.data = torch.tensor([[2.0]])
    model.bias.data = torch.tensor([0.0])

    # Loss coefficient
    a = nn.Parameter(torch.tensor([3.0]))

    # Get parameters as dict
    params = dict(model.named_parameters())
    print(f"\nOriginal weight: {params['weight'].item():.4f}")

    # Compute gradients
    x = torch.tensor([[1.0]])
    y = torch.tensor([[2.0]])

    # Use functional_call to forward with specific parameters
    output = torch.func.functional_call(model, params, x)
    loss = a * ((output - y) ** 2).mean()

    print(f"Loss: {loss.item():.4f}")
    print(f"Loss requires_grad: {loss.requires_grad}")
    print(f"Loss grad_fn: {loss.grad_fn}")

    # Compute gradients w.r.t. parameters
    grads = torch.autograd.grad(loss, params.values(), create_graph=True)
    grad_dict = {name: grad for name, grad in zip(params.keys(), grads)}

    print(f"\nWeight gradient: {grad_dict['weight'].item():.4f}")
    print(f"Weight gradient has grad_fn: {grad_dict['weight'].grad_fn}")

    # Update parameters (functional style)
    lr = 0.1
    updated_params = {
        name: param - lr * grad_dict[name]
        for name, param in params.items()
    }

    print(f"Updated weight: {updated_params['weight'].item():.4f}")
    print(f"Updated weight has grad_fn: {updated_params['weight'].grad_fn}")

    # Use updated parameters in next forward pass
    output2 = torch.func.functional_call(model, updated_params, x)
    meta_loss = ((output2 - y) ** 2).mean()

    print(f"\nMeta-loss: {meta_loss.item():.4f}")
    print(f"Meta-loss grad_fn: {meta_loss.grad_fn}")

    # Test if we can backprop to 'a'
    a.grad = None
    meta_loss.backward()

    if a.grad is not None:
        print(f"\n✅ SUCCESS with torch.func.functional_call!")
        print(f"   ∂(meta_loss)/∂a = {a.grad.item():.6f}")
        print("\n   This is the PROPER PyTorch built-in solution!")
    else:
        print("\n❌ Failed - gradients still None")
else:
    print("❌ torch.func.functional_call not available")
    print("   Requires PyTorch >= 1.11")

print("\n" + "="*80)
print("OPTION 2: higher library [Recommended for Meta-Learning]")
print("="*80)

try:
    import higher
    print("Testing higher library...")

    # Create model and optimizer
    model = nn.Linear(1, 1)
    model.weight.data = torch.tensor([[2.0]])
    model.bias.data = torch.tensor([0.0])

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Loss coefficient
    a = nn.Parameter(torch.tensor([3.0]))

    # Data
    x = torch.tensor([[1.0]])
    y = torch.tensor([[2.0]])

    print("\nUsing higher.innerloop_ctx()...")

    # higher creates a differentiable copy of the model
    with higher.innerloop_ctx(model, optimizer) as (fmodel, diffopt):
        # fmodel is a differentiable functional model
        # diffopt is a differentiable optimizer

        # Inner loop
        output = fmodel(x)
        loss = a * ((output - y) ** 2).mean()

        print(f"Inner loss: {loss.item():.4f}")

        # This step() is differentiable!
        diffopt.step(loss)

        # Meta-objective with updated model
        output2 = fmodel(x)
        meta_loss = ((output2 - y) ** 2).mean()

        print(f"Meta-loss: {meta_loss.item():.4f}")
        print(f"Meta-loss grad_fn: {meta_loss.grad_fn}")

        # Backprop to 'a'
        a.grad = None
        meta_loss.backward()

        if a.grad is not None:
            print(f"\n✅ SUCCESS with higher library!")
            print(f"   ∂(meta_loss)/∂a = {a.grad.item():.6f}")
            print("\n   This is the PROPER meta-learning solution!")
            print("   - Supports all optimizers (SGD, Adam, etc.)")
            print("   - Preserves momentum, adaptive LR, etc.")
            print("   - Clean API designed for meta-learning")
        else:
            print("\n❌ Failed")

except ImportError:
    print("❌ higher library not installed")
    print("   Install with: pip install higher")
except Exception as e:
    print(f"❌ Error testing higher: {e}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("""
PROPER SOLUTIONS (ranked):

1. **higher library** (RECOMMENDED)
   - Purpose-built for meta-learning
   - Supports all PyTorch optimizers (SGD, Adam, etc.)
   - Clean API: higher.innerloop_ctx(model, optimizer)
   - Maintains optimizer state (momentum, adaptive LR)
   - Used by Facebook Research and many MAML papers

   Install: pip install higher
   Usage:
   ```python
   with higher.innerloop_ctx(model, optimizer) as (fmodel, diffopt):
       loss = criterion(fmodel(x), y)
       diffopt.step(loss)  # Differentiable step!
       meta_loss = eval(fmodel, val_data)
       meta_loss.backward()  # Gradients flow!
   ```

2. **torch.func.functional_call()** (PyTorch >= 1.11)
   - Built into PyTorch (no extra dependencies)
   - Functional API for models
   - Requires manual gradient application
   - Good for simple cases

3. **Manual parameter updates** (AVOID)
   - Loses optimizer features (momentum, Adam, etc.)
   - Error-prone
   - Only supports vanilla SGD
   - Not recommended

RECOMMENDATION: Use the `higher` library for production code.
""")
