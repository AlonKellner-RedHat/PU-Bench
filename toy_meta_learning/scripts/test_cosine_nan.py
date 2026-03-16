#!/usr/bin/env python3
"""Test to confirm cosine similarity produces NaN with near-zero gradients."""

import torch
import torch.nn.functional as F

print("Testing cosine similarity with various gradient norms:\n")

# Case 1: Normal gradients
pu_grad = torch.randn(100)
bce_grad = torch.randn(100)
cos_sim = F.cosine_similarity(pu_grad.unsqueeze(0), bce_grad.unsqueeze(0))
print(f"1. Normal gradients:")
print(f"   PU norm: {pu_grad.norm():.6f}, BCE norm: {bce_grad.norm():.6f}")
print(f"   Cosine sim: {cos_sim.item():.6f}, Cosine loss: {(1 - cos_sim).item():.6f}")
print(f"   Has NaN: {torch.isnan(cos_sim).item()}\n")

# Case 2: Very small gradients (approaching saturation)
pu_grad = torch.randn(100) * 1e-10
bce_grad = torch.randn(100) * 1e-10
cos_sim = F.cosine_similarity(pu_grad.unsqueeze(0), bce_grad.unsqueeze(0))
print(f"2. Very small gradients (1e-10):")
print(f"   PU norm: {pu_grad.norm():.10f}, BCE norm: {bce_grad.norm():.10f}")
print(f"   Cosine sim: {cos_sim.item()}, Cosine loss: {(1 - cos_sim).item()}")
print(f"   Has NaN: {torch.isnan(cos_sim).item()}\n")

# Case 3: Zero gradients (perfect saturation)
pu_grad = torch.zeros(100)
bce_grad = torch.zeros(100)
cos_sim = F.cosine_similarity(pu_grad.unsqueeze(0), bce_grad.unsqueeze(0))
print(f"3. Zero gradients:")
print(f"   PU norm: {pu_grad.norm():.6f}, BCE norm: {bce_grad.norm():.6f}")
print(f"   Cosine sim: {cos_sim.item()}, Cosine loss: {(1 - cos_sim).item()}")
print(f"   Has NaN: {torch.isnan(cos_sim).item()}\n")

# Case 4: One zero, one non-zero
pu_grad = torch.zeros(100)
bce_grad = torch.randn(100)
cos_sim = F.cosine_similarity(pu_grad.unsqueeze(0), bce_grad.unsqueeze(0))
print(f"4. One zero, one non-zero:")
print(f"   PU norm: {pu_grad.norm():.6f}, BCE norm: {bce_grad.norm():.6f}")
print(f"   Cosine sim: {cos_sim.item()}, Cosine loss: {(1 - cos_sim).item()}")
print(f"   Has NaN: {torch.isnan(cos_sim).item()}\n")

# Case 5: Extremely small but non-zero (typical in saturated model)
pu_grad = torch.randn(100) * 1e-20
bce_grad = torch.randn(100) * 1e-20
cos_sim = F.cosine_similarity(pu_grad.unsqueeze(0), bce_grad.unsqueeze(0))
print(f"5. Extremely small gradients (1e-20):")
print(f"   PU norm: {pu_grad.norm():.25f}, BCE norm: {bce_grad.norm():.25f}")
print(f"   Cosine sim: {cos_sim.item()}, Cosine loss: {(1 - cos_sim).item()}")
print(f"   Has NaN: {torch.isnan(cos_sim).item()}\n")

print("="*70)
print("CONCLUSION:")
print("F.cosine_similarity produces NaN when both vectors have norm=0 (or very close to 0).")
print("This happens during meta-training when model outputs saturate (very large logits).")
print("="*70)
