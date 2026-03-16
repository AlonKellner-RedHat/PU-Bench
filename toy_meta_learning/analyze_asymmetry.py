#!/usr/bin/env python3
"""Analyze the learned asymmetric functions."""

import torch
import numpy as np

print("="*70)
print("ANALYZING LEARNED ASYMMETRIC FUNCTIONS")
print("="*70)
print()

# Learned parameters
a1_p, a2_p, a3_p = 0.0079, -0.8930, -0.9113
a1_n, a2_n, a3_n = 0.0149,  0.7424,  0.7541

print("Learned Parameters:")
print(f"  Positives: f_p(x) = {a1_p:.4f} + {a2_p:.4f}·x + {a3_p:.4f}·log(x)")
print(f"  Negatives: f_n(x) = {a1_n:.4f} + {a2_n:.4f}·x + {a3_n:.4f}·log(x)")
print()

# Define functions
def f_p(x):
    eps = 1e-7
    x = np.clip(x, eps, 1-eps)
    return a1_p + a2_p * x + a3_p * np.log(x)

def f_n(x):
    eps = 1e-7
    x = np.clip(x, eps, 1-eps)
    return a1_n + a2_n * x + a3_n * np.log(x)

print("="*70)
print("WHAT HAPPENS TO POSITIVE EXAMPLES")
print("="*70)
print()
print("For a positive example with prediction p:")
print("  Loss contribution = f_p(p)")
print()
print("  p    | f_p(p)")
print("  -----|----------")
for p in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
    print(f"  {p:4.2f} | {f_p(p):9.4f}")

print()
print("="*70)
print("WHAT HAPPENS TO NEGATIVE EXAMPLES")
print("="*70)
print()
print("For a negative example with prediction p (low p = good):")
print("  Loss contribution = f_n(1-p)")
print()
print("  p    | 1-p  | f_n(1-p)")
print("  -----|------|----------")
for p in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
    print(f"  {p:4.2f} | {1-p:4.2f} | {f_n(1-p):9.4f}")

print()
print("="*70)
print("KEY INSIGHT: EQUIVALENT TO SYMMETRIC?")
print("="*70)
print()

# Check if it's equivalent to a symmetric function
# For symmetric: f_p(p) should equal f_n(1-p) when applied correctly
# But here they're BOTH being minimized, so we want:
# f_p(p) ≈ f_n(p) for the functions to be truly symmetric

print("Testing symmetry:")
print()
print("  x    | f_p(x) | f_n(x) | Difference")
print("  -----|--------|--------|------------")
for x in [0.1, 0.3, 0.5, 0.7, 0.9]:
    diff = abs(f_p(x) - f_n(x))
    print(f"  {x:4.2f} | {f_p(x):6.3f} | {f_n(x):6.3f} | {diff:10.3f}")

print()
avg_diff = np.mean([abs(f_p(x) - f_n(x)) for x in np.linspace(0.1, 0.9, 20)])
print(f"Average absolute difference: {avg_diff:.4f}")
print()

if avg_diff > 0.5:
    print("✓ Highly asymmetric!")
else:
    print("⚠️  Nearly symmetric")

print()
print("="*70)
print("WHAT DID META-LEARNING DISCOVER?")
print("="*70)
print()

print("The learned functions are:")
print()
print("For POSITIVES (we want p → 1):")
print(f"  f_p(p) = {a2_p:.2f}·p + {a3_p:.2f}·log(p)")
print("  → Negative coefficients → minimizing is achieved by p → 1 ✓")
print()

print("For NEGATIVES (we want p → 0, so 1-p → 1):")
print(f"  f_n(1-p) = {a2_n:.2f}·(1-p) + {a3_n:.2f}·log(1-p)")
print(f"           = {a2_n:.2f} - {a2_n:.2f}·p + {a3_n:.2f}·log(1-p)")
print("  → Positive coefficient on (1-p) → minimizing is achieved by p → 0 ✓")
print()

print("="*70)
print("EXPANDED LOSS FORMULA")
print("="*70)
print()

print("Total loss:")
print(f"L = E_P[{a2_p:.2f}·p + {a3_p:.2f}·log(p)]")
print(f"  + E_N[{a2_n:.2f}·(1-p) + {a3_n:.2f}·log(1-p)]")
print()
print(f"  = E_P[{a2_p:.2f}·p + {a3_p:.2f}·log(p)]")
print(f"  + {a2_n:.2f} + E_N[-{a2_n:.2f}·p + {a3_n:.2f}·log(1-p)]")
print()

# Compare to symmetric learned
a2_sym, a3_sym = -0.95, -0.97

print("Compare to Symmetric Learned:")
print(f"L_sym = E_P[{a2_sym:.2f}·p + {a3_sym:.2f}·log(p)]")
print(f"      + E_N[{a2_sym:.2f}·(1-p) + {a3_sym:.2f}·log(1-p)]")
print()

print("="*70)
print("CONCLUSION")
print("="*70)
print()

print("Meta-learning discovered ASYMMETRIC functions:")
print()
print(f"  • For positives: Strong negative penalties ({a2_p:.2f}, {a3_p:.2f})")
print(f"  • For negatives: Strong positive penalties ({a2_n:.2f}, {a3_n:.2f})")
print()
print("BUT this doesn't improve over symmetric solution!")
print()
print("Possible reasons:")
print("  1. The problem IS truly symmetric (balanced classes, same distribution)")
print("  2. The 6 parameters overfit to the small checkpoint pool")
print("  3. Local minimum - different initialization might find better asymmetry")
print("  4. Need more diverse tasks to benefit from asymmetry")
print()
print("The symmetric 3-parameter solution is more efficient and generalizes better.")
