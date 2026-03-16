# Asymmetric PU Loss for Extended Meta-Learning

## Change from Symmetric to Asymmetric

**Previous:** 3-parameter symmetric loss
```
f(x) = a₁ + a₂·x + a₃·log(x)
L_PU = E_P[f(p)] + E_U[f(1-p)]
```

**Current:** 6-parameter asymmetric loss
```
f_p(x) = a1_p + a2_p·x + a3_p·log(x)  # For labeled positives
f_u(x) = a1_u + a2_u·x + a3_u·log(x)  # For unlabeled

L_PU = E_P[f_p(p)] + E_U[f_u(1-p)]
```

## Why Asymmetric for PU?

The key insight from previous experiments: **Labeled positives and unlabeled have fundamentally different characteristics**

### Labeled Positives (Clean)
- **Trustworthy labels**: We know these are true positives
- **Can use aggressive penalties**: Push predictions toward p → 1
- Expected: Large negative coefficients (e.g., a2_p ≈ -0.95, a3_p ≈ -0.97)

### Unlabeled (Noisy Mixture)
- **Contains hidden positives**: ~70% of unlabeled are actually positive (since only 30% labeled)
- **Systematically biased**: Naive training treats all unlabeled as negative
- **Need robustness**: Soft penalties to avoid hurting hidden positives
- Expected: Opposite signs or different magnitudes than labeled positives

## Previous Asymmetric PU Results (Small Pool)

From the earlier toy experiments with 12 checkpoints:

**Learned Functions:**
```
Labeled Pos: f_p(p) = 0.0156 - 0.9494·p - 0.9663·log(p)
Unlabeled:   f_u(1-p) = -0.0195 + 0.9522·(1-p) + 0.9618·log(1-p)
```

**Key Finding:** Nearly perfect anti-symmetry
- f_u(x) ≈ -f_p(x)
- The loss learned to **subtract** the unlabeled term: `E_P[f(p)] - E_U[f(1-p)]`
- This isn't "different functions" - it's **sign-flipping** the unlabeled contribution!

**Performance:**
```
Asymmetric PU (6-param): 5.092 BCE  ← Best!
Symmetric PU (3-param):  5.183 BCE  (+1.75% worse)
Pure BCE:                5.168 BCE  (+1.47% worse)
```

## What to Expect from Extended Pool (560 Checkpoints)

### Hypothesis 1: Anti-Symmetry Emerges Again
With more diverse checkpoints, we expect to see the same pattern:
- a2_p ≈ -a2_u (opposite signs)
- a3_p ≈ -a3_u (opposite signs)
- Effectively learning: `E_P[f(p)] - E_U[f(1-p)]`

This makes sense because:
- Labeled positives should minimize when p → 1
- Unlabeled should maximize when p → 1 (to avoid penalizing hidden positives)

### Hypothesis 2: True Asymmetry (Different Magnitudes)
Alternatively, we might see:
- Different coefficient magnitudes (not just opposite signs)
- More aggressive for labeled positives (trustworthy)
- More conservative for unlabeled (noisy)

### Hypothesis 3: Oracle vs Naive Affects Learning
The extended pool includes:
- **280 oracle checkpoints**: Trained with perfect PN labels
- **280 naive checkpoints**: Trained with biased PU labels

The learned loss might:
- Help naive checkpoints MORE (they need more correction)
- Maintain oracle checkpoints (already good)
- Discover asymmetry that specifically fixes naive bias

## Success Metrics

### 1. Performance on Naive Checkpoints
**Test run baseline** (symmetric 3-param):
- Naive epoch 1: 0.625 BCE
- Naive epoch 100: 3.342 BCE (5.3× degradation)

**Target with asymmetric**:
- Reduce degradation: Naive epoch 100 < 2.0 BCE
- Ideally: Maintain or improve from epoch 1

### 2. Symmetry Measure
Track `abs(a1_p - a1_u) + abs(a2_p - a2_u) + abs(a3_p - a3_u)` over training:
- High symmetry (> 1.5): True asymmetry discovered
- Near anti-symmetric (≈ sum of absolute values): Sign-flipping strategy
- Low symmetry (< 0.1): Collapsed to symmetric solution

### 3. Oracle-Naive Gap
**Test run baseline:**
- Oracle average: 0.335 BCE
- Naive average: 1.750 BCE
- Gap: 421% (naive 5.2× worse)

**Target:**
- Reduce gap to < 200% (naive < 2× worse)
- Ideally: < 100% (comparable performance)

### 4. Learned Coefficients
Compare to previous results:
```
Previous (12 checkpoints):
  a1_p =  0.0156,  a2_p = -0.9494,  a3_p = -0.9663
  a1_u = -0.0195,  a2_u = +0.9522,  a3_u = +0.9618

Expected (560 checkpoints):
  Stronger magnitudes (more data)
  Clearer anti-symmetry pattern
  More stable convergence
```

## Comparison to PN Learning

**Important contrast:**

**PN Learning** (both groups clean labels):
- Asymmetric (6-param): 5.005 BCE
- Symmetric (3-param): 4.996 BCE ← Better!
- **Conclusion:** Asymmetry doesn't help PN

**PU Learning** (labeled clean, unlabeled noisy):
- Asymmetric (6-param): 5.092 BCE ← Better!
- Symmetric (3-param): 5.183 BCE
- **Conclusion:** Asymmetry helps PU

This validates that **problem structure determines optimal parameterization**:
- Clean/clean (PN) → symmetric
- Clean/noisy (PU) → asymmetric

## Training Configuration

**Extended Pool:**
- 560 total checkpoints
- 4 difficulties × 2 overlaps × 5 seeds = 40 base tasks
- 2 training methods (oracle + naive)
- 7 training stages (1 to 200 epochs)

**Meta-Training:**
- 500 iterations
- Batch size: 16 checkpoints
- Learning rate: 0.001 (Adam)
- Inner loop: 3 steps, lr=0.01

**Output:**
- Log file: `extended_training_asymmetric.log`
- Monitor: `./quick_status_asym.sh`
- Expected time: 2-3 hours total

## Key Questions to Answer

1. **Does anti-symmetry emerge again?**
   - Are the learned functions f_p and f_u negatives of each other?

2. **Does asymmetry help naive checkpoints specifically?**
   - Compare oracle vs naive improvement from learned loss

3. **How does training stage affect the benefit?**
   - Does asymmetric loss help early, mid, or late checkpoints more?

4. **Is the benefit consistent across task difficulties?**
   - Easy tasks (mean_sep=3.0) vs hard tasks (mean_sep=1.5)

5. **Does the larger pool lead to stronger coefficients?**
   - More diverse data → more confident learning?

## Expected Final Results

Based on previous patterns, we expect:

```
FINAL LEARNED ASYMMETRIC PU LOSS
=================================

Learned Functions:
  Labeled Pos: f_p(x) = a1_p + a2_p·x + a3_p·log(x)
  Unlabeled:   f_u(x) = a1_u + a2_u·x + a3_u·log(x)

Expected Pattern (anti-symmetric):
  a1_p ≈  +c,  a1_u ≈ -c
  a2_p ≈ -0.9 to -1.0,  a2_u ≈ +0.9 to +1.0
  a3_p ≈ -0.9 to -1.0,  a3_u ≈ +0.9 to +1.0

Performance by Training Method:
  Oracle checkpoints:  ~0.3-0.4 BCE (stable)
  Naive checkpoints:   ~1.0-1.5 BCE (improved from 1.75)

Performance by Epoch:
  Early (1-10):   Minimal degradation
  Mid (20-50):    Moderate improvement
  Late (100-200): Large improvement (fixing overfit)

Overall Gap:
  Oracle-Naive gap: ~150-200% (down from 421%)
```

The full results will validate whether asymmetric PU loss can successfully bridge the performance gap between oracle and naive training, especially for the heavily overfit late-stage naive checkpoints.
