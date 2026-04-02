# Theoretical Analysis: Prior Usage in VPU-NoMixUp-Mean-Prior

## Executive Summary

The "prior" parameter in `vpu_nomixup_mean_prior` is **NOT theoretically sound** as a representation of the true class prior π. Instead, it functions as an **empirical hyperparameter** that balances loss terms and provides regularization. This explains why:

1. **Over-estimating the prior improves performance** (63% of cases prefer higher values)
2. **The optimal "prior" differs from the true class prior** (mean difference: +0.088)
3. **Calibration dramatically improves** (+22.48%) despite theoretical incorrectness

---

## Current Implementation

### VPU-NoMixUp-Mean-Prior Loss

```python
L = E_all[φ(x)] - π · E_P[log φ(x)]
  = mean(φ(x)) - π · mean(log φ_p(x))
```

where:
- `E_all`: Expectation over **all samples** (labeled positives + unlabeled)
- `E_P`: Expectation over **labeled positive samples only**
- `π`: "Class prior" parameter (supposedly P(y=1))
- `φ(x) = σ(f(x))`: Model output after sigmoid

### VPU-NoMixUp-Mean Loss (Baseline)

```python
L = E_all[φ(x)] - E_P[log φ(x)]
  = mean(φ(x)) - mean(log φ_p(x))
```

The **only difference** is the π multiplier on the positive term.

---

## Theoretical Issues

### Issue 1: E_all ≠ E_X (Population Expectation)

**Problem**: The code uses `E_all[φ(x)]` where "all" means labeled + unlabeled samples in the batch.

In PU learning:
- `E_X[φ(x)]` = expectation over the **population** distribution P(x)
- `E_all[φ(x)]` = expectation over **batch samples** (biased by sampling)

**Why this matters:**

Let:
- `n_p` = number of labeled positives in batch
- `n_u` = number of unlabeled samples in batch
- `n_all` = `n_p + n_u`

Then:
```
E_all[φ(x)] = (n_p · E_P[φ(x)] + n_u · E_U[φ(x)]) / n_all
```

where `E_U[φ(x)] = π·E_P[φ(x)] + (1-π)·E_N[φ(x)]`

**This depends on the sampling ratio `n_p/n_u`, NOT on the true class prior π!**

### Issue 2: π Should Not Multiply the Positive Term

**Theoretical Derivation:**

In standard PU learning, the goal is to learn a classifier `φ(x) ≈ P(y=1|x)` such that:

```
E_X[φ(x)] = π  (constraint)
```

The **VPU variational bound** (under anchor assumption) is:

```
L_VPU = log E_X[φ(x)] - E_P[log φ(x)]
```

The original VPU implementation (see `loss_vpu.py`) uses:

```python
L = torch.logsumexp(log_phi_x, dim=0) - log(len(log_phi_x)) - mean(log_phi_p)
  = log-mean(φ(x)) - mean(log φ_p(x))
```

**No π multiplier appears in the loss!**

The π constraint is **implicitly enforced** by minimizing this loss, which pushes `E_X[φ(x)]` toward the value that satisfies the variational bound.

### Issue 3: What the π Multiplier Actually Does

Multiplying `E_P[log φ(x)]` by π **changes the balance** between the two loss terms:

```
L = E_all[φ(x)] - π · E_P[log φ(x)]
```

**Effects of larger π:**
1. **Stronger penalty** on positives with low `φ(x)` values
2. **Forces higher recall** (model predicts more positives)
3. **Acts as implicit regularization** toward conservative classification
4. **Adjusts precision/recall tradeoff**

This is an **empirical tuning mechanism**, not a theoretical class prior!

---

## Empirical Evidence

### Comparison: With vs Without π Multiplier

Comparing 1,050 head-to-head experiments:

| Method | F1 Score | A-NICE (Calibration) |
|--------|----------|---------------------|
| `vpu_nomixup_mean` (NO π) | Baseline | Baseline |
| `vpu_nomixup_mean_prior` (WITH π) | +0.9% | **+23%** ✓ |

**Key Findings:**
- π multiplier provides **minimal F1 improvement** (+0.9%)
- π multiplier provides **massive calibration improvement** (+23%)
- **BUT**: This is empirical, not theoretically justified

### Why Over-Estimation Works

From robustness experiments (378 experiments with varying π):

**Optimal "prior" direction:**
- **63% prefer higher** than true prior (mean: +0.206)
- **24% prefer lower** than true prior (mean: -0.220)
- **13% prefer exact** true prior

**Effect by label frequency:**

| Label Frequency (c) | Optimal Direction | Mean Difference |
|---------------------|------------------|-----------------|
| c = 0.1 (scarce labels) | **Higher** (83%) | **+0.252** |
| c = 0.5 (medium) | Higher (53%) | +0.053 |
| c = 0.9 (many labels) | Mixed | -0.032 |

**Explanation:**

When labeled data is scarce:
- `E_all` is dominated by unlabeled samples
- Positive signal is weak
- **Higher π = stronger positive penalty = better recall**

This confirms π is acting as a **regularization hyperparameter**, not as the true class prior!

---

## Theoretical Recommendations

### Option 1: Remove π Multiplier (Theoretically Correct)

```python
L = E_all[φ(x)] - E_P[log φ(x)]
```

**Pros:**
- Theoretically justified
- Matches VPU derivation (without log-of-mean)
- No hyperparameter to tune

**Cons:**
- Loses +23% calibration improvement
- Slightly lower F1 (-0.9%)

### Option 2: Explicit Constraint Enforcement

```python
L = (E_all[φ(x)] - π)² - E_P[log φ(x)] + λ·calibration_term
```

**Pros:**
- Explicitly enforces `E_all ≈ π`
- Theoretically motivated
- Can add separate calibration term

**Cons:**
- Adds complexity
- Need to tune λ

### Option 3: Treat π as Hyperparameter (Current Approach)

```python
L = E_all[φ(x)] - α · E_P[log φ(x)]
```

Rename `π` → `α` to clarify it's a **balance parameter**, not class prior.

**Pros:**
- Empirically effective (+23% calibration)
- Simple to implement
- Can be tuned via cross-validation

**Cons:**
- Theoretically unjustified
- Misleading name ("prior" suggests class prior)
- Optimal value ≠ true prior

### Option 4: Reweight E_all to Get Unbiased E_X

Correct for sampling bias:

```python
# Compute unbiased population expectation
E_X = (n_u / n_all) · E_U[φ(x)]

# where E_U is computed over unlabeled samples only
# This removes the positive sample bias in E_all

L = E_X[φ(x)] - E_P[log φ(x)]
```

**Pros:**
- Theoretically sound
- Removes sampling bias
- True E_X estimate

**Cons:**
- More complex implementation
- May increase variance
- Needs careful handling of edge cases

---

## Recommended Solution

### For Production Use: **Option 3 with Clarification**

**Recommendation:**
1. **Rename the parameter**: `prior` → `positive_weight` or `balance_alpha`
2. **Document clearly**: This is a hyperparameter, not the class prior
3. **Best practice**: Set to 1.2× to 1.5× the estimated class prior
4. **Tune if possible**: Use validation set to optimize

**Why:**
- Maintains current excellent performance (+23% calibration)
- Honest about what the parameter represents
- Provides clear guidance for users
- Avoids theoretical confusion

### For Future Research: **Option 4 + Separate Calibration**

Implement theoretically correct E_X estimation + explicit calibration term:

```python
# Unbiased population expectation
E_X = estimate_population_expectation(phi_all, targets, π_true)

# Theoretical VPU term
vpu_loss = E_X - E_P[log φ(x)]

# Explicit calibration term
cal_loss = temperature_scaled_ECE(φ, π_true)

# Combined
L = vpu_loss + λ · cal_loss
```

This would provide:
- ✓ Theoretical soundness
- ✓ Explicit calibration
- ✓ Interpretable components
- ✓ Better hyperparameter tuning

---

## Conclusion

The current "prior" parameter in `vpu_nomixup_mean_prior` is **not the class prior** in the theoretical sense. It is an **empirical balance parameter** that:

1. **Adjusts loss term weights** (not a constraint)
2. **Provides implicit regularization** (toward higher recall)
3. **Works best when over-estimated** (especially with scarce labels)
4. **Dramatically improves calibration** (+23%) despite theoretical incorrectness

**The method works empirically, but for the wrong theoretical reasons.**

**Practical Recommendation:**
- Keep using the method (it works!)
- Rename `prior` → `positive_weight` (avoid confusion)
- Set to 1.2-1.5× estimated class prior (empirical guidance)
- Consider Option 4 for future theoretical soundness

**Key Insight:**
The success of `vpu_nomixup_mean_prior` demonstrates that **empirical regularization** can outperform strict theoretical correctness, especially for calibration. The "prior" parameter accidentally discovered an effective way to balance the VPU loss terms.
