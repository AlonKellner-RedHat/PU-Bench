# Prior Parameter Guide: Theory and Practice

## TL;DR

**What we found:**
1. The "prior" parameter in `vpu_nomixup_mean_prior` is **NOT the theoretical class prior**
2. It's an empirical hyperparameter that balances loss terms
3. We derived a theoretically-motivated formula: `α = π · [1 + 0.4·(1-c)]`
4. But **empirical validation is inconclusive** due to dataset-specific effects and noise

**What you should do:**
- ✅ **Best:** Use true prior computed from labeled data (`α = π`)
- ⚠️ **Fallback:** Use formula when prior unknown and must be estimated
- 🔬 **Advanced:** Tune on validation set for your specific dataset

---

## The Problem

The current `vpu_nomixup_mean_prior` method uses:

```python
loss = E_all[φ(x)] - α · E_P[log φ(x)]
```

The parameter α is called "prior" but it's **not the class prior** π in the theoretical sense. See [THEORETICAL_ANALYSIS.md](THEORETICAL_ANALYSIS.md) for details.

**Key issues:**
1. E_all (batch samples) ≠ E_X (population) due to sampling bias
2. The α multiplier has no theoretical justification in VPU formulation
3. Empirically, α > π often works better (especially with scarce labels)

---

## Theoretical Derivation

### Lagrangian Regularization Perspective

If we view VPU learning as:
```
minimize: -E_P[log φ(x)]  (maximize positive likelihood)
subject to: E_X[φ(x)] ≈ π  (calibration constraint)
```

The Lagrangian gives:
```
L = λ·E_X[φ] - E_P[log φ]
```

Normalizing by λ:
```
L = E_X[φ] - (1/λ)·E_P[log φ]
  = E_X[φ] - α·E_P[log φ]
```

where **α = 1/λ** is the dual weight on the positive term.

### Label Scarcity Adjustment

When labeled positives are scarce (small c), the positive likelihood term has:
- Higher variance (fewer samples)
- Weaker learning signal
- Need for stronger weighting to compensate

This motivates:
```
α = π · [1 + κ·(1-c)]
```

where:
- π = base class prior
- c = label frequency (fraction of positives that are labeled)
- κ = scarcity compensation factor

**Interpretation:**
- c → 1 (many labels): α → π (standard formulation)
- c → 0 (few labels): α → π·(1+κ) (boost to compensate)

---

## Empirical Validation

We fitted this formula to 47 experiments from robustness analysis:

**Results:**
- Fitted κ = 0.426 ± 0.103
- MAE = 0.15 (mean absolute error)
- R² = -0.09 (negative = poor fit)

**Why the poor fit?**

1. **High seed variance:** Same (dataset, c, π) combination has different optimal α across seeds
2. **Coarse grid:** Only 6 discrete values tested [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
3. **Dataset effects:** Optimal weight depends on data characteristics beyond c and π
4. **Small differences:** F1 improvements often ~1-2%, within noise

**Example: 20News with c=0.5, π=0.66**
- Seed 42: best α = 0.9 (+36% over π)
- Seed 456: best α = 0.1 (-85% under π)
- Seed 789: best α = 0.1 (-85% under π)

This high variance means **there's no single "correct" formula** that works universally.

---

## Practical Recommendations

### Scenario 1: You Have Labeled Positives (Best Case)

```python
# Compute true prior from labeled data
true_prior = (train_labels == 1).float().mean()

# Use it directly (simplest and often best)
positive_weight = true_prior

# Pass to loss
criterion = VPUNoMixUpMeanPriorLoss(positive_weight)
```

**Why:** This is what the current implementation does, and it works well (+22% calibration improvement).

### Scenario 2: You Only Have Coarse Prior Estimate

```python
def compute_positive_weight(prior: float, label_frequency: float,
                           scarcity_factor: float = 0.4) -> float:
    """
    Formula: α = π · [1 + κ·(1-c)]

    Args:
        prior: Estimated class prior (e.g., from domain knowledge)
        label_frequency: Fraction of positives labeled (c)
        scarcity_factor: Boost parameter (default 0.4)

    Returns:
        Positive weight for loss function
    """
    return prior * (1 + scarcity_factor * (1 - label_frequency))

# Example: You know prior is "between 0.5 and 0.75"
estimated_prior = 0.625  # Use bin center
c = 0.1  # You have 10% of positives labeled

positive_weight = compute_positive_weight(estimated_prior, c)
# → 0.85 (36% boost over estimate)
```

**When to use:**
- True prior is unknown
- You have domain knowledge for rough estimate (±0.25 resolution)
- Scarce labeled data (c < 0.5)

**When NOT to use:**
- You have labeled positives (use Scenario 1 instead)
- You have validation set (use Scenario 3 instead)

### Scenario 3: You Have Validation Set (Most Robust)

```python
def tune_scarcity_factor(train_data, val_data, prior_estimate, c):
    """Tune κ on validation set"""
    best_kappa = None
    best_val_f1 = 0

    for kappa in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        alpha = prior_estimate * (1 + kappa * (1 - c))

        # Train with this alpha
        model = train_vpu(train_data, alpha)
        val_f1 = evaluate(model, val_data)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_kappa = kappa

    return best_kappa

# Use it
best_kappa = tune_scarcity_factor(train, val, prior_estimate=0.6, c=0.1)
positive_weight = compute_positive_weight(0.6, 0.1, best_kappa)
```

**Why:** Dataset-specific tuning accounts for characteristics beyond c and π.

---

## Decision Tree

```
Do you have labeled positives in your training set?
│
├─ YES → Use α = (train_labels == 1).mean()  ✅ BEST
│
└─ NO → Do you have a validation set?
    │
    ├─ YES → Tune κ on validation set  ✅ ROBUST
    │
    └─ NO → Do you have coarse prior estimate?
        │
        ├─ YES → Use α = π·[1 + 0.4·(1-c)]  ⚠️ FALLBACK
        │
        └─ NO → Use α = 0.5  ⚠️ GUESS
```

---

## Example Values

Using formula with κ = 0.4:

| Prior (π) | Label Freq (c) | Positive Weight (α) | Boost |
|-----------|---------------|-------------------|-------|
| 0.3 | 0.1 | 0.41 | +38% |
| 0.3 | 0.5 | 0.36 | +21% |
| 0.3 | 0.9 | 0.31 | +4% |
| **0.5** | **0.1** | **0.69** | **+38%** |
| **0.5** | **0.5** | **0.61** | **+21%** |
| **0.5** | **0.9** | **0.52** | **+4%** |
| 0.7 | 0.1 | 0.97 | +38% |
| 0.7 | 0.5 | 0.85 | +21% |
| 0.7 | 0.9 | 0.73 | +4% |

**Pattern:** Boost is larger when:
- Label frequency c is smaller (scarce labels)
- Boost is independent of π (scales proportionally)

---

## Key Takeaways

### ✅ What Works

1. **Using true prior from labeled data** (current implementation)
   - +22% calibration improvement
   - +0.9% F1 improvement
   - Simple and reliable

2. **Boosting for scarce labels** (theoretical formula)
   - Theoretically motivated
   - Provides principled starting point
   - Reasonable default when prior unknown

### ❌ What Doesn't Work

1. **Expecting a universal formula**
   - High variance across seeds and datasets
   - Dataset-specific effects dominate

2. **Over-interpreting the "prior" parameter**
   - It's NOT the theoretical class prior
   - It's an empirical balance parameter
   - Optimal value depends on many factors

### 🎯 Recommended Actions

1. **For existing code:** No changes needed, current implementation is good
2. **For documentation:** Rename "prior" → "positive_weight" to avoid confusion
3. **For new users:** Recommend using true prior when available
4. **For researchers:** Consider theoretically correct alternatives (see THEORETICAL_ANALYSIS.md Option 4)

---

## References

- **THEORETICAL_ANALYSIS.md**: Detailed analysis of why current implementation is not theoretically sound
- **POSITIVE_WEIGHT_DERIVATION.md**: Full mathematical derivation of the formula
- **results_robustness/**: Empirical robustness experiments (378 runs)
- **results_robustness/optimal_prior_analysis.csv**: Per-experiment optimal values
- **results_robustness/formula_validation/**: Validation plots and results
