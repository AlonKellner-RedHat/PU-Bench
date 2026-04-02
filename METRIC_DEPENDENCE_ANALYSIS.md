# Metric Dependence of Optimal Positive Weight

## Executive Summary

**Critical Finding:** The "optimal" positive weight parameter is **highly metric-dependent** and **cannot be reliably predicted** by simple formulas.

- **F1 (threshold-dependent):** Prefers α ≈ π · [1 + 0.43·(1-c)] — favors higher weights
- **AP (threshold-invariant):** Prefers α ≈ π · [1 + 0.08·(1-c)] — barely any boost
- **Agreement between metrics:** Only **24.1%** across F1, AP, AUC, Max F1

**Recommendation:** Use the true class prior (α = π) when available. Formula-based predictions are unreliable.

---

## Why F1 is Problematic for Evaluation

### Threshold Dependence

F1 score depends on the classification threshold:
```
F1 = 2·P·R / (P + R)
```

where P (precision) and R (recall) are computed at a **fixed threshold** (typically 0.5).

**Problem:** Different methods may favor different thresholds:
- Method A with α=0.7 might maximize F1 at threshold=0.3
- Method B with α=0.5 might maximize F1 at threshold=0.5

Comparing their F1 scores is **not a fair comparison** if they're evaluated at the same fixed threshold.

### Better Alternatives

**Threshold-invariant metrics:**

1. **Average Precision (AP)** ✓
   - Area under Precision-Recall curve
   - Considers all thresholds
   - Robust to class imbalance

2. **AUC-ROC** ✓
   - Area under ROC curve
   - Considers all thresholds
   - Standard in binary classification

3. **Max F1** ✓
   - Best F1 across all thresholds
   - Still F1-based but threshold-optimized

---

## Empirical Results

### Direction Preference by Metric

| Metric | Prefer Higher | Prefer Lower | Prefer Auto | Mean Δπ |
|--------|--------------|--------------|-------------|---------|
| **F1** | 63% | 24% | 13% | **+0.088** |
| **AP** | 35% | 43% | 22% | **-0.028** |
| **AUC** | 35% | 40% | 24% | **-0.014** |
| **Max F1** | 41% | 39% | 20% | **+0.013** |

**Observation:** F1 has strong bias toward higher weights; threshold-invariant metrics show no consistent bias.

### By Label Frequency (c)

**Using AP:**

| c | Prefer Higher | Prefer Lower | Prefer Auto | Mean Δπ (higher) | Mean Δπ (lower) |
|---|--------------|--------------|-------------|-----------------|-----------------|
| 0.1 | 61% | 33% | 6% | +0.38 | -0.14 |
| 0.5 | 22% | 56% | 22% | +0.28 | -0.38 |
| 0.9 | 22% | 39% | 39% | +0.16 | -0.34 |

**Using F1:**

| c | Prefer Higher | Prefer Lower | Prefer Auto | Mean Δπ (higher) | Mean Δπ (lower) |
|---|--------------|--------------|-------------|-----------------|-----------------|
| 0.1 | 83% | 6% | 11% | +0.27 | -0.02 |
| 0.5 | 50% | 22% | 28% | +0.22 | -0.33 |
| 0.9 | 56% | 44% | 0% | +0.09 | -0.19 |

**Key difference:**
- F1 strongly prefers higher at all c values
- AP prefers higher only at c=0.1, but prefers **lower** at c=0.5 and c=0.9

---

## Formula Performance

### F1-Based Formula

```
α = π · [1 + 0.426·(1-c)]
```

**Validation:**
- MAE: 0.150
- R²: -0.09
- Mean relative error: 25%

**Example (π=0.5, c=0.1):** α = 0.69 (+38% boost)

### AP-Based Formula

```
α = π · [1 + 0.078·(1-c)]
```

**Validation:**
- MAE: 0.310
- R²: -0.30
- Mean relative error: 55%

**Example (π=0.5, c=0.1):** α = 0.54 (+7% boost)

### Comparison

| Aspect | F1 Formula | AP Formula |
|--------|-----------|-----------|
| Boost strength | High (+38% at c=0.1) | Low (+7% at c=0.1) |
| MAE | 0.150 | 0.310 (worse!) |
| R² | -0.09 | -0.30 (much worse!) |
| Theoretical motivation | Weak | Weak |

**Both formulas have negative R²**, meaning they perform **worse than simply predicting the mean** for all cases!

---

## Metric Agreement Analysis

Out of 54 experiments (6 datasets × 3 seeds × 3 c values):

- **13 experiments (24.1%):** All 4 metrics agree
- **41 experiments (75.9%):** Metrics disagree

### Example Disagreements

| Dataset | Seed | c | True π | Optimal by F1 | Optimal by AP | Optimal by AUC | Optimal by Max F1 |
|---------|------|---|--------|--------------|--------------|---------------|------------------|
| 20News | 42 | 0.1 | 0.59 | 0.9 | 0.5 | 0.7 | 0.3 |
| 20News | 42 | 0.5 | 0.66 | 0.9 | 0.2 | 0.2 | 0.2 |
| 20News | 456 | 0.5 | 0.66 | 0.1 | 0.1 | 0.1 | **0.5** |
| 20News | 789 | 0.9 | 0.71 | 0.1 | **0.9** | 0.9 | 0.7 |

**These are the same (dataset, seed, c) configurations!** Yet different metrics choose radically different optimal priors.

---

## Why the High Variance?

### 1. Metric Objectives are Different

- **F1:** Harmonic mean of precision and recall at fixed threshold
- **AP:** Area under P-R curve (emphasizes ranking)
- **AUC:** Area under ROC curve (emphasizes discrimination)
- **Max F1:** Best precision-recall tradeoff (threshold-optimized F1)

Each metric has different preferences for model calibration and decision boundaries.

### 2. Small Performance Differences

Looking at actual performance values:

| Config | F1 @ π | F1 @ best | Δ F1 | AP @ π | AP @ best | Δ AP |
|--------|--------|-----------|------|--------|-----------|------|
| 20News, seed=42, c=0.1 | 0.867 | 0.877 | **+0.010** | 0.872 | 0.876 | **+0.004** |

The differences are often **< 1-2%**, which is:
- Within random seed variance
- Smaller than hyperparameter tuning noise
- Not practically significant

### 3. Dataset-Specific Effects

Some datasets show strong preferences:
- **Spambase, c=0.1:** All metrics prefer higher (0.7-0.9)
- **FashionMNIST, c=0.5:** Mixed preferences (0.5-0.9)
- **20News, c=0.5:** Strong disagreement (0.1-0.9)

These patterns don't follow simple rules based on c or π alone.

---

## Practical Implications

### What This Means for Users

1. **Don't over-optimize** the positive_weight parameter
   - Performance differences are small (1-2%)
   - Metric-dependent
   - Seed-dependent

2. **Use true prior when available**
   - Simple: α = (train_labels == 1).mean()
   - Works well across metrics
   - No hyperparameter tuning needed

3. **If you must tune, use validation set**
   - Grid search over α ∈ {0.5π, 0.75π, π, 1.25π, 1.5π}
   - Optimize for your target metric
   - Accept that optimal α is metric-specific

### What This Means for the "Prior" Parameter

The parameter is **not the class prior** in any meaningful sense:

1. **Theoretically:** Not derived from VPU formulation (see THEORETICAL_ANALYSIS.md)
2. **Empirically:** Optimal value is metric-dependent
3. **Practically:** Using true prior (α = π) is often competitive

**It's a hyperparameter** that balances loss terms, period.

---

## Recommendations

### For Documentation

**Rename the parameter:**
```python
# Before (misleading)
VPUNoMixUpMeanPriorLoss(prior=0.5)

# After (honest)
VPUNoMixUpMeanPriorLoss(positive_weight=0.5)
```

**Update docstring:**
```python
"""
Args:
    positive_weight: Weight on the positive term in the loss.
                    Recommended: Use the true class prior from labeled data.
                    Advanced: Can be tuned as hyperparameter on validation set.
                    Note: Optimal value is metric-dependent.
"""
```

### For Users

**Decision tree:**

```
1. Do you have labeled positives?
   YES → Use α = (train_labels == 1).mean()  ✓ SIMPLE & EFFECTIVE
   NO  → Go to 2

2. Do you have validation set?
   YES → Grid search α ∈ {0.5π, π, 1.5π} for your metric  ✓ ROBUST
   NO  → Go to 3

3. Do you have approximate prior estimate?
   YES → Use α = π_estimated  ⚠️ FALLBACK
   NO  → Use α = 0.5  ⚠️ GUESS
```

**DO NOT use formulas like α = π·[1 + κ·(1-c)]** — they are:
- Theoretically weak
- Empirically poor (R² < 0)
- Metric-dependent
- Not worth the complexity

### For Future Research

**Option A: Fix the theory**
- Implement theoretically correct E_X estimation (see THEORETICAL_ANALYSIS.md Option 4)
- Add explicit calibration term
- Eliminate the ad-hoc α multiplier

**Option B: Accept empirical reality**
- Treat α as hyperparameter
- Tune on validation set
- Document metric-dependence
- Provide reasonable defaults (α = π)

---

## Conclusion

The positive weight parameter in `vpu_nomixup_mean_prior` is:

✗ **Not** the theoretical class prior
✗ **Not** predictable by simple formulas
✗ **Not** consistent across metrics
✓ **Is** a metric-dependent hyperparameter
✓ **Is** often well-approximated by the true prior
✓ **Is** not worth over-optimizing (small gains, high variance)

**Best practice:** Use α = π (true prior from labeled data) and move on. The 1-2% potential gain from tuning is not worth the complexity, especially given metric-dependence.

---

## References

- **THEORETICAL_ANALYSIS.md:** Why the prior parameter is not theoretically sound
- **POSITIVE_WEIGHT_DERIVATION.md:** Attempted theoretical derivation (inconclusive)
- **PRIOR_PARAMETER_GUIDE.md:** Practical usage guide
- **results_robustness/optimal_prior_by_*.csv:** Optimal values for each metric
- **results_robustness/metric_comparison.csv:** Cross-metric comparison
