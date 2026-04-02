# Revised Default Prior Recommendation

## Critical Limitation Discovered

**The robustness experiments only test datasets with true priors in the range [0.42, 0.71].**

All analyzed datasets are **moderately balanced**:
- 20News: π ∈ [0.59, 0.71]
- FashionMNIST: π ∈ [0.52, 0.66]
- IMDB: π ∈ [0.52, 0.66]
- MNIST: π ∈ [0.52, 0.65]
- Mushrooms: π ∈ [0.51, 0.64]
- Spambase: π ∈ [0.42, 0.55]

**Mean true prior: 0.581 ± 0.075**

### What This Means

✅ **Recommendations ARE valid for:** Moderately balanced datasets (π ≈ 0.4-0.7)

❌ **Recommendations MAY NOT generalize to:**
- Highly imbalanced datasets with π < 0.4 (rare positive class)
- Highly imbalanced datasets with π > 0.7 (rare negative class)

---

## Key Finding: Method Prior Choice Is Less Critical Than Expected

**Surprising result:** No significant difference (p=0.73) between:
- **In-range method priors** (0.5-0.7): AP = 0.913
- **Out-of-range method priors** (0.1-0.3, 0.9-1.0): AP = 0.908

The performance difference is only **0.5%**, which is:
- Smaller than seed-to-seed variance
- Not statistically significant
- Likely within measurement noise

**Exception:** `prior=1.0` is consistently bad (AP=0.897, -1.5% vs in-range)

---

## Revised Recommendations

### For Moderately Balanced Datasets (π ≈ 0.4-0.7)

Based on analysis of 756 experiments across 6 datasets:

#### 1️⃣ **If you have labeled positives** (BEST)
```python
positive_weight = (train_labels == 1).float().mean()
```

**Why:**
- Uses actual class prior from data
- Best calibration (ECE = 0.061)
- Good performance (AP = 0.912)
- Moderate convergence (11.1 epochs)

#### 2️⃣ **If NO prior information available** (SAFE DEFAULT)

**Performance priority:**
```python
positive_weight = 0.7  # or 0.9
# AP: 0.914 (best)
# Convergence: 10.9-12.3 epochs
# Calibration: ECE = 0.070-0.085
```

**Balanced (all metrics):**
```python
positive_weight = 0.5
# AP: 0.913 (only -0.1% vs best)
# Convergence: 10.3 epochs (30% faster than prior=1.0)
# Calibration: ECE = 0.075
```

**Fast convergence priority:**
```python
positive_weight = 0.1  # or 0.5
# AP: 0.889 (-2.5% vs best)
# Convergence: 10.3 epochs (fastest)
# Calibration: ECE = 0.075
```

#### 3️⃣ **AVOID:**
```python
positive_weight = 1.0  # Equivalent to vpu_nomixup_mean
# AP: 0.897 (-1.7% vs best)
# Convergence: 14.8 epochs (slowest)
# Calibration: ECE = 0.110 (worst)
```

---

## Performance Summary (for π ≈ 0.4-0.7)

| Prior | AP | Epochs | ECE | Recommendation |
|-------|-----|--------|-----|----------------|
| **0.5** | 0.913 | 10.3 | 0.075 | ⭐ **Balanced default** |
| **0.7** | 0.914 | 10.9 | 0.070 | ✓ Best performance |
| **0.9** | 0.914 | 12.3 | 0.085 | ✓ Best performance |
| **auto** | 0.912 | 11.1 | **0.061** | ✓ **Best calibration** |
| 0.1 | 0.889 | **10.3** | 0.075 | ⚠️ Fastest convergence |
| 0.2 | 0.903 | 10.3 | 0.072 | ⚠️ Acceptable |
| 0.3 | 0.901 | 10.4 | 0.069 | ⚠️ Acceptable |
| **1.0** | **0.897** | **14.8** | **0.110** | ❌ **Avoid** |

---

## For Highly Imbalanced Datasets (π < 0.4 or π > 0.7)

### ⚠️ **Uncertain Territory**

The experiments **did not test** datasets with:
- **Rare positive class:** π < 0.4 (e.g., fraud detection, rare disease)
- **Rare negative class:** π > 0.7 (e.g., quality control with high pass rate)

### Hypothetical Recommendations (Extrapolation)

**For π < 0.4 (rare positives):**
- Likely need **higher positive_weight** to compensate for scarcity
- Consider: `positive_weight = π * 1.5` or tune on validation set
- Risk: Over-weighting may hurt calibration

**For π > 0.7 (rare negatives):**
- Likely need **lower positive_weight** to avoid over-penalizing positives
- Consider: `positive_weight = π * 0.8` or tune on validation set
- Risk: Under-weighting may hurt recall on positives

**Best practice:**
- If possible, obtain even a small labeled positive set to compute true prior
- Otherwise, **tune on validation set** rather than relying on defaults

---

## Multi-Objective Trade-offs

### Performance vs Convergence Speed

| Objective | Default Prior | AP | Epochs | Trade-off |
|-----------|--------------|-----|--------|-----------|
| **Max Performance** | 0.7-0.9 | **0.914** | 10.9-12.3 | +0.1% AP, +20% time |
| **Balanced** ⭐ | **0.5** | 0.913 | **10.3** | Best all-around |
| **Fast Convergence** | 0.1-0.5 | 0.889-0.913 | **10.3** | Save 30% time |
| **Best Calibration** | auto | 0.912 | 11.1 | Requires labeled data |

### When to Optimize What

**Optimize for performance (0.7-0.9):**
- Production models where accuracy matters
- Sufficient compute budget
- Don't need rapid iteration

**Optimize for speed (0.1-0.5):**
- Hyperparameter search
- Prototyping
- Limited compute budget
- Many experiments to run

**Optimize for calibration (auto):**
- Medical diagnosis
- Financial risk assessment
- Any application requiring probability estimates

---

## Theoretical vs Empirical Reality

### What Theory Says
- Prior parameter should equal true class prior π
- Based on PU learning variational bound
- Should ensure `E_X[φ(x)] ≈ π`

### What Experiments Show
- **For moderately balanced datasets (π ≈ 0.4-0.7):**
  - Choice of method_prior is **not very sensitive** (in-range vs out-of-range differ by only 0.5%)
  - Using true prior (auto) works well but is not dramatically better
  - Values in range [0.5-0.9] all perform similarly
  - **Exception:** 1.0 is consistently poor

- **For highly imbalanced datasets (π < 0.4 or > 0.7):**
  - **Unknown** — not tested in experiments
  - Theoretical guidance less reliable
  - Empirical tuning recommended

### Why the Discrepancy?

The "prior" parameter is **not theoretically sound** (see THEORETICAL_ANALYSIS.md):
1. Uses `E_all` (batch samples) not `E_X` (population)
2. The π multiplier has no theoretical justification
3. Functions as empirical regularization hyperparameter

This explains why:
- Empirical optimal ≠ theoretical class prior
- Performance is not very sensitive to exact value
- Extreme values (1.0) hurt but moderate range [0.5-0.9] all work

---

## Decision Tree

```
┌─ Do you have labeled positive samples?
│
├─ YES ────────────────────────────────────────────────┐
│   Use auto (true prior)                              │
│   ✓ Best calibration                                 │
│   ✓ Theoretically motivated                          │
│   ✓ No guessing required                             │
│                                                       │
└─ NO ─────────────────────────────────────────────────┤
    │                                                   │
    ├─ Is your dataset moderately balanced (π≈0.4-0.7)?│
    │                                                   │
    ├─ YES ───────────────────────────────────────────┐│
    │   Choose based on priority:                     ││
    │   • Performance → use 0.7 or 0.9                ││
    │   • Balanced → use 0.5 ⭐                        ││
    │   • Speed → use 0.1 or 0.5                      ││
    │   (All similar, ±0.5% performance)              ││
    │                                                  ││
    └─ NO / UNKNOWN ─────────────────────────────────┐││
        Highly imbalanced dataset                    │││
        ⚠️ Tested recommendations may not apply      │││
        Options:                                      │││
        1. Get labeled data → use auto               │││
        2. Tune on validation set                    │││
        3. Extrapolate (risky):                      │││
           - π < 0.4 → try 1.5×π                     │││
           - π > 0.7 → try 0.8×π                     │││
                                                      │││
        ✓ SAFE: Use 0.5 and validate results         │││
        ✓ BEST: Run experiments to find optimal      │││
```

---

## Limitations of Current Analysis

1. **Limited true prior range:** Only tested π ∈ [0.42, 0.71]
   - Mean: 0.581, Std: 0.075
   - All datasets moderately balanced
   - Cannot generalize to highly imbalanced cases

2. **Dataset diversity:** Only 6 datasets tested
   - 2 vision (MNIST, FashionMNIST)
   - 2 text (IMDB, 20News)
   - 2 tabular (Mushrooms, Spambase)
   - May not cover all application domains

3. **Metric dependence:** Optimal prior varies by metric
   - F1 prefers higher values (0.9)
   - AP prefers moderate values (0.5-0.9)
   - Calibration prefers true prior (auto)

4. **Small performance differences:** Most recommendations differ by <1%
   - Within seed-to-seed variance
   - Practical significance uncertain

---

## Future Work Needed

To strengthen recommendations:

1. **Test highly imbalanced datasets:**
   - π < 0.3 (rare positive class)
   - π > 0.8 (rare negative class)
   - Fraud detection, anomaly detection use cases

2. **Larger-scale experiments:**
   - More datasets per prior range
   - More seeds for statistical power
   - Finer prior value grid

3. **Theoretical investigation:**
   - Why is method_prior not very sensitive?
   - Can we derive better default from theory?
   - Fix theoretical issues (E_all vs E_X)

---

## Bottom Line

**For most practical scenarios (moderately balanced datasets):**

1. **Best practice:** Use auto (true prior) if you have any labeled positives
2. **Safe default:** Use 0.5 if you have no prior information
3. **Avoid:** Never use 1.0 (equivalent to vpu_nomixup_mean)

**For highly imbalanced datasets:**
- Current recommendations are **uncertain**
- Best to obtain labeled data or tune empirically
- Consider theoretical alternatives (see THEORETICAL_ANALYSIS.md)

**Overall:** Choice of method_prior is **less critical than expected** for moderately balanced datasets. The improvement from optimal tuning is small (<1%), so spending too much effort on this parameter may not be worthwhile.
