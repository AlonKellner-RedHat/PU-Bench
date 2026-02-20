# PUDRa-Unified: A Negative Result

## Summary

**PUDRa-Unified is a FAILED variant** that demonstrates why separate averaging over positive and unlabeled samples is essential for PU learning.

**Result:** Catastrophic miscalibration (A-NICE = 2.873, **251% worse** than PUDRa-naive)

---

## Hypothesis

**Question:** Would averaging the loss elementwise over ALL samples (rather than separately over positive and unlabeled) provide better gradient stability?

**Motivation:**
- Unified averaging: `E_all[loss(x, t)]` - single expectation
- Might provide more stable gradients
- Natural weighting by sample count
- Potentially better for varying positive/unlabeled ratios

---

## Implementation

### PUDRa-naive (Correct Approach)

```python
# Separate averaging - balanced contributions
positive_risk = mean(-log p + p for p in positives)
unlabeled_risk = mean(p for p in unlabeled)
total_loss = positive_risk + unlabeled_risk
```

**Key property:** Both terms contribute equally regardless of group sizes.

### PUDRa-unified (Failed Approach)

```python
# Unified averaging - imbalanced by sample count
elementwise_loss = []
for x, t in batch:
    if t == 1:  # positive
        elementwise_loss.append(-log p + p)
    else:  # unlabeled
        elementwise_loss.append(p)

total_loss = mean(elementwise_loss)  # Average over ALL samples
```

**Key problem:** Larger groups dominate by sheer numbers.

---

## Results

### Performance Comparison

| Metric | PUDRa-naive | PUDRa-unified | Δ (Unified - Naive) |
|--------|-------------|---------------|---------------------|
| **Avg F1** | **86.47%** | 85.16% | **-1.3%** 🔴 |
| **Avg AUC** | **92.08%** | 90.54% | **-1.5%** 🔴 |
| **Avg A-NICE** | **0.819** | 2.873 | **+2.054 (+251%)** 🔴 |

### Calibration Analysis

**PUDRa-unified has the WORST calibration of any method tested:**

| Method | Avg A-NICE | Calibration Quality |
|--------|------------|---------------------|
| VPU | 0.465 | Excellent |
| PUDRa-prior | 0.574 | Good |
| PUDRa-naive | 0.819 | Moderate |
| **Baseline (random)** | **1.000** | **Random** |
| nnPU | 1.055 | Poor (worse than random) |
| **PUDRa-unified** | **2.873** | **Catastrophic (187% worse than random!)** 🔴 |
| PN-Naive | 3.039 | Catastrophic |

**PUDRa-unified is the 2nd worst calibrated method**, only better than PN-Naive.

---

## Dataset-by-Dataset Results

**Every single dataset shows worse calibration with unified averaging:**

| Dataset | PUDRa-naive A-NICE | PUDRa-unified A-NICE | Δ A-NICE |
|---------|-------------------|---------------------|----------|
| MNIST | 0.576 | 3.181 | +2.605 🔴 |
| FashionMNIST | 0.639 | 3.410 | +2.771 🔴 |
| CIFAR10 | 0.367 | 2.989 | +2.622 🔴 |
| AlzheimerMRI | 0.742 | 1.412 | +0.671 🔴 |
| Connect4 | 0.697 | 2.788 | +2.091 🔴 |
| Mushrooms | 1.540 | 3.379 | +1.839 🔴 |
| Spambase | 1.278 | 2.879 | +1.601 🔴 |
| IMDB | 0.706 | 2.917 | +2.212 🔴 |
| 20News | 0.826 | 2.897 | +2.071 🔴 |

**No exceptions** - unified averaging is universally worse.

---

## Why Unified Averaging Failed

### The Problem: Imbalanced Gradient Contributions

**Example batch with 100 positives + 900 unlabeled:**

**PUDRa-naive (correct):**
```
positive_risk = mean(100 positive losses) → 1 term in final loss
unlabeled_risk = mean(900 unlabeled losses) → 1 term in final loss
total = positive_risk + unlabeled_risk → Equal contribution (50%/50%)
```

**PUDRa-unified (broken):**
```
total_loss = mean(all 1000 losses)
  = (sum of 100 positive losses + sum of 900 unlabeled losses) / 1000
  → Positive contribution: 10%
  → Unlabeled contribution: 90%
```

### Gradient Analysis

**PUDRa-unified gradient:**
```python
∂L/∂θ = (1/N) * Σ ∂loss_i/∂θ

For batch of 100 pos + 900 unlab:
  = (1/1000) * [100 * ∂(positive_loss)/∂θ + 900 * ∂(unlabeled_loss)/∂θ]
  = 0.1 * ∂(positive_loss)/∂θ + 0.9 * ∂(unlabeled_loss)/∂θ
```

**Result:** Model learns to minimize unlabeled risk (90% weight) and largely ignores positive risk (10% weight).

**PUDRa-naive gradient:**
```python
∂L/∂θ = ∂(positive_risk)/∂θ + ∂(unlabeled_risk)/∂θ
  = 0.5 * ∂(positive_loss)/∂θ + 0.5 * ∂(unlabeled_loss)/∂θ  (effective)
```

**Result:** Both risks contribute equally, balanced training.

---

## Key Insights

### 1. Separate Averaging is ESSENTIAL for PU Learning

**PUDRa-naive's design is not arbitrary** - it's mathematically necessary:
- Ensures balanced gradient contributions
- Prevents majority class (unlabeled) from dominating
- Maintains calibration (A-NICE = 0.819 vs 2.873)

### 2. Sample Count Weighting is HARMFUL

**Natural weighting by sample count sounds intuitive but fails:**
- In PU learning, positive and unlabeled are NOT equally important
- Both need equal representation in the loss
- Sample count imbalance must be corrected, not reflected

### 3. This Validates Standard PU Learning Practice

**Most PU methods use separate averaging:**
- nnPU: `π * E_P[loss_P] - E_U[loss_U]`
- uPU: `π * E_P[loss_P] + E_U[loss_U]`
- PUDRa: `π * E_P[-log p] + E_U[p]`

**All use separate expectations to ensure balanced contributions.**

### 4. Batch Composition Matters

**PUDRa-unified's behavior depends on batch composition:**
- Batch with 50% positive: unified averaging ≈ separate averaging
- Batch with 10% positive: unlabeled dominates (90% weight)
- **Real PU scenarios typically have <10% positive** (c=0.1 standard)

**This explains the catastrophic calibration** - most batches are heavily imbalanced.

---

## Theoretical Analysis

### Loss Function Properties

**PUDRa-naive:**
```
L = E_P[-log p + p] + E_U[p]
  = (1/|P|) Σ_P[-log p + p] + (1/|U|) Σ_U[p]
```

**Gradient:**
```
∂L/∂θ = (1/|P|) Σ_P ∂(-log p + p)/∂θ + (1/|U|) Σ_U ∂p/∂θ
```

**PUDRa-unified:**
```
L = E_all[loss(x,t)]
  = (1/|P|+|U|) [Σ_P[-log p + p] + Σ_U[p]]
```

**Gradient:**
```
∂L/∂θ = (|P|/(|P|+|U|)) * ∂(positive_loss)/∂θ
      + (|U|/(|P|+|U|)) * ∂(unlabeled_loss)/∂θ
```

**The implicit weighting by group size destroys balance.**

---

## Practical Implications

### For Method Design

**✅ DO:**
- Use separate averaging over positive and unlabeled
- Ensure balanced gradient contributions
- Weight by group type, not group size

**❌ DON'T:**
- Average elementwise over all samples
- Rely on "natural" sample count weighting
- Assume larger groups should contribute more

### For PU Learning Research

**This result demonstrates:**
1. Not all "natural" formulations work in PU learning
2. Class imbalance correction is critical
3. Negative results are valuable for understanding design choices

### For Practitioners

**If implementing PU methods:**
- Always use separate expectations for positive and unlabeled
- Don't be tempted by "simpler" unified formulations
- Test calibration, not just AUC/F1 (AUC hides miscalibration)

---

## Comparison to Other Methods

### PUDRa Variant Rankings

| Rank | Variant | Avg F1 | Avg A-NICE | Notes |
|------|---------|--------|------------|-------|
| #1 | **VPUDRa-Fixed** | 87.0% | 0.498 | Prior + MixUp |
| #2 | **VPUDRa-naive-logmse** | 86.7% | 0.689 | No prior + MixUp |
| #3 | **PUDRa-naive** | 86.5% | 0.819 | No prior, no MixUp |
| #4 | PUDRa-prior | 77.7% | 0.574 | Prior, no MixUp (catastrophic on tabular) |
| #5 | **PUDRa-unified** | 85.2% | **2.873** | **FAILED variant** 🔴 |

**PUDRa-unified ranks last among PUDRa variants** in calibration.

---

## Conclusion

### Summary

**PUDRa-Unified is a FAILED method** that should NEVER be used:
- **251% worse calibration** than PUDRa-naive
- **187% worse than random baseline**
- **Universally worse** across all 9 datasets
- **No compensating advantages** (F1 also worse by 1.3%)

### The Lesson

**Separate averaging is not optional in PU learning** - it's essential for:
1. Balanced gradient contributions
2. Preventing majority class dominance
3. Maintaining reasonable calibration

**This negative result validates:**
- PUDRa-naive's design choices
- Standard PU learning practice
- The importance of balanced loss formulations

### Recommendation

**Never use unified averaging for PU learning.**

If you need a simple PU method without MixUp:
- ✅ Use **PUDRa-naive** (separate averaging)
- ❌ Avoid **PUDRa-unified** (catastrophic calibration)

---

**Generated:** February 2026
**Method:** PUDRa-Unified (elementwise averaging)
**Datasets:** 9 (MNIST, FashionMNIST, CIFAR10, AlzheimerMRI, Connect4, Mushrooms, Spambase, IMDB, 20News)
**Seed:** 42
**Status:** ❌ NEGATIVE RESULT - Do Not Use
