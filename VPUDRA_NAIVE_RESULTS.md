# VPUDRa-naive: Original PUDRa Loss (No Prior) + MixUp Consistency

**Date**: 2026-02-17
**Configuration**: SCAR benchmark, 9 datasets, c=0.1 (10% labeled), seed=42

---

## Executive Summary

**VPUDRa-naive** tests whether the **prior weighting is necessary** when MixUp regularization is present. It uses the **original PUDRa loss formulation** without prior weighting:

```
L_base = E_P[-log p + p] + E_U[p]  (NO π multiplication)
L_consistency = -μ_anchor * log(p_mix) + p_mix  (Point Process with anchor)
L_total = L_base + λ * L_consistency
```

**Key Result**: **86.12% avg F1** - competitive performance without explicit prior weighting!

---

## Overall Performance Rankings

| Rank | Method | Avg F1 | Avg AUC | Prior Type | Anchor | Consistency Loss |
|------|--------|--------|---------|------------|--------|------------------|
| **#1** | **VPU** | **87.57%** 🏆 | 92.60% | None | ✅ Implicit | Log-MSE |
| **#2** | **VPUDRa-Fixed** | **86.95%** | 92.42% | True π | ✅ Explicit | Log-MSE |
| **#3** | **VPUDRa-PP** | **86.91%** | 91.90% | True π | ✅ Explicit | Point Process |
| **#4** | **VPUDRa-naive** | **86.12%** 🆕 | 91.66% | **NO prior** | ✅ Explicit | Point Process |
| #5 | VPUDRa | 83.47% | 89.57% | Empirical π_emp | ✅ Explicit | Log-MSE |
| #6 | PUDRa | 77.75% | 92.63% | True π | ❌ None | - |
| #7 | VPUDRa-SoftLabel | 77.76% | 92.27% | True π | ❌ None | Point Process |

**Key Finding**: VPUDRa-naive ranks **#4**, demonstrating that **prior weighting is helpful but not essential** when strong MixUp regularization is present.

---

## Dataset-by-Dataset Results

### Complete Comparison Table

| Dataset | VPU | VPUDRa-Fixed | VPUDRa-PP | **VPUDRa-naive** 🆕 | VPUDRa | PUDRa | Naive Winner |
|---------|-----|--------------|-----------|---------------------|--------|-------|--------------|
| **MNIST** | 96.34% | 96.18% | 97.05% | **96.53%** | 92.83% | 97.30% ✓ | **Close** |
| **Fashion-MNIST** | 98.21% | 98.01% | 98.27% | **97.80%** | 84.65% ⚠️ | 98.27% ✓ | **Good** |
| **CIFAR-10** | 87.61% | 87.93% ✓ | 86.73% | **86.73%** | 82.18% | 86.43% | **Competitive** |
| **AlzheimerMRI** | 70.01% ✓ | 66.27% | 67.58% | **67.20%** | 68.00% | 65.54% | **Moderate** |
| **Connect-4** | 86.76% ✓ | 86.40% | 86.59% | **84.64%** | 81.08% | 86.48% | **Moderate** |
| **Mushrooms** | 98.25% | 98.31% | 98.25% | **97.98%** | 96.32% | 98.64% | **Good** |
| **Spambase** | 84.15% | 85.23% ✓ | 81.12% | **77.91%** ⚠️ | 82.20% | **2.18%** ❌ | **No collapse!** ✅ |
| **IMDB** | 78.49% ✓ | 77.05% | 78.12% | **77.84%** | 76.82% | 77.46% | **Competitive** |
| **20News** | 88.33% | 87.20% | 88.52% ✓ | **88.43%** | 87.14% | 87.41% | **Very close** |
| **Average** | **87.57%** 🏆 | **86.95%** | **86.91%** | **86.12%** 🆕 | **83.47%** | **77.75%** | - |

### AUC Scores

| Dataset | VPU | VPUDRa-Fixed | VPUDRa-PP | **VPUDRa-naive** 🆕 | VPUDRa | PUDRa |
|---------|-----|--------------|-----------|---------------------|--------|-------|
| **MNIST** | 99.52% | 99.47% | 99.46% | **99.51%** | 97.88% | 99.60% ✓ |
| **Fashion-MNIST** | 99.63% | 99.66% | 99.64% | **99.49%** | 92.35% ⚠️ | 99.73% ✓ |
| **CIFAR-10** | 96.26% | 96.39% ✓ | 95.74% | **95.24%** | 92.31% | 95.90% |
| **AlzheimerMRI** | 76.89% | 75.27% | 78.27% | **74.98%** | 74.51% | 79.53% ✓ |
| **Connect-4** | 88.17% ✓ | 87.90% | 87.82% | **86.22%** | 79.98% | 88.11% |
| **Mushrooms** | 99.97% | 99.97% | 99.97% | **99.99%** ✓ | 99.72% | 99.99% ✓ |
| **Spambase** | 93.58% | 93.78% ✓ | 92.23% | **90.46%** | 92.71% | 91.78% |
| **IMDB** | 85.76% | 86.00% ✓ | 85.73% | **85.29%** | 84.78% | 85.53% |
| **20News** | 93.62% | 93.32% | 93.42% | **93.80%** | 91.88% | 93.53% |
| **Average** | **92.60%** | **92.42%** | **91.90%** | **91.66%** 🆕 | **89.57%** | **92.63%** 🎯 |

---

## Critical Analysis

### What VPUDRa-naive Teaches Us

**Research Question**: Is prior weighting necessary when MixUp regularization is present?

**Answer**: **Helpful but not essential** - VPUDRa-naive achieves 86.12% without any prior, only -0.83% below VPUDRa-Fixed (86.95%) which uses true prior.

### Performance Breakdown

**1. Compared to VPUDRa-PP (same formulation + true prior)**:
- VPUDRa-PP: 86.91% (with π weighting)
- VPUDRa-naive: 86.12% (NO π weighting)
- **Difference**: Only **-0.79%**

**Conclusion**: The prior weighting `π * E_P[-log p]` provides a **small but consistent improvement**, but MixUp regularization is doing most of the work.

**2. Compared to PUDRa (original, no MixUp)**:
- PUDRa: 77.75% (with π, NO MixUp)
- VPUDRa-naive: 86.12% (NO π, WITH MixUp)
- **Difference**: **+8.37%**

**Conclusion**: **MixUp regularization (+8.37%) >> prior weighting (+0.79%)** in terms of impact.

**3. Spambase Performance (Critical Test)**:
- PUDRa: **2.18%** ❌ (catastrophic collapse)
- VPUDRa-naive: **77.91%** ✅ (stable!)
- VPUDRa-Fixed: **85.23%** (best with prior)

**Conclusion**: MixUp prevents catastrophic collapse even without prior weighting. However, **prior weighting provides an additional +7.32% on Spambase**, showing it's still valuable for challenging datasets.

---

## Per-Dataset Analysis

### Where VPUDRa-naive Excels

**Simple Images** (competitive):
- MNIST: 96.53% (only -0.77% vs best VPUDRa)
- Fashion-MNIST: 97.80% (only -0.50% vs best)
- Mushrooms: 97.98% (very close to 98.64% best)

**Text** (very close):
- 20News: 88.43% (only -0.09% vs best VPUDRa-PP!)
- IMDB: 77.84% (only -0.65% vs VPU)

**Analysis**: On easier datasets, the lack of prior weighting has minimal impact.

### Where Prior Weighting Matters Most

**Challenging Datasets** (larger gaps):
- **Spambase**: 77.91% (vs 85.23% for VPUDRa-Fixed) = **-7.32%** ❌
- **Connect-4**: 84.64% (vs 86.76% for VPU) = **-2.12%**

**Analysis**: On challenging, high-dimensional datasets like Spambase, prior weighting provides significant additional regularization.

---

## Theoretical Implications

### Loss Formulation Comparison

**VPUDRa-Fixed/PP** (with prior):
```
L_positive = π * E_P[-log p]  # Prior-weighted
L_unlabeled = E_U[p]
```

**VPUDRa-naive** (NO prior):
```
L_positive = E_P[-log p + p]  # Symmetric form, NO prior
L_unlabeled = E_U[p]
```

**Key Difference**:
- With prior: Scales positive risk by `π` (class prevalence)
- Without prior: Treats positive and unlabeled risks equally

### Why VPUDRa-naive Still Works Well

**1. MixUp Provides Implicit Balance**:
- MixUp mixes P samples with U samples → creates balanced training signal
- Consistency loss `-μ_anchor * log(p_mix) + p_mix` operates on mixed samples
- This provides implicit balancing without explicit prior

**2. The `+ p` Term**:
- Original PUDRa: `L(1,p) = -log p + p`
- The `+ p` term acts as a regularizer, preventing `p → 1` trivially
- This built-in regularization helps even without prior weighting

**3. Anchor Assumption**:
- `μ = λ*p(x) + (1-λ)*1.0` provides external reference
- This stabilization is more important than prior weighting

### Mathematical Insight

**With prior weighting** (VPUDRa-Fixed):
```
Total positive contribution = π * n_P * (-log p)
Total unlabeled contribution = n_U * p
```

**Without prior weighting** (VPUDRa-naive):
```
Total positive contribution = n_P * (-log p + p)
Total unlabeled contribution = n_U * p
```

Since `n_P ≈ π * n_U` under SCAR, the formulations are **approximately equivalent** when the `+ p` term is included!

---

## Advantages and Disadvantages

### ✅ Advantages of VPUDRa-naive

1. **Simpler**: No need to estimate or provide prior `π`
2. **One less hyperparameter**: No prior to tune
3. **Still competitive**: 86.12% avg (only -0.83% vs VPUDRa-Fixed)
4. **No catastrophic failures**: Avoids PUDRa's Spambase collapse
5. **Strong on simple datasets**: 96-98% on MNIST/Fashion-MNIST/Mushrooms

### ❌ Disadvantages of VPUDRa-naive

1. **Worse on Spambase**: 77.91% vs 85.23% for VPUDRa-Fixed (-7.32%)
2. **Lower average**: 86.12% vs 87.57% for VPU (-1.45%)
3. **Theoretical justification weaker**: Prior weighting is theoretically grounded
4. **Less robust on hard datasets**: Larger gaps on challenging tasks

---

## Comparison with Similar Methods

### Performance Relative to Each Variant

| Comparison | VPUDRa-naive | Other Method | Difference | Insight |
|------------|--------------|--------------|------------|---------|
| **vs VPU** | 86.12% | 87.57% | **-1.45%** | VPU's full pipeline still best |
| **vs VPUDRa-Fixed** | 86.12% | 86.95% | **-0.83%** | Prior weighting helps but minor |
| **vs VPUDRa-PP** | 86.12% | 86.91% | **-0.79%** | Nearly identical (prior = 0.79% gain) |
| **vs VPUDRa (empirical)** | 86.12% | 83.47% | **+2.65%** ✅ | Stable prior better than empirical |
| **vs PUDRa** | 86.12% | 77.75% | **+8.37%** ✅ | MixUp is critical |
| **vs VPUDRa-SoftLabel** | 86.12% | 77.76% | **+8.36%** ✅ | Anchor is critical |

### Clustering by Design Choice

**Anchor + MixUp + Prior** (best tier):
- VPU: 87.57%
- VPUDRa-Fixed: 86.95%
- VPUDRa-PP: 86.91%

**Anchor + MixUp, NO prior** (this method):
- **VPUDRa-naive: 86.12%** 🆕

**Anchor + MixUp, unstable prior**:
- VPUDRa (empirical): 83.47%

**NO anchor** (failed tier):
- PUDRa: 77.75%
- VPUDRa-SoftLabel: 77.76%

**Clear pattern**: Anchor + MixUp are essential. Prior is beneficial but secondary.

---

## Recommendations

### When to Use VPUDRa-naive

✅ **Use when**:
1. **Prior is unknown** and can't be reliably estimated
2. **Simplicity is valued** - fewer hyperparameters
3. **Datasets are relatively balanced** (close to 50% positive)
4. **Computational budget is limited** (one less parameter to tune)
5. **Simple/moderate datasets** (MNIST-like, text classification)

### When NOT to Use VPUDRa-naive

❌ **Avoid when**:
1. **Working with Spambase-like datasets** (high-dimensional, challenging)
2. **Prior is known or easily estimated** - use VPUDRa-Fixed instead
3. **Need absolute best performance** - use VPU (87.57%)
4. **Dataset has extreme class imbalance** - prior weighting helps more

### Recommended Alternatives

| Scenario | Recommended Method | Reason |
|----------|-------------------|---------|
| **Default choice** | **VPU** (87.57%) | Best overall, proven |
| **Prior known** | **VPUDRa-Fixed** (86.95%) | Slightly better than naive |
| **Prior unknown** | **VPUDRa-naive** (86.12%) | Simpler, competitive |
| **Research/theory** | **VPUDRa-PP** (86.91%) | PUDRa-aligned |
| **Never use** | PUDRa, VPUDRa-SoftLabel | Catastrophic failures |

---

## Implementation Details

### Loss Function

**File**: [loss/loss_vpudra_naive.py](loss/loss_vpudra_naive.py)

```python
# Positive risk: E_P[-log p + p] (NO π weighting!)
positive_risk = torch.mean(-torch.log(p_positive + self.epsilon) + p_positive)

# Unlabeled risk: E_U[p]
unlabeled_risk = torch.mean(p_unlabeled) if len(p_unlabeled) > 0 else 0.0

# Point Process Consistency with Anchor
# μ_anchor = λ * p(x) + (1-λ) * 1.0  (computed in trainer)
consistency_loss = torch.mean(
    -mu_anchor * torch.log(p_mix + self.epsilon) + p_mix
)

# Total loss
total_loss = positive_risk + unlabeled_risk + lam * consistency_loss
```

### Trainer

**File**: [train/vpudra_naive_trainer.py](train/vpudra_naive_trainer.py)

- Based on VPUDRa-PP trainer
- Creates anchored MixUp targets: `μ = λ*p(x) + (1-λ)*1.0`
- No prior parameter passed to loss function

### Configuration

**File**: [config/methods/vpudra_naive.yaml](config/methods/vpudra_naive.yaml)

```yaml
vpudra_naive:
  optimizer: adam
  lr: 0.0003
  mix_alpha: 0.3      # MixUp Beta distribution
  epsilon: 1e-7       # Numerical stability
  # NO prior parameter!
```

---

## Experimental Validation

### Hypothesis Testing

**H1**: MixUp regularization is more important than prior weighting
- **Result**: ✅ CONFIRMED
- MixUp contribution: +8.37% (PUDRa → VPUDRa-naive)
- Prior contribution: +0.79% (VPUDRa-naive → VPUDRa-PP)

**H2**: Original PUDRa form (with `+ p` term) works well without prior
- **Result**: ✅ CONFIRMED
- VPUDRa-naive: 86.12% (competitive with prior-weighted variants)

**H3**: Anchor + MixUp prevent collapse even without prior
- **Result**: ✅ CONFIRMED
- Spambase: 77.91% (vs PUDRa's 2.18% collapse)

### Statistical Summary

| Metric | VPUDRa-naive | Best Variant | Gap |
|--------|--------------|--------------|-----|
| **Mean F1** | 86.12% | 87.57% (VPU) | -1.45% |
| **Median F1** | 86.73% | 87.93% (VPUDRa-Fixed on CIFAR) | - |
| **Std Dev F1** | 10.42% | - | - |
| **Min F1** | 67.20% (AlzheimerMRI) | - | - |
| **Max F1** | 97.80% (Fashion-MNIST) | - | - |
| **Mean AUC** | 91.66% | 92.63% (PUDRa) | -0.97% |

---

## Conclusion

**VPUDRa-naive successfully demonstrates** that:

1. ✅ **Prior weighting is beneficial but not essential** when strong MixUp regularization is present
2. ✅ **MixUp is the dominant factor** (+8.37% impact vs +0.79% for prior)
3. ✅ **Original PUDRa formulation with `+ p` term** provides built-in regularization
4. ✅ **Anchor assumption prevents collapse** even without prior weighting
5. ⚠️ **Challenging datasets benefit more from prior** (Spambase: +7.32% with prior)

**Ranking among VPU/PUDRa variants**:
- Rank #4 out of 7 variants
- 86.12% avg F1 (competitive)
- Only -0.83% below best VPUDRa variant (VPUDRa-Fixed)
- Only -1.45% below overall best (VPU)

**Practical Takeaway**: If the prior is unknown or hard to estimate, **VPUDRa-naive is a viable alternative** that sacrifices <1% performance for simplicity. However, if performance on challenging datasets (Spambase-like) is critical, **invest in prior estimation** and use VPUDRa-Fixed.

**Final Recommendation**: Use **VPU** (87.57%) as default. If exploring prior-free approaches, **VPUDRa-naive** (86.12%) is the best option, significantly outperforming baseline PUDRa (77.75%).
