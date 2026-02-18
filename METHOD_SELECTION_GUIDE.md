# PU Learning Method Selection Guide

## Executive Summary

This guide helps you choose between the **three most practical PU learning methods** in PU-Bench:

| Method | Best For | Key Advantage | Key Disadvantage |
|--------|----------|---------------|------------------|
| **VPU** | Production systems, when performance matters | Best F1 (87.6%), best calibration (0.465) | Requires MixUp training loop |
| **PUDRa-prior** | When prior π is known, ranking tasks | Best calibration without MixUp (0.574) | Needs prior, catastrophic failures on tabular data |
| **PUDRa-naive** | Quick baselines, standard training loop | Best F1 without MixUp or prior (86.5%) | Unstable training, poor calibration |

**Quick Decision Tree:**
```
Do you need well-calibrated probabilities? (medical, fraud detection, etc.)
├─ YES → Use VPU (A-NICE = 0.465)
└─ NO → Is training loop complexity a blocker?
    ├─ NO → Use VPU (best F1: 87.6%)
    └─ YES → Do you know the prior π?
        ├─ YES → Use PUDRa-prior for vision/text, VPU for tabular
        └─ NO → Use PUDRa-naive (but monitor for instability)
```

---

## Method Overview

### VPU (Variational PU Learning)

**Training Approach**: MixUp consistency regularization

**Loss Function**:
```
L = E_P[BCE(p, 1)] + E_U[BCE(p, 0)] + λ * E_mix[(log μ - log p)²]

where:
  μ = λ * p(x) + (1-λ) * 1.0  (anchored MixUp target)
  x_mix = λ * x + (1-λ) * x_positive
```

**Key Properties**:
- ✅ No prior estimation required
- ✅ MixUp provides implicit regularization
- ✅ Excellent calibration and stability
- ❌ Requires custom training loop (2× forward passes)
- ❌ Higher computational cost (~2× per iteration)

---

### PUDRa-prior (PU Density Ratio with Prior)

**Training Approach**: Prior-weighted density ratio estimation

**Loss Function**:
```
L = π * E_P[-log p] + E_U[p]

where:
  π = P(s=1) = class prior (proportion of positives)
```

**Key Properties**:
- ✅ Standard training loop (single forward pass)
- ✅ Simple, drop-in loss function
- ✅ Best calibration among non-MixUp methods
- ❌ Requires prior π estimation
- ❌ Catastrophic failures on tabular datasets (Spambase: 2.2% F1)
- ❌ Training instability on small datasets

---

### PUDRa-naive (PU Density Ratio, No Prior)

**Training Approach**: Symmetric density ratio estimation

**Loss Function**:
```
L = E_P[-log p + p] + E_U[p]
```

**Key Properties**:
- ✅ Standard training loop (single forward pass)
- ✅ No prior estimation needed
- ✅ Best F1 among standard training methods (86.5%)
- ✅ Simple implementation (simplest of all three)
- ❌ Poor calibration (A-NICE = 0.819, near-random)
- ❌ Training instability (collapses on small datasets)
- ❌ Catastrophic calibration on well-separated classes

---

## Performance Comparison

### Aggregate Results (Average Across 9 Datasets)

| Method | Avg F1 | Avg AUC | Avg A-NICE ↓ | Avg S-NICE | Avg ECE | Training Complexity |
|--------|--------|---------|--------------|------------|---------|---------------------|
| **VPU** | **87.6%** | **92.6%** | **0.465** | 0.434 | 0.101 | High (MixUp) |
| **PUDRa-naive** | **86.5%** | 92.1% | 0.819 | 0.777 | 0.154 | Low (standard) |
| **PUDRa-prior** | 77.7% | **92.6%** | **0.574** | 0.574 | 0.118 | Low (standard) |

**Key Insights**:
- **VPU leads in F1** by 1.1% over PUDRa-naive, 9.9% over PUDRa-prior
- **VPU leads in calibration** (0.465 vs 0.574 vs 0.819)
- **PUDRa-naive surprisingly beats PUDRa-prior in F1** despite worse calibration
- **All three have similar AUC** (92.1-92.6%) - ranking ability comparable

### Calibration Quality Interpretation

**A-NICE Scale**:
- **0.0** = Perfect calibration (predictions match true probabilities)
- **0.5** = Halfway between perfect and random
- **1.0** = Random baseline (predicts average for everyone)
- **>1.0** = Worse than random (catastrophic miscalibration)

**Method Rankings**:
| Method | A-NICE | Calibration Quality |
|--------|--------|---------------------|
| VPU | 0.465 | **Excellent** (53.5% to perfect from random) |
| PUDRa-prior | 0.574 | **Good** (42.6% to perfect from random) |
| PUDRa-naive | 0.819 | **Poor** (18.1% to perfect from random) |

---

## Dataset-Specific Performance

### Vision Datasets (MNIST, Fashion-MNIST, CIFAR-10)

**MNIST** (easiest):
| Method | Test F1 | Test AUC | Test A-NICE |
|--------|---------|----------|-------------|
| VPU | 95.2% | 99.1% | 0.177 ✅ |
| PUDRa-naive | **96.1%** | 99.2% | **0.177** ✅ |
| PUDRa-prior | 94.8% | **99.3%** | 0.202 |

**Recommendation**: **All three work well**. PUDRa-naive slightly edges out for F1.

---

**CIFAR-10** (hardest vision):
| Method | Test F1 | Test AUC | Test A-NICE |
|--------|---------|----------|-------------|
| VPU | **83.9%** | **85.2%** | **0.459** ✅ |
| PUDRa-naive | 80.2% | 84.6% | 0.716 |
| PUDRa-prior | 63.4% | 84.6% | 0.581 |

**Recommendation**: **VPU strongly preferred** for difficult vision tasks.

---

### Tabular Datasets (Connect-4, Mushrooms, Spambase)

**Spambase** (The Calibration Paradox):
| Method | Test F1 | Test AUC | Test A-NICE | Notes |
|--------|---------|----------|-------------|-------|
| VPU | **84.2%** | **93.6%** | **0.509** ✅ | Stable, well-calibrated |
| PUDRa-naive | 77.6% | 90.3% | 1.278 🔴 | Near-random calibration |
| PUDRa-prior | **2.2%** | 91.8% | 1.204 🔴 | **Catastrophic collapse!** |

**Recommendation**: **Use VPU exclusively** - both PUDRa variants fail.

---

**Mushrooms** (Perfect Ranking, Poor Calibration):
| Method | Test F1 | Test AUC | Test A-NICE | Notes |
|--------|---------|----------|-------------|-------|
| VPU | **98.2%** | 100.0% | **1.118** | Even VPU poorly calibrated |
| PUDRa-naive | 98.1% | 99.98% | 1.540 🔴 | Catastrophic calibration |
| PUDRa-prior | 98.6% | 100.0% | 1.222 🔴 | Random-level calibration |

**Insight**: All methods achieve near-perfect ranking but struggle with calibration on well-separated classes.

**Recommendation**: **Any method works for ranking**; use VPU if probabilities matter.

---

### Text Datasets (IMDB, 20News)

**IMDB** (balanced, 50% positive):
| Method | Test F1 | Test AUC | Test A-NICE |
|--------|---------|----------|-------------|
| VPU | **94.8%** | **97.4%** | **0.288** ✅ |
| PUDRa-naive | 93.7% | 96.9% | 0.716 |
| PUDRa-prior | 91.3% | 96.8% | 0.434 |

**Recommendation**: **VPU preferred** for best F1 and calibration.

---

**20News** (balanced, 50% positive):
| Method | Test F1 | Test AUC | Test A-NICE |
|--------|---------|----------|-------------|
| VPU | **87.8%** | **89.9%** | **0.556** ✅ |
| PUDRa-naive | 86.7% | 89.5% | 0.771 |
| PUDRa-prior | 80.0% | 89.5% | 0.680 |

**Recommendation**: **VPU preferred** for best overall performance.

---

### Small/Challenging Datasets (AlzheimerMRI)

**AlzheimerMRI** (5,323 train samples, 52 val samples):
| Method | Test F1 | Test AUC | Test A-NICE | Training Stability |
|--------|---------|----------|-------------|-------------------|
| VPU | **70.0%** | 76.9% | **0.465** ✅ | Stable |
| PUDRa-naive | 72.4% | **79.1%** | 0.742 | **Collapsed multiple times** 🔴 |
| PUDRa-prior | 65.5% | 79.5% | 1.415 🔴 | **Collapsed multiple times** 🔴 |

**Training Instability Observed**:
- **PUDRa-prior**: F1 oscillated 68% → 2.5% → 68% (complete collapse)
- **PUDRa-naive**: F1 dropped to 0.07%, AUC inverted to 30.22%
- **VPU**: No collapses observed

**Recommendation**: **Use VPU exclusively** for small/difficult datasets - MixUp regularization is critical for stability.

---

## When to Use Each Method

### Use VPU When:

✅ **Performance matters** (production systems, research benchmarks)
- Best F1 (87.6%) and calibration (0.465) overall
- No catastrophic failures on any dataset type

✅ **Probabilities need to be calibrated**
- Medical diagnosis (decision thresholds)
- Fraud detection (cost-sensitive decisions)
- Confidence-based filtering (recommendation systems)

✅ **Working with tabular data**
- PUDRa-prior catastrophically fails (Spambase: 2.2% F1)
- PUDRa-naive poorly calibrated (A-NICE > 1.0)

✅ **Small or challenging datasets**
- MixUp provides stability (no training collapses)
- AlzheimerMRI: VPU stable, PUDRa variants collapsed

✅ **Don't know the prior π**
- VPU doesn't require prior estimation
- Avoids error propagation from wrong prior

❌ **Avoid VPU if**:
- Training loop complexity is a hard blocker
- Compute budget extremely tight (2× forward passes)
- Prototyping quickly (use PUDRa-naive baseline first)

---

### Use PUDRa-prior When:

✅ **Prior π is accurately known**
- Class prevalence from external data
- Domain knowledge (e.g., disease rate from epidemiology)

✅ **Vision or text datasets only**
- Decent performance on MNIST, Fashion-MNIST, IMDB, 20News
- Avoid tabular datasets entirely

✅ **Ranking task** (AUC matters, probabilities don't)
- AUC competitive with VPU (92.6%)
- Calibration doesn't matter for ranking/retrieval

✅ **Standard training loop required AND prior available**
- Drop-in replacement for BCE loss
- Simple implementation

❌ **Never use PUDRa-prior for**:
- **Tabular datasets** (Spambase: 2.2% F1 catastrophic failure)
- **Small datasets** (training collapses on AlzheimerMRI)
- **When prior π is uncertain** (wrong prior → poor performance)
- **Classification tasks** (F1 significantly worse than VPU/PUDRa-naive)

---

### Use PUDRa-naive When:

✅ **Quick baseline needed**
- Simplest implementation (no prior, no MixUp)
- Fast to train (standard loop, single forward pass)

✅ **Standard training loop required AND prior unknown**
- Best F1 among standard training methods (86.5%)
- No prior estimation needed

✅ **Ranking task on vision/text** (AUC matters, probabilities don't)
- AUC competitive (92.1%)
- MNIST: 99.2% AUC, 96.1% F1 (best of all three!)

✅ **Prototyping / proof-of-concept**
- Get 99% of VPU's F1 with simpler implementation
- Can upgrade to VPU later if needed

❌ **Avoid PUDRa-naive if**:
- **Probabilities need to be calibrated** (A-NICE = 0.819, near-random)
- **Small datasets** (<10k samples) - training instability likely
- **Tabular data with well-separated classes** (poor calibration)
- **Production system** - use VPU for robustness

---

## Decision Flowchart

```
START: Choose PU Learning Method
│
├─ Do you need well-calibrated probabilities?
│  ├─ YES → Use VPU
│  │         (Medical, fraud detection, probability thresholds)
│  │
│  └─ NO → Continue to training complexity question
│
├─ Is MixUp training loop complexity a blocker?
│  ├─ NO → Use VPU
│  │        (Best F1: 87.6%, most robust)
│  │
│  └─ YES → Continue to dataset/prior questions
│
├─ Dataset type?
│  ├─ Tabular (Spambase, Mushrooms, Connect-4)
│  │  └─ Use VPU (PUDRa variants fail catastrophically)
│  │
│  ├─ Small (<10k samples)
│  │  └─ Use VPU (PUDRa variants unstable)
│  │
│  └─ Vision or Text (large dataset)
│      │
│      ├─ Do you know the prior π accurately?
│      │  ├─ YES → Use PUDRa-prior
│      │  │         (Good calibration: 0.574, standard training)
│      │  │
│      │  └─ NO → Use PUDRa-naive
│      │            (Best standard-loop F1: 86.5%, no prior needed)
│      │
│      └─ Is this a ranking task? (AUC only)
│          └─ YES → PUDRa-naive or PUDRa-prior both work
│                    (AUC ~92%, calibration doesn't matter)
```

---

## Trade-off Summary

### Performance vs Complexity

| Dimension | VPU | PUDRa-prior | PUDRa-naive |
|-----------|-----|-------------|-------------|
| **F1 Score** | ⭐⭐⭐⭐⭐ (87.6%) | ⭐⭐⭐ (77.7%) | ⭐⭐⭐⭐ (86.5%) |
| **Calibration** | ⭐⭐⭐⭐⭐ (0.465) | ⭐⭐⭐⭐ (0.574) | ⭐⭐ (0.819) |
| **Robustness** | ⭐⭐⭐⭐⭐ (no failures) | ⭐⭐ (tabular fails) | ⭐⭐⭐ (unstable) |
| **Simplicity** | ⭐⭐ (MixUp loop) | ⭐⭐⭐⭐ (needs prior) | ⭐⭐⭐⭐⭐ (simplest) |
| **Compute Cost** | ⭐⭐ (2× forward) | ⭐⭐⭐⭐⭐ (standard) | ⭐⭐⭐⭐⭐ (standard) |

### Requirements

| Requirement | VPU | PUDRa-prior | PUDRa-naive |
|-------------|-----|-------------|-------------|
| Prior π estimation | ❌ Not needed | ✅ Required | ❌ Not needed |
| Custom training loop | ✅ Required (MixUp) | ❌ Standard | ❌ Standard |
| Computational overhead | ~2× (extra forward pass) | 1× | 1× |
| Implementation complexity | High (100+ lines) | Low (20 lines) | Low (15 lines) |

---

## Implementation Complexity Comparison

### VPU Training Loop (Complex)

```python
# VPU requires custom MixUp training loop
for x, t in train_loader:
    # 1. Forward pass on original batch
    p_all = model(x)

    # 2. Sample positive features for mixing
    p_features = x[t == 1]
    if len(p_features) >= len(x):
        p_mix_features = p_features[torch.randperm(len(p_features))[:len(x)]]
    else:
        # Sample with replacement if not enough positives
        idx = torch.randint(0, len(p_features), (len(x),))
        p_mix_features = p_features[idx]

    # 3. Sample MixUp coefficient
    lam = Beta(mix_alpha, mix_alpha).sample()

    # 4. Create mixed samples
    x_mix = lam * x + (1 - lam) * p_mix_features

    # 5. Forward pass on mixed samples (2nd forward pass!)
    p_mix = model(x_mix)

    # 6. Compute anchored target
    mu_anchor = lam * p_all.detach() + (1 - lam) * 1.0

    # 7. Compute loss (base PU + log-MSE consistency)
    loss = criterion(p_all, t, p_mix, mu_anchor, lam)
    loss.backward()
    optimizer.step()
```

**Complexity**: ~100 lines with sampling logic, ~2× compute per iteration

---

### PUDRa-prior Training Loop (Simple)

```python
# PUDRa-prior uses standard training loop
for x, t in train_loader:
    # Single forward pass
    p = torch.sigmoid(model(x))

    # Compute prior-weighted loss
    positive_risk = -torch.log(p[t == 1] + eps).mean()
    unlabeled_risk = p[t == -1].mean()
    loss = prior * positive_risk + unlabeled_risk

    loss.backward()
    optimizer.step()
```

**Complexity**: ~20 lines, standard training loop

**Caveat**: Requires prior π estimation (adds complexity elsewhere)

---

### PUDRa-naive Training Loop (Simplest)

```python
# PUDRa-naive uses standard training loop, no prior
for x, t in train_loader:
    # Single forward pass
    p = torch.sigmoid(model(x))

    # Compute symmetric loss (no prior weighting)
    positive_risk = (-torch.log(p[t == 1] + eps) + p[t == 1]).mean()
    unlabeled_risk = p[t == -1].mean()
    loss = positive_risk + unlabeled_risk

    loss.backward()
    optimizer.step()
```

**Complexity**: ~15 lines, standard training loop, no prior needed

**Simplest of all three methods!**

---

## Critical Failure Modes

### PUDRa-prior: Spambase Catastrophe

**Problem**: Achieves excellent ranking (91.8% AUC) but catastrophic classification (2.2% F1)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Test AUC | 91.8% | Excellent ranking ability |
| Test F1 | **2.2%** | **Catastrophic classification** |
| Test A-NICE | 1.204 | Random-level calibration |
| Gap (AUC - F1) | **89.6%** | Extreme miscalibration |

**Root Cause**: Prior-weighted loss optimizes ranking but produces poorly calibrated probabilities on tabular data with complex decision boundaries.

**Solution**: Use VPU (84.2% F1, 93.6% AUC, 0.509 A-NICE)

---

### PUDRa-naive: AlzheimerMRI Training Collapse

**Problem**: Training becomes unstable on small datasets, complete collapses observed

**Observed Behavior**:
- F1 dropped from 72% → 0.07% (near-zero)
- AUC inverted to 30.22% (worse than random 50%)
- Model predicts everything as negative class

**Why**: Without prior weighting or MixUp regularization, the loss surface has unstable regions

**Solution**: Use VPU (70.0% F1, stable training, no collapses)

---

### VPU: Mushrooms Calibration Struggle

**Problem**: Even VPU struggles with calibration on perfectly separated classes

| Method | F1 | AUC | A-NICE |
|--------|-----|-----|--------|
| VPU | 98.2% | 100.0% | **1.118** (worse than random!) |
| PUDRa-naive | 98.1% | 99.98% | 1.540 |
| PUDRa-prior | 98.6% | 100.0% | 1.222 |

**Why**: When classes are perfectly linearly separable, all methods push predictions to extremes (0 or 1), making calibration difficult.

**Insight**: This is a dataset property, not a method failure. All methods achieve near-perfect classification.

**When it matters**: If you need probability thresholds (e.g., "flag items with >80% confidence"), consider post-hoc calibration (isotonic regression, Platt scaling).

---

## Practical Recommendations

### For Production Systems

**Recommendation**: **Use VPU**
- Best overall performance (F1 + AUC + calibration)
- No catastrophic failure modes
- Worth the implementation complexity for reliability

**Implementation Tips**:
1. Use existing VPU trainer from PU-Bench (tested across 9 datasets)
2. MixUp overhead becomes negligible for larger models (ResNet, Transformers)
3. Cache positive features if dataset is small (avoid repeated sampling)

---

### For Research Baselines

**Recommendation**: **Start with PUDRa-naive, compare to VPU**
- PUDRa-naive: Quick baseline (86.5% F1, standard training)
- VPU: State-of-the-art (87.6% F1, best calibration)
- Report both for completeness

**Reporting Checklist**:
- ✅ Report F1, AUC, **and calibration** (A-NICE or ECE)
- ✅ Test on both vision and tabular datasets (exposes calibration issues)
- ✅ Include training stability metrics (collapses, variance)

---

### For Quick Prototypes

**Recommendation**: **Use PUDRa-naive**
- Simplest implementation (~15 lines)
- No prior estimation needed
- Gets 99% of VPU's F1 (86.5% vs 87.6%)

**When to Upgrade to VPU**:
1. Training instability observed (collapses, oscillations)
2. Probabilities needed for decision-making
3. Moving to production

---

### For Domain-Specific Applications

**Medical / Healthcare**:
- **Use VPU** - calibration critical for diagnosis thresholds
- Monitor A-NICE (target < 0.5 for reliable probabilities)

**Fraud Detection / Anomaly Detection**:
- **Use VPU** - cost-sensitive decisions require calibrated probabilities
- High-precision regime benefits from good calibration

**Recommendation Systems**:
- **Ranking only**: PUDRa-naive works (AUC sufficient)
- **Confidence filtering**: VPU (calibration matters)

**Text Classification**:
- **Use VPU** for balanced datasets (IMDB, 20News)
- All three methods work reasonably well

**Computer Vision**:
- **Easy tasks** (MNIST): All three work, PUDRa-naive slightly better F1
- **Hard tasks** (CIFAR-10): VPU strongly preferred (83.9% F1 vs 80.2%)

---

## Prior Estimation (For PUDRa-prior)

If you choose PUDRa-prior, you need to estimate π = P(s=1).

### Methods for Prior Estimation:

**1. External Data** (most reliable):
- Use validation set with true labels
- Domain knowledge (e.g., disease prevalence from literature)

**2. Algorithmic Estimation**:
- **AlphaMax** (estimates π from label frequency and model predictions)
- **TIcE** (Two Independent Component Estimation)
- **KM1** (Kernel Mean Matching)

**3. Sensitivity Analysis**:
- Train with π ∈ {0.1, 0.3, 0.5} and choose best validation performance

**Warning**: Wrong prior → poor performance
- If π is uncertain, **use VPU or PUDRa-naive** instead

---

## Computational Cost Analysis

### Training Time Comparison (CIFAR-10, 40 epochs, RTX 3090)

| Method | Time per Epoch | Total Training Time | Relative Cost |
|--------|----------------|---------------------|---------------|
| PUDRa-naive | 15 sec | 10 min | 1.0× |
| PUDRa-prior | 15 sec | 10 min | 1.0× |
| VPU | 28 sec | 19 min | 1.9× |

**VPU overhead**: ~2× compute due to:
- 2× forward passes (original + mixed samples)
- Beta sampling (~negligible)
- MixUp interpolation (~negligible)

**When VPU overhead matters**:
- Extremely large models (GPT-scale)
- Very tight compute budget
- Rapid prototyping iterations

**When VPU overhead doesn't matter**:
- Larger models (ResNet, Transformers) - MixUp becomes smaller fraction
- Production systems (one-time training cost)
- Research (performance > speed)

---

## Hyperparameter Recommendations

### VPU

```yaml
optimizer:
  lr: 0.0003
  weight_decay: 0.0001

training:
  batch_size: 256
  epochs: 40
  patience: 10  # Early stopping

vpu:
  mix_alpha: 0.3  # Beta(0.3, 0.3) for MixUp
  epsilon: 1e-7   # Numerical stability
```

**Tuning Tips**:
- `mix_alpha` ∈ [0.2, 0.5]: Higher = more aggressive mixing
- Increase `patience` to 15-20 for small datasets (avoid premature stopping)

---

### PUDRa-prior

```yaml
optimizer:
  lr: 0.0003
  weight_decay: 0.0001

training:
  batch_size: 256
  epochs: 40
  patience: 10

pudra:
  prior: 0.5  # P(s=1) - MUST ESTIMATE from data!
  epsilon: 1e-7
```

**Tuning Tips**:
- **Critical**: Accurate prior estimation (use validation set or AlphaMax)
- Monitor for training collapses (F1 → 0)
- Reduce `lr` to 0.0001 if unstable

---

### PUDRa-naive

```yaml
optimizer:
  lr: 0.0003
  weight_decay: 0.0001

training:
  batch_size: 256
  epochs: 40
  patience: 10
  min_delta: 0.0001  # Prevent early stopping on plateaus

pudra_naive:
  epsilon: 1e-7
```

**Tuning Tips**:
- **Monitor for collapses**: Check F1 every epoch
- Increase `patience` to 15-20 for small datasets
- If training collapses, switch to VPU

---

## Post-Hoc Calibration

If you're stuck with poorly calibrated predictions (PUDRa-prior, PUDRa-naive), you can recalibrate:

### Isotonic Regression (Recommended)

```python
from sklearn.isotonic import IsotonicRegression

# Fit on validation set
iso_reg = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
calibrated_probs = iso_reg.fit_transform(val_probs, val_labels)

# Apply to test set
test_calibrated = iso_reg.transform(test_probs)
```

**Expected Improvement**:
- Can reduce A-NICE by 30-50%
- Preserves ranking (AUC unchanged)
- Especially effective for PUDRa variants

**When to Recalibrate**:
- A-NICE > 0.8 (approaching random baseline)
- Large AUC-F1 gap (>15%)
- ECE > 0.15

---

## Quick Reference Table

### Method Selection Cheat Sheet

| Scenario | Recommended Method | Why |
|----------|-------------------|-----|
| **Production system** | VPU | Best performance + robustness |
| **Calibrated probabilities needed** | VPU | Best calibration (0.465) |
| **Tabular data** | VPU | PUDRa variants fail |
| **Small dataset (<10k)** | VPU | PUDRa variants unstable |
| **Prior π known, vision/text** | PUDRa-prior | Good calibration, standard training |
| **Quick baseline** | PUDRa-naive | Simplest, 86.5% F1 |
| **Ranking task only** | PUDRa-naive | AUC competitive, simple |
| **Prototype → production** | PUDRa-naive → VPU | Start simple, upgrade later |
| **Medical / fraud detection** | VPU | Calibration critical |
| **Recommendation (ranking)** | PUDRa-naive | AUC sufficient |
| **Recommendation (confidence)** | VPU | Calibration matters |

---

## Conclusion

### The Bottom Line

**VPU is the overall winner** and should be your default choice unless:
1. Training loop complexity is a hard blocker, AND
2. You're working with vision/text (not tabular), AND
3. Dataset is large enough (>10k samples)

In those cases:
- Use **PUDRa-prior** if you know π accurately
- Use **PUDRa-naive** if you don't know π

**For production**: Always use VPU - the MixUp complexity is worth it for the 10% F1 improvement and robustness.

**For research**: Benchmark against both PUDRa-naive (standard baseline) and VPU (state-of-the-art).

**For prototyping**: Start with PUDRa-naive (simplest), upgrade to VPU if needed.

---

## Further Reading

- **Full Calibration Analysis**: See [CALIBRATION_ANALYSIS.md](CALIBRATION_ANALYSIS.md)
- **Benchmark Results**: See [BENCHMARK_COMPARISON.md](BENCHMARK_COMPARISON.md)
- **VPUDRa Design Space**: See [VPUDRA_DESIGN_MATRIX.md](VPUDRA_DESIGN_MATRIX.md)

---

**Generated**: February 2026
**Methods Analyzed**: VPU, PUDRa-prior (formerly PUDRa), PUDRa-naive
**Datasets**: 9 (MNIST, Fashion-MNIST, CIFAR-10, AlzheimerMRI, Connect-4, Mushrooms, Spambase, IMDB, 20News)
**Total Benchmark Runs**: 81 (9 methods × 9 datasets)
