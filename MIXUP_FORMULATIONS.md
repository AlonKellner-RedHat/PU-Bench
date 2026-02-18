# MixUp Consistency Formulations: VPU vs PUDRa-Aligned Alternatives

**Author**: Analysis based on VPU (NeurIPS 2020) and PUDRa implementations
**Date**: 2026-02-17

## Executive Summary

VPU's MixUp formulation makes a **strong anchor assumption** (p(positive) = 1.0) that doesn't align with PUDRa's philosophy. We propose three theoretically grounded alternatives that maintain smoothness regularization while being more consistent with PUDRa's assumptions:

1. **Point Process Soft Label** ⭐ (Recommended) - Natural extension of PUDRa's loss to soft labels
2. **Multi-Strategy Mixing** - Treats P-P, P-U, U-U mixing differently
3. **Manifold Smoothness** - Enforces convex combination property without anchor assumption

---

## Current VPU MixUp Formulation

### Implementation
```python
# Mix samples
sam_data = λ * x + (1-λ) * p_mix  # p_mix is a random positive sample

# Create target (ANCHOR ASSUMPTION)
sam_target = λ * p(x) + (1-λ) * 1.0  # assumes p(positive) = 1.0 exactly

# MixUp regularization (log-MSE)
reg_mix_log = ((log(sam_target) - log(p(sam_data)))²).mean()

# Total loss (weighted by λ)
total_loss = var_loss + λ * reg_mix_log
```

### Theoretical Issues

**Problem 1: Anchor Assumption**
- Assumes **all positive samples** have p(x) = 1.0 exactly
- In reality, positive samples may have p(x) < 1 (label noise, overlap regions, etc.)
- **Not justified theoretically** - VPU paper states "anchor assumption" but doesn't prove it holds

**Problem 2: Positive-Only Mixing**
- Only mixes with positive samples
- Doesn't explore unlabeled-unlabeled manifold smoothness
- **Ad-hoc design choice** - standard MixUp mixes randomly within batch

**Problem 3: λ Weighting**
- Weights consistency loss by λ (non-standard)
- VPU paper doesn't justify this specific weighting
- **Likely empirically tuned** rather than principled

**Problem 4: Log-MSE vs PUDRa Structure**
- Uses symmetric log-MSE: (log a - log b)²
- PUDRa uses asymmetric Point Process: -log p + p
- **Inconsistent** with PUDRa's theoretical foundation

### What VPU Gets Right ✅
- ✅ MixUp provides smoothness regularization (proven effective)
- ✅ Prevents overfitting and trivial classifier collapse
- ✅ Empirically works very well (87.57% avg F1 on SCAR)

---

## Alternative 1: Point Process Soft Label Consistency ⭐

### Core Idea
**Extend PUDRa's Point Process loss to soft labels via weighted combination.**

For a discrete label y ∈ {0,1}, PUDRa uses:
- L(1, p) = -log p + p  (for positives)
- L(0, p) = p  (for negatives/unlabeled)

For a **soft label μ ∈ [0,1]** (expected label), natural extension:
- L(μ, p) = μ · L(1, p) + (1-μ) · L(0, p)
- **Simplifies to**: L(μ, p) = -μ log p + p

### Mathematical Formulation
```python
# Mix samples (same as VPU)
sam_data = λ * x + (1-λ) * p_mix

# Compute soft label (NO ANCHOR ASSUMPTION)
μ = λ * p(x).detach() + (1-λ) * p(p_mix).detach()

# Point Process soft label loss
p_pred = p(sam_data)
consistency_loss = -μ * log(p_pred + ε) + p_pred

# Total loss
L = π * E_P[-log p] + E_U[p] + λ * consistency_loss
```

### Advantages ✅
1. ✅ **Theoretically grounded**: Natural extension of PUDRa's loss structure
2. ✅ **No anchor assumption**: Uses actual predictions, not assumed p=1
3. ✅ **Asymmetric penalty**: Inherits PUDRa's properties
   - -μ log p term encourages p→1 when μ is high
   - +p term prevents collapse when μ is low
4. ✅ **Interpretable**: Expected loss under mixture distribution
5. ✅ **Simple**: Only one additional term, easy to implement

### When This Works Best
- When positive samples may not all have p=1 (label noise, overlap)
- When you want consistency with PUDRa's theoretical framework
- When interpretability matters

### Potential Concerns ⚠️
- ⚠️ Asymmetric penalty may behave differently than log-MSE
- ⚠️ Less empirical validation (VPU's log-MSE is battle-tested)
- ⚠️ Still mixes with positives only (like VPU)

---

## Alternative 2: Multi-Strategy Mixing

### Core Idea
**Use three different mixing strategies aligned with PUDRa's treatment of positive vs unlabeled.**

Unlike VPU which only mixes with positives, we recognize three distinct scenarios:

1. **Positive-Positive (P-P)**: Both samples are positive
   - Expected label ≈ 1 (both are labeled positives)
   - Use hard constraint: L(1, p) = -log p + p

2. **Positive-Unlabeled (P-U)**: One positive, one unlabeled
   - Expected label ≈ λ * 1 + (1-λ) * p(u)
   - Use soft label: L(μ, p) = -μ log p + p

3. **Unlabeled-Unlabeled (U-U)**: Both samples are unlabeled
   - Expected label ≈ λ * p(u1) + (1-λ) * p(u2) ≈ π (prior)
   - Use soft label: L(μ, p) = -μ log p + p

### Mathematical Formulation
```python
# Three mixing strategies
sam_pp = λ * p1 + (1-λ) * p2  # both positive
sam_pu = λ * p + (1-λ) * u     # positive + unlabeled
sam_uu = λ * u1 + (1-λ) * u2   # both unlabeled

# Corresponding soft labels
μ_pp = λ * 1 + (1-λ) * 1 = 1  # hard label
μ_pu = λ * 1 + (1-λ) * p(u).detach()
μ_uu = λ * p(u1).detach() + (1-λ) * p(u2).detach()

# Point Process consistency for each strategy
L_pp = -μ_pp * log(p(sam_pp)) + p(sam_pp)
L_pu = -μ_pu * log(p(sam_pu)) + p(sam_pu)
L_uu = -μ_uu * log(p(sam_uu)) + p(sam_uu)

# Total consistency (average across strategies)
consistency_loss = (L_pp + L_pu + L_uu) / 3

# Total loss
L = π * E_P[-log p] + E_U[p] + λ * consistency_loss
```

### Advantages ✅
1. ✅ **More thorough**: Enforces smoothness across entire manifold (not just toward positives)
2. ✅ **Philosophically aligned**: Treats P and U differently like PUDRa
3. ✅ **Richer signal**: Three strategies provide diverse consistency constraints
4. ✅ **U-U mixing**: Explores unlabeled manifold (VPU doesn't)

### When This Works Best
- When you have sufficient unlabeled samples for U-U mixing
- When manifold structure is complex (benefits from comprehensive smoothness)
- When you want maximum consistency with PUDRa's philosophy

### Potential Concerns ⚠️
- ⚠️ More complex implementation (three strategies vs one)
- ⚠️ Slower training (3× mixing operations per batch)
- ⚠️ Batch composition matters (need enough P and U samples)
- ⚠️ Untested empirically (novel formulation)

---

## Alternative 3: Manifold Smoothness (Convex Combination)

### Core Idea
**Enforce that predictions on mixed samples follow convex combinations, without anchor assumptions.**

Under the manifold hypothesis, for mixed samples:
```
sam_data = λ * x1 + (1-λ) * x2
```

Prediction should satisfy (approximately):
```
p(sam_data) ≈ λ * p(x1) + (1-λ) * p(x2)
```

This is a **geometric smoothness constraint** - no label assumptions needed!

### Mathematical Formulation
```python
# Mix samples
sam_data = λ * x + (1-λ) * x_mix

# Target: convex combination of predictions (detached)
p_target = λ * p(x).detach() + (1-λ) * p(x_mix).detach()

# Smoothness penalty (choose one):

# Option A: MSE in probability space
smoothness = |p(sam_data) - p_target|²

# Option B: MSE in log space (KL-like)
smoothness = |log p(sam_data) - log p_target|²

# Option C: KL divergence
smoothness = KL(p_target || p(sam_data))

# Total loss
L = π * E_P[-log p] + E_U[p] + λ * smoothness
```

### Advantages ✅
1. ✅ **No label assumptions**: Pure geometric smoothness (manifold hypothesis only)
2. ✅ **Symmetric**: Can mix any pairs (P-P, P-U, U-U) with same formulation
3. ✅ **Interpretable**: Direct enforcement of convex combination property
4. ✅ **Flexible**: Multiple distance metrics (prob, log, KL)
5. ✅ **Generalizable**: Applies to any density estimation, not just classification

### When This Works Best
- When you trust the manifold hypothesis more than anchor assumption
- When you want maximum flexibility in mixing strategies
- When you value geometric interpretability

### Potential Concerns ⚠️
- ⚠️ Weaker constraint than VPU's (doesn't enforce toward p=1)
- ⚠️ Detached predictions mean no gradient feedback on smoothness
- ⚠️ May not prevent collapse as effectively as VPU's formulation
- ⚠️ Choice of distance metric affects performance

---

## Theoretical Comparison

| Aspect | VPU (Original) | Soft Label | Multi-Mix | Manifold |
|--------|---------------|------------|-----------|----------|
| **Anchor Assumption** | ✗ Requires p(pos)=1 | ✅ No assumption | ✅ No assumption | ✅ No assumption |
| **PUDRa Consistency** | ✗ Uses log-MSE | ✅ Point Process | ✅ Point Process | ⚠️ Geometric only |
| **Mixing Strategy** | Positive-only | Positive-only | P-P, P-U, U-U | Flexible |
| **Theoretical Rigor** | ⚠️ Ad-hoc | ✅ Natural extension | ✅ Systematic | ✅ Manifold hypothesis |
| **Implementation** | ✅ Simple | ✅ Simple | ⚠️ Complex | ✅ Simple |
| **Empirical Validation** | ✅ Proven (87.57%) | ❓ Untested | ❓ Untested | ❓ Untested |

---

## Recommended Path Forward

### Option A: Empirical Validation (Recommended)
Test all three alternatives on SCAR benchmark and compare:

**Hypothesis 1 - Soft Label**:
- Should match VPU on simple datasets where anchor holds (MNIST, Fashion-MNIST)
- Should outperform VPU on complex datasets where anchor fails (AlzheimerMRI, Spambase)
- **Prediction**: 85-88% avg F1 on SCAR

**Hypothesis 2 - Multi-Mix**:
- Should provide best consistency constraints (most comprehensive)
- May be slower but more robust
- **Prediction**: 86-89% avg F1 on SCAR, best SAR robustness

**Hypothesis 3 - Manifold**:
- Most principled but weakest constraint
- May underperform on datasets requiring strong regularization
- **Prediction**: 83-86% avg F1 on SCAR

**Test protocol**:
```bash
# Create configs for all three variants
# Run on 2-3 datasets first (quick validation)
uv run python run_train.py \
  --dataset-config config/datasets_typical/param_sweep_mnist_single.yaml \
                    config/datasets_typical/param_sweep_spambase_single.yaml \
  --methods vpudra_softlabel vpudra_multimix vpudra_manifold vpudra_fixed vpu

# If promising, run full SCAR benchmark (9 datasets)
# If very promising, run SAR benchmark (4 datasets × 2 strategies)
```

### Option B: Theoretical Analysis First
**Derive formal properties**:
1. Show soft label loss is equivalent to expected Point Process loss under mixture
2. Prove multi-mix provides stronger Lipschitz constraint than single-strategy
3. Analyze manifold smoothness connection to generalization bounds

**Then validate empirically** on datasets where theory predicts advantages.

---

## Implementation Notes

All three variants are implemented in:
- `loss/loss_vpudra_softlabel.py` - Point Process soft label consistency
- `loss/loss_vpudra_multimix.py` - Multi-strategy mixing (P-P, P-U, U-U)
- `loss/loss_vpudra_manifold.py` - Manifold smoothness (convex combination)

**Key implementation details**:
1. All use `detach()` on target predictions to prevent gradient feedback
2. All weight consistency by λ (matching VPU's design)
3. All maintain PUDRa's base loss: π * E_P[-log p] + E_U[p]
4. All handle edge cases (no positives, no unlabeled, empty batches)

**To test**:
- Create trainer classes (similar to `vpudra_fixed_trainer.py`)
- Create config files with hyperparameters
- Register in `run_train.py`
- Run benchmarks

---

## Open Questions

1. **Does anchor assumption matter in practice?**
   - VPU assumes p(positive) = 1, but AlzheimerMRI suggests positives may have p < 1
   - Soft label formulation tests this directly

2. **Is positive-only mixing optimal?**
   - VPU only mixes toward positives
   - Multi-mix explores entire manifold
   - Which provides better regularization?

3. **Is VPU's λ weighting justified?**
   - All variants inherit this design choice
   - Should we test constant weighting instead?

4. **What's the right distance metric?**
   - VPU uses log-MSE
   - Soft label uses Point Process
   - Manifold offers prob/log/KL
   - Which aligns best with PU learning objectives?

5. **Can we combine ideas?**
   - Soft label + Multi-mix?
   - Manifold smoothness + Point Process penalty?
   - Adaptive weighting instead of λ?

---

## Conclusion

VPU's MixUp formulation is **empirically effective but theoretically questionable** from a PUDRa perspective. The three proposed alternatives offer:

1. **Soft Label**: Most direct PUDRa extension, removes anchor assumption
2. **Multi-Mix**: Most comprehensive, explores entire manifold
3. **Manifold**: Most principled, pure geometric smoothness

**Next step**: Implement trainers and run empirical validation to determine which formulation provides the best balance of theoretical rigor and practical performance.

**My recommendation**: Start with **Soft Label** as it's the most straightforward extension and directly addresses VPU's anchor assumption issue while maintaining simplicity.
