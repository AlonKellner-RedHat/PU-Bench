# Extended Checkpoint Pool Design

## Motivation

Previous toy experiments used small checkpoint pools (12-18 checkpoints) from only early training stages (1, 5, 10 epochs). This limited:
1. Diversity of training stages
2. Understanding of how meta-learning handles well-trained vs poorly-trained models
3. Comparison between oracle (PN) and naive (PU) training approaches

## New Extended Pool Design

### Pool Size: 560 Checkpoints

**Breakdown:**
- 4 task difficulties (mean_separation: 1.5, 2.0, 2.5, 3.0)
- 2 overlap levels (std: 0.8, 1.2)
- 5 random seeds (42, 123, 456, 789, 999)
- 2 training methods (oracle, naive)
- 7 training stages (1, 5, 10, 20, 50, 100, 200 epochs)

Total: 4 × 2 × 5 × 2 × 7 = **560 checkpoints**

### Training Methods

**Oracle Checkpoints (280 total):**
- Train with ground truth PN labels
- Represents best-case scenario (access to all true labels)
- Should converge to near-optimal classifiers
- Performance should improve monotonically with epochs

**Naive Checkpoints (280 total):**
- Train with PU labels only (30% positives labeled, rest unlabeled)
- Treats unlabeled as negative (naive PU approach)
- Expected to overfit badly - treating hidden positives as negatives
- Performance may degrade with more training (as seen in test run)

### Training Stages

**Early (1, 5, 10 epochs):**
- Models just beginning to learn
- High variance, underfitting
- Both oracle and naive may perform similarly

**Mid-training (20, 50 epochs):**
- Models starting to converge
- Oracle improves, naive starts to degrade
- Gap between oracle and naive widens

**Near-convergence (100, 200 epochs):**
- Oracle: well-converged, optimal performance
- Naive: severely overfit to wrong labels
- Largest performance gap

## Expected Results from Test Run (48 checkpoints)

From the smaller test run, we observed:

### Oracle Performance (PN-trained):
```
Epoch   1: 0.588 BCE
Epoch   5: 0.351 BCE
Epoch  10: 0.285 BCE
Epoch  20: 0.264 BCE
Epoch  50: 0.261 BCE
Epoch 100: 0.263 BCE
```
**Pattern:** Converges around epoch 20-50, stable performance

### Naive Performance (PU-trained):
```
Epoch   1: 0.625 BCE
Epoch   5: 0.856 BCE
Epoch  10: 1.244 BCE
Epoch  20: 1.765 BCE
Epoch  50: 2.666 BCE
Epoch 100: 3.342 BCE
```
**Pattern:** DEGRADES with training! Overfitting to wrong labels.

### Key Insight: Naive PU Gets Worse

This is expected behavior:
1. **Early training:** Model underfits, makes random predictions
2. **Mid training:** Model learns to classify labeled positives correctly
3. **Late training:** Model confidently misclassifies hidden positives as negatives
   - Unlabeled examples actually contain ~70% positives (since only 30% are labeled)
   - Naive approach treats all unlabeled as negative
   - More training = more confident wrong predictions on hidden positives

## Meta-Learning Objectives

The learned loss should:

1. **Help naive checkpoints more than oracle:**
   - Oracle already has good labels, doesn't need much help
   - Naive has systematically wrong labels, needs correction

2. **Be robust across training stages:**
   - Should work on early (epoch 1), mid (epoch 20), and late (epoch 100) checkpoints
   - May need to adapt differently based on how converged the model is

3. **Bridge the oracle-naive gap:**
   - Current gap: 421% worse (1.750 vs 0.335 BCE)
   - Good meta-learned loss should reduce this gap significantly
   - Ideally: help naive checkpoints approach oracle performance

## What We're Testing

### Question 1: Can meta-learning fix naive PU?
- Naive checkpoints are systematically biased (treating positives as negatives)
- Can a learned loss correct this during the 3-step inner loop?
- Or is the bias too deeply ingrained in the weights?

### Question 2: How does training stage affect meta-learning?
- Does the learned loss work better on early checkpoints (less overfit)?
- Or late checkpoints (more converged, clearer signal)?

### Question 3: Does the learned loss discover asymmetry?
- Previous experiments showed asymmetric loss helps PU
- Will this emerge again with the extended pool?
- Will it be more pronounced given the oracle vs naive split?

### Question 4: Checkpoint diversity
- With 560 checkpoints spanning multiple dimensions of variation:
  - Task difficulty (mean_separation)
  - Task overlap (std)
  - Random initialization (5 seeds)
  - Training method (oracle vs naive)
  - Training stage (7 epochs)
- Can meta-learning generalize across all this diversity?

## Expected Final Learned Loss

Based on previous results, we expect:
- a₂ ≈ -0.3 to -1.0 (linear confidence term)
- a₃ ≈ -0.3 to -1.0 (log term, approaching BCE)
- a₁ ≈ 0 (constant offset, usually minimal)

The test run gave: `a1=0.0037, a2=-0.3068, a3=-0.2764`

With more checkpoints and diversity, we may see:
- Stronger coefficients (closer to -1)
- More stable convergence
- Better reduction of oracle-naive gap

## Success Metrics

1. **Naive checkpoint improvement:**
   - Without learned loss: 1.750 BCE (from test)
   - Target: < 1.0 BCE (50%+ improvement)
   - Stretch goal: < 0.5 BCE (approaching oracle)

2. **Oracle checkpoint stability:**
   - Should remain around 0.3-0.4 BCE
   - Learned loss shouldn't hurt already-good checkpoints

3. **Gap reduction:**
   - Current: 421% gap (naive 5.2x worse than oracle)
   - Target: < 200% gap (naive 2x worse than oracle)
   - Stretch: < 100% gap (comparable performance)

4. **Convergence stability:**
   - Meta-learning should converge smoothly
   - No catastrophic forgetting or oscillation
   - Steady improvement over 500 iterations

## Computational Requirements

Creating 560 checkpoints:
- 40 base tasks (4 difficulties × 2 stds × 5 seeds)
- 2 training methods per task
- 7 checkpoints per method
- Max 200 epochs training per checkpoint

Estimated time:
- Fast tasks (mean_sep=3.0): ~5-10 sec to epoch 200
- Hard tasks (mean_sep=1.5): ~10-20 sec to epoch 200
- Total checkpoint creation: ~30-60 minutes
- Meta-training (500 iterations): ~30-60 minutes
- **Total: 1-2 hours**

## File Structure

```
toy_meta_learning/
├── config/
│   ├── toy_gaussian_meta_extended.yaml  # Full 560-checkpoint config
│   └── toy_gaussian_meta_test.yaml      # Test 48-checkpoint config
├── train_extended_pool.py               # Main training script
├── train_test_pool.py                   # Test script
└── EXTENDED_POOL_DESIGN.md              # This document
```

## Next Steps After Training

1. Analyze learned loss parameters
2. Compare to previous symmetric/asymmetric results
3. Evaluate on each checkpoint category:
   - Oracle early/mid/late
   - Naive early/mid/late
4. Test if we should use asymmetric loss (separate f_p, f_u)
5. Compare to baseline PU methods (if applicable)
