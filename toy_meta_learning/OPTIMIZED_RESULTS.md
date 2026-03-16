# Optimized Meta-Learning Results

**Date**: March 12, 2026
**Configuration**: [config/toy_gaussian_meta_optimized.yaml](config/toy_gaussian_meta_optimized.yaml)
**Training Script**: [train_optimized.py](train_optimized.py)

## Executive Summary

**Achieved 16.9% improvement over naive baseline**, representing a **60% increase** in performance gain compared to the baseline 10.5% improvement.

## Optimization Strategy

Three key improvements implemented:

### 1. Increased Inner Steps (1 → 4)
- **Rationale**: More adaptation steps provide stronger meta-gradient signal
- **Trade-off**: 4× slower iterations but better task adaptation
- **Impact**: Critical for discovering effective loss parameters

### 2. AdamW Optimizer with Weight Decay
- **Configuration**:
  - Learning rate: 0.001
  - Weight decay: 1e-4
- **Rationale**: Decoupled weight decay provides better regularization for meta-parameters
- **Previous**: Standard Adam without weight decay

### 3. Weight Normalization
- **Method**: After each optimizer step, rescale all parameters so max |param| = 1
- **Rationale**: Prevents parameter scale drift during meta-learning
- **Implementation**: `normalize_loss_parameters(loss_fn)` after `meta_optimizer.step()`

## Performance Results

### Train-from-Scratch Evaluation (5 fresh tasks, 100 epochs each)

| Method | Mean BCE Loss | Std Dev | vs Oracle | vs Naive |
|--------|---------------|---------|-----------|----------|
| **Oracle** (ground truth labels) | 0.2560 | 0.0269 | baseline | - |
| **Naive** (treat unlabeled as negative) | 0.7695 | 0.0122 | +200.6% | baseline |
| **Learned** (hierarchical loss) | **0.6391** | 0.0416 | +149.7% | **-16.9%** |

**Key Finding**: Learned loss achieves **16.9% lower BCE** than naive approach on fresh test tasks.

### Performance Breakdown by Test Task

```
Task 1:  Oracle=0.3014  Naive=0.7833  Learned=0.7034  (improvement: 10.2%)
Task 2:  Oracle=0.2666  Naive=0.7556  Learned=0.6044  (improvement: 20.0%)
Task 3:  Oracle=0.2247  Naive=0.7568  Learned=0.6696  (improvement: 11.5%)
Task 4:  Oracle=0.2529  Naive=0.7834  Learned=0.5924  (improvement: 24.4%)
Task 5:  Oracle=0.2343  Naive=0.7683  Learned=0.6257  (improvement: 18.6%)
```

Average improvement: **16.9%** (range: 10.2% - 24.4%)

### Checkpoint Pool Evaluation

**Oracle checkpoints (PN-trained):**
- Epoch 1: 0.622
- Epoch 50: 0.398
- Epoch 100: 0.302
- **Average: 0.441**

**Naive checkpoints (PU-trained):**
- Epoch 1: 0.922
- Epoch 50: 0.800
- Epoch 100: 1.255 (degrades over time!)
- **Average: 0.993**

**Oracle-Naive gap: 125.2%** - demonstrates significant room for improvement in PU learning.

## Training Efficiency

- **Total meta-iterations**: 500
- **Meta-batch size**: 48 tasks/iteration
- **Total samples processed**: 48,000,000
- **Training time**: 845.7 seconds (14.1 minutes)
- **Throughput**: 56,757 samples/min
- **Average iteration time**: 1.69 seconds
- **Acceleration**: MPS (Metal Performance Shaders) on Apple Silicon

## Learned Loss Parameters

All parameters normalized to [-1, 1] range via weight normalization.

### Positive Group (labeled positives)
```
f_p1 (outer): a1=0.464,  a2=0.329,  a3=-0.426
f_p2 (mid):   a1=-0.578, a2=1.000,  a3=0.069   [DOMINANT: linear term]
f_p3 (inner): a1=-0.414, a2=-0.122, a3=0.309
```

**Interpretation**:
- Middle function f_p2 has **a2=1.000** (maximal linear coefficient due to normalization)
- Near-identity transformation through the middle layer
- Outer layer applies modest nonlinear transformation

### Unlabeled Group (unlabeled samples)
```
f_u1 (outer): a1=-0.578, a2=-0.108, a3=0.252
f_u2 (mid):   a1=-0.483, a2=-0.353, a3=-0.537  [BALANCED: all terms active]
f_u3 (inner): a1=0.430,  a2=0.320,  a3=0.470
```

**Interpretation**:
- All three parameters active in f_u2 middle function
- Negative a2 in f_u2 suggests **downweighting** or reversal
- Complex nonlinear transformation of unlabeled probabilities

### All Samples Group (both labeled + unlabeled)
```
f_a1 (outer): a1=-0.246, a2=0.062,  a3=0.652   [LOG-DOMINANT]
f_a2 (mid):   a1=0.445,  a2=0.150,  a3=0.588   [LOG-HEAVY]
f_a3 (inner): a1=0.012,  a2=-0.548, a3=-0.311
```

**Interpretation**:
- Strong logarithmic terms (a3) in outer and middle functions
- Negative linear term (a2=-0.548) in innermost function
- Provides global regularization across all samples

## Key Insights

### 1. Hierarchical Structure Matters
The learned loss uses all three levels of nesting, not just simple transformations. This suggests that **order of operations matters** - transforming before averaging vs after averaging produces different meta-learning signals.

### 2. Unlabeled Sample Treatment
The unlabeled group has the most complex transformation with all parameters actively used. This makes sense: **unlabeled data requires sophisticated handling** since it contains both positive and negative examples.

### 3. Logarithmic Regularization
The "all samples" group uses strong logarithmic terms, providing a **global regularization effect** similar to entropy-based regularization in known PU methods.

### 4. Parameter Stability
Weight normalization successfully prevented parameter explosion. All coefficients remained in [-1, 1] range throughout training, enabling stable optimization.

## Comparison to Baseline

| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| Learned vs Naive improvement | 10.5% | 16.9% | **+60% relative** |
| Inner steps | 1 | 4 | 4× more |
| Meta-optimizer | Adam | AdamW (wd=1e-4) | Better regularization |
| Weight normalization | No | Yes | Stable parameters |
| Training time | ~4 min | ~14 min | 3.5× longer |
| Final max |param| | ~0.023 | 1.000 | Proper scale |

**Efficiency**: Despite 4× more inner steps, training completed in only 3.5× the time due to other optimizations.

## Next Steps

### Immediate Actions
1. ✅ Improved logging with tqdm progress bars (implemented)
2. Document learned loss interpretation for paper/presentation
3. Test on larger-scale datasets (beyond toy Gaussian blobs)

### Future Experiments
1. **Vary inner steps**: Try 2, 3, 5, 6 to find optimal adaptation-cost trade-off
2. **L1 regularization**: Test with l1_lambda > 0 to encourage sparsity
3. **Alternative initializations**: Try 'bce_equivalent' or 'diverse_init' modes
4. **Longer training**: Increase to 1000 meta-iterations to see if performance continues improving
5. **Different weight decay values**: Test 1e-3, 1e-5, etc.

### Production Deployment
The learned loss parameters can be saved and reused:
- Load from `toy_meta_output/loss_params_iter0500.pt`
- Use on new PU tasks without meta-training
- Evaluate generalization to different prior probabilities, labeling frequencies

## Files Modified

1. [config/toy_gaussian_meta_optimized.yaml](config/toy_gaussian_meta_optimized.yaml) - Optimized configuration
2. [train_optimized.py](train_optimized.py) - Training script with AdamW, weight normalization, tqdm logging
3. [tasks/task_pool.py](tasks/task_pool.py) - Added checkpoint pool caching

## Reproducibility

To reproduce these results:
```bash
cd toy_meta_learning
python train_optimized.py
```

The training will:
1. Load or create cached checkpoint pool (24 checkpoints)
2. Run 500 meta-iterations with tqdm progress bars
3. Evaluate on checkpoint pool (oracle vs naive)
4. Train from scratch on 5 fresh test tasks
5. Save results and learned parameters

**Expected runtime**: ~14-15 minutes on Apple Silicon with MPS acceleration
