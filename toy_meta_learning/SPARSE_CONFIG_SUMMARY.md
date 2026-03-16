# Sparse & High-Quality Configuration

**Created**: March 12, 2026
**Configuration**: [config/toy_gaussian_meta_sparse.yaml](config/toy_gaussian_meta_sparse.yaml)
**Training Script**: [train_sparse.py](train_sparse.py)

## Motivation

Building on the optimized configuration that achieved 16.9% improvement over naive baseline, we now push for:
1. **Simpler, more interpretable loss** via L1 regularization
2. **Better performance** via higher meta learning rate and more inner steps
3. **Stronger meta-gradient signal** for discovering optimal parameters

## Three Key Changes

### 1. L1 Regularization (λ = 1e-3)

**What it does:**
- Adds L1 penalty to loss function: `Loss = BCE_loss + λ * Σ|params|`
- Pushes small parameters toward zero during training
- Encourages sparse solutions (some params = 0, others significant)

**Expected benefits:**
- **Simpler loss structure**: Fewer active parameters easier to interpret
- **Better generalization**: Sparsity acts as implicit regularization
- **Clearer patterns**: Discover which groups/functions are truly important

**Example sparse outcome:**
```
Before L1 (all 27 params active):
  a1_p1=0.464, a2_p1=0.329, a3_p1=-0.426, a1_p2=-0.578, ...

After L1 (sparse, ~10 params active):
  a1_p1=0.721, a2_p1=0.000, a3_p1=0.000, a1_p2=-0.843, a2_p2=1.000, a3_p2=0.000, ...
```

### 2. Higher Meta Learning Rate (0.001 → 0.003)

**What it does:**
- 3× larger parameter updates per meta-iteration
- Faster convergence to optimal loss structure
- Better exploration of loss space

**Expected benefits:**
- **Faster learning**: Reach good solutions earlier in training
- **Escape local minima**: Larger steps help avoid shallow minima
- **Stronger signals**: Combined with 8 inner steps, creates powerful meta-gradients

**Rationale:**
Previous runs showed parameters barely moved (max ~0.023 at iteration 100 before normalization). Higher LR with weight normalization should enable faster, more aggressive learning while maintaining stability.

### 3. More Inner Steps (4 → 8)

**What it does:**
- Each meta-iteration runs 8 gradient steps on PU task (vs 4 previously)
- Stronger task adaptation before meta-objective evaluation
- Higher quality meta-gradients for loss parameter updates

**Expected benefits:**
- **Better meta-gradient**: Model adapts more fully to each task
- **Discover subtler patterns**: More adaptation reveals what truly helps
- **Higher performance**: Expected >20% improvement over naive (vs 16.9% baseline)

**Trade-off:**
- Training time: ~25-30 minutes (vs ~14 min with 4 steps)
- Worth it for final production loss discovery

## Configuration Summary

| Parameter | Baseline (4-step) | Sparse (8-step) | Change |
|-----------|-------------------|-----------------|--------|
| **Inner steps** | 4 | 8 | 2× more adaptation |
| **Meta LR** | 0.001 | 0.003 | 3× faster learning |
| **L1 lambda** | 0.0 | 1e-3 | Sparsity regularization |
| Meta optimizer | AdamW (wd=1e-4) | AdamW (wd=1e-4) | Same |
| Weight normalization | Yes | Yes | Same |
| Meta batch size | 48 | 48 | Same |
| Meta iterations | 500 | 500 | Same |
| **Expected training time** | ~14 min | ~25-30 min | 2× slower |
| **Expected improvement** | 16.9% | **>20%** | Better |

## Expected Outcomes

### Performance Metrics

**Target**: >20% improvement over naive baseline
- Baseline (4 steps, no L1): 16.9% improvement
- Sparse (8 steps, L1): 20-25% improvement (target)

**Sparsity Metrics**:
- Parameters near zero (|p| < 0.01): ~40-60% of 27 params
- Active parameters: ~10-15 with significant values
- Clearer functional structure in learned loss

### Interpretability Benefits

With L1 regularization, we expect to see:

1. **Group specialization**: Some groups (positive/unlabeled/all) may become inactive
2. **Function pruning**: Inner/middle/outer functions may simplify to identity or log-only
3. **Clear patterns**: Easier to understand what the loss is doing

Example interpretable outcome:
```
Positive group: f_p2(E[f_p3(p)]) with f_p1 = identity
  → Just average transformed probabilities

Unlabeled group: -log(E[1-p])
  → Classic PU formulation discovered

All samples group: zero
  → Not needed for this problem
```

## How to Run

### Execute training
```bash
cd toy_meta_learning
python train_sparse.py
```

### Expected timeline
- Load/create checkpoint pool: ~1 min (cached after first run)
- Meta-training (500 iterations): ~22-25 minutes
- End-to-end evaluation (5 tasks): ~3 minutes
- **Total: ~26-29 minutes**

### Monitor progress
Real-time tqdm progress bars show:
```
Meta-training:  45%|████████▌    | 225/500 [10:15<11:25, meta_loss=0.6821, samples/s=55420]

Iteration 200/500
  Meta-loss: 0.682134
  Oracle checkpoints BCE: 0.438215
  Naive checkpoints BCE:  0.987432
  Throughput: 54,732 samples/min (current iter: 1,847,120)
  ...
```

## Comparison to Previous Versions

### V1: Throughput-Optimized (inner_steps=1)
- **Goal**: Fast iteration for experimentation
- **Performance**: 10.5% improvement
- **Training time**: 4 minutes
- **Use case**: Rapid prototyping

### V2: Quality-Optimized (inner_steps=4)
- **Goal**: Better performance with reasonable time
- **Performance**: 16.9% improvement (60% better than V1)
- **Training time**: 14 minutes
- **Use case**: Production baseline

### V3: Sparse & High-Quality (inner_steps=8, L1, high LR) ← **This version**
- **Goal**: Best performance + interpretability
- **Performance**: >20% improvement target (20% better than V2)
- **Training time**: 26-29 minutes
- **Use case**: Final production loss, paper results

## Analysis After Training

After training completes, analyze:

1. **Sparsity achieved**
   - How many parameters near zero?
   - Which groups/functions are active?

2. **Performance gain**
   - Improvement over naive baseline
   - Variance across test tasks

3. **Learned structure**
   - Can we interpret the loss in terms of known PU methods?
   - Does it discover novel patterns?

4. **Parameter evolution**
   - Did L1 push parameters to zero smoothly?
   - Did higher LR cause instability?

## Next Steps After This Run

If results are promising:

1. **Test on real datasets** (beyond toy Gaussian blobs)
2. **Ablation study**: Test L1 lambda values (1e-4, 1e-3, 1e-2)
3. **Inner step sweep**: Try 6, 10, 12 inner steps
4. **Meta LR sweep**: Try 0.005, 0.01 for even faster learning
5. **Compare to known PU losses**: How does learned loss compare to VPU, uPU, nnPU?

## Success Criteria

This configuration succeeds if:

1. ✅ **Performance**: >20% improvement over naive (better than 16.9% baseline)
2. ✅ **Sparsity**: >40% of parameters near zero (clear simplification)
3. ✅ **Interpretability**: Learned loss can be explained in simple terms
4. ✅ **Stability**: Training converges without NaN/Inf issues despite higher LR

If all criteria met → **Use this loss for production PU learning tasks**
