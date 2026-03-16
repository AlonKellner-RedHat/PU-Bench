# Aligned Meta-Objective Design

## Problem Statement

**Previous stabilization attempt failed to learn:**
- Parameters frozen (gradient norm = 0.0)
- Performance oscillated (0.295-0.394 range)
- Average degradation (~0.330 vs 0.293 baseline)
- Root cause: Aggressive clipping prevented all learning

**Your question:** "How can we make sure that the meta-objective and gradients are aligned to final performance while still safe and stable?"

## Solution: Multi-Timescale Meta-Objective + Soft Regularization

### Key Changes

#### 1. Multi-Timescale Meta-Objective (Alignment)

**Problem:** Optimizing only for 3-step performance doesn't guarantee good 50-epoch performance

**Solution:** Weighted average across adaptation depths

```python
# OLD: Single-point evaluation
meta_loss = evaluate_at_3_steps(model, val_data)

# NEW: Multi-timescale evaluation
loss_3step = evaluate_at_3_steps(model, val_data)
loss_10step = evaluate_at_10_steps(model, val_data)
meta_loss = 0.3 * loss_3step + 0.7 * loss_10step  # Weight longer adaptation more
```

**Why this helps:**
- Aligns meta-objective with final performance (50 epochs)
- 10-step performance is better proxy for final convergence than 3-step
- Encourages loss functions that work across time scales

**Trade-off:**
- ~3x slower per iteration (need to train to both 3 and 10 steps)
- But much better alignment → worth it

#### 2. Soft Regularization (Stability with Learning)

**Problem:** Hard clipping (max_norm=0.1) freezes parameters completely

**Solution:** Soft penalty for parameter drift

```python
# Store initial parameters
params_init = loss_fn.get_parameters().clone().detach()

# During meta-training
params_current = loss_fn.get_parameters()
drift_penalty = lambda_drift * torch.norm(params_current - params_init)
total_loss = meta_loss + drift_penalty

# Normal clipping (not aggressive)
clip_grad_norm_(loss_fn.parameters(), max_norm=1.0)  # Was 0.1
```

**Why this helps:**
- **Soft constraint** (penalty increases with distance from init)
- Allows learning (gradients still flow)
- Keeps parameters near safe initialization
- Self-balancing: large drift → large penalty → pulls back

**Tuning:**
- `lambda_drift = 0.01`: Start here
- If unstable: increase to 0.05 or 0.1
- If too conservative: decrease to 0.005

#### 3. Trust Region (Safety Net)

**Problem:** Soft regularization might not prevent sudden jumps

**Solution:** Hard bound on parameter change per iteration

```python
params_before = loss_fn.get_parameters().clone()
# ... gradient step ...
params_after = loss_fn.get_parameters()

param_change = torch.norm(params_after - params_before)
if param_change > max_param_change:
    # Scale back to trust region boundary
    direction = (params_after - params_before) / param_change
    params_safe = params_before + max_param_change * direction
    # Set parameters to safe values
```

**Why this helps:**
- **Hard bound** on parameter change (safety net)
- Prevents sudden divergence (like iter 20 in unstabilized VPU)
- Only activates when soft regularization isn't enough

**Tuning:**
- `max_param_change = 0.05`: Start here
- Monitor activation rate
- If activating >50% of time: increase limit
- If never activating: can decrease for faster learning

#### 4. Deterministic Validation (Comparability)

**Problem:** Validation variance makes it hard to assess true learning

**Solution:** Fixed random seeds for each validation task

```python
def set_deterministic_seed(task_idx, model_type):
    seed = 50000 + task_idx * 100 + hash(model_type) % 100
    torch.manual_seed(seed)
    np.random.seed(seed)
    # ... set all random states ...
```

**Why this helps:**
- Same model initialization every validation
- Same training dynamics every validation
- Only difference is the loss function parameters
- **All validations directly comparable**

**Expected:**
- If params frozen → performance should be identical across iterations
- If params changing → performance change reflects actual learning

## Expected Outcomes

### If VPU Can Work with These Techniques

**Optimistic scenario:**
```
Iter   0: 0.293 BCE (baseline)
Iter  20: 0.290 BCE (small improvement, no divergence!)
Iter  40: 0.285 BCE (continuing to improve)
Iter 100: 0.270 BCE (significant improvement)
```

**Characteristics:**
- ✓ Stable (no divergence)
- ✓ Learning (parameters actually change)
- ✓ Improving (better than baseline)
- ✓ Deterministic validation (no oscillation)

**This would prove:** VPU structure can work for meta-learning with proper alignment and regularization

### If VPU Still Can't Learn

**Realistic scenario:**
```
Iter   0: 0.293 BCE (baseline)
Iter  20: 0.293 BCE (stable, minimal change)
Iter  40: 0.293 BCE (still at baseline)
Iter 100: 0.293 BCE (no improvement, but no degradation)
```

**Characteristics:**
- ✓ Stable (no divergence)
- ~ Learning (small parameter changes, but constrained)
- ~ Flat (can't improve beyond initialization)
- ✓ Deterministic (performance consistent)

**This would prove:** VPU structure is fundamentally incompatible with few-shot meta-learning, even with alignment and regularization

### Metrics to Monitor

**1. Meta-Loss Components**
```
Meta-loss (weighted):     0.55
  - 3-step component:     0.60  ← Quick adaptation
  - 10-step component:    0.52  ← Final performance proxy
```
- Should both decrease for true improvement
- If 3-step improves but 10-step worsens → misalignment (bad)

**2. Regularization Metrics**
```
Drift penalty:            0.003  ← Soft constraint
Total drift from init:    0.12   ← Cumulative change
Param change (current):   0.008  ← Per-iteration step
Trust region activations: 5/100  ← Safety net usage
```
- Drift should be non-zero (learning happening)
- But bounded (not exploding)
- Trust region should activate occasionally (< 20% of time)

**3. Gradient Metrics**
```
Gradient norm:            0.8    ← Should be non-zero
Param change (avg):       0.01   ← Should be consistent
```
- NOT 0.0 like before (that was frozen)
- Consistent across iterations (not spiking)

**4. Validation (Deterministic)**
```
Learned:      0.285 ← Should change smoothly
VPU baseline: 0.293 ← Should stay constant
```
- Learned should change monotonically (or stay flat)
- No oscillations (deterministic seeds prevent this)

## Comparison to Previous Approaches

| Approach | Meta-Objective | Stabilization | Learning? | Stable? |
|----------|---------------|---------------|-----------|---------|
| **Unstabilized VPU** | 3-step only | None | Yes (broken) | ❌ No |
| **Aggressive Stabilization** | 3-step only | Hard clip (0.1) | ❌ Frozen | ✓ Yes |
| **Aligned (New)** | 3-step + 10-step | Soft reg + trust | ✓ Expected | ✓ Expected |
| **PUDRa (Baseline)** | 3-step only | None needed | ✓ Yes | ✓ Yes |

## Success Criteria

### Minimum Success (Stability)
- ✓ No catastrophic divergence
- ✓ Performance within 5% of initialization
- ✓ Deterministic validation (no oscillation)
- Parameters change at least slightly (not frozen)

**If this succeeds:** VPU can at least be stable with proper techniques

### Full Success (Learning)
- ✓ Stable throughout training
- ✓ Consistent improvement (5%+ better than init)
- ✓ Beats or matches PUDRa final performance
- Parameters evolve meaningfully

**If this succeeds:** VPU is viable for meta-learning with aligned objective

### If Neither
- If unstable despite soft regularization
- Or if stable but completely frozen again

**Then:** VPU is fundamentally incompatible, stick with PUDRa

## Implementation Details

### Hyperparameters

```python
# Meta-objective
eval_steps = [3, 10]
weights = [0.3, 0.7]

# Regularization
drift_lambda = 0.01      # Soft constraint strength
max_param_change = 0.05  # Trust region bound

# Optimization
meta_lr = 1e-4           # Normal (not reduced)
max_grad_norm = 1.0      # Normal clipping (not 0.1)
inner_lr = 0.3           # Standard

# Training
meta_iterations = 100    # Test first
meta_batch_size = 8
```

### Tuning Guidelines

**If unstable (diverging):**
1. Increase `drift_lambda` to 0.05 or 0.1
2. Decrease `max_param_change` to 0.02
3. Reduce `meta_lr` to 5e-5

**If too conservative (not learning):**
1. Decrease `drift_lambda` to 0.005 or 0.001
2. Increase `max_param_change` to 0.1
3. Check trust region activation rate (should be < 20%)

**If 10-step performance worse than 3-step:**
1. Increase weight on 10-step (e.g., 0.2/0.8 instead of 0.3/0.7)
2. Consider adding 20-step evaluation
3. May indicate meta-objective still misaligned

## Expected Timeline

**With 8 tasks per meta-batch:**
- Each iteration: ~2x slower (evaluating at 3-step AND 10-step)
- 100 iterations: ~6-8 minutes
- 200 iterations: ~12-15 minutes

**Validation frequency:**
- Every 20 iterations (same as before)
- Deterministic → directly comparable

## Next Steps Based on Results

### If VPU Works Well
1. Run full 200 iterations
2. Compare to PUDRa (both should improve)
3. Document optimal hyperparameters
4. Consider hybrid approach (combine both)

### If VPU Stays Flat
1. Confirm it's truly stable (not oscillating)
2. Compare parameter changes (should be small but non-zero)
3. Accept that VPU initialization is optimal as-is
4. Use PUDRa for meta-learning instead

### If VPU Still Fails
1. Try even stronger drift penalty (λ = 0.1)
2. Try adaptive inner LR (lower for "all" group)
3. If still fails → VPU is incompatible
4. Stick with PUDRa recommendation

## Conclusion

This design addresses your key question: **alignment + stability**.

**Alignment:** Multi-timescale meta-objective ensures we optimize for final performance, not just quick adaptation.

**Stability:** Soft regularization + trust region allow learning while preventing divergence.

**Comparability:** Deterministic validation eliminates noise, making true learning visible.

**Expected outcome:** Either VPU works (great!), or we definitively prove PUDRa is better for this paradigm (also valuable!).
