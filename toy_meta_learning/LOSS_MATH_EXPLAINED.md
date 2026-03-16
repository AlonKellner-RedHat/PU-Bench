# How the Basis Function Forms a Loss (and Matches BCE)

## The Basis Function

```python
f(x) = a₁ + a₂·x + a₃·log(x)
```

This is NOT the loss itself - it's a **building block** applied to probabilities.

## How It Forms the PN Loss

For Positive-Negative (PN) learning, the loss is constructed as:

```
L_PN = E_P[f(p)] + E_N[f(1-p)]
```

Where:
- `p = sigmoid(logits)` = predicted probability of being positive
- `E_P[·]` = expectation over labeled positive samples
- `E_N[·]` = expectation over labeled negative samples
- `f(p)` is applied to positives
- `f(1-p)` is applied to negatives (flip probability)

## Expanding the Math

Let's expand what happens with our basis function:

```
L_PN = E_P[a₁ + a₂·p + a₃·log(p)] + E_N[a₁ + a₂·(1-p) + a₃·log(1-p)]
```

Breaking this down:
```
L_PN = a₁ + E_P[a₂·p + a₃·log(p)]  ← Applied to positive examples
     + a₁ + E_N[a₂·(1-p) + a₃·log(1-p)]  ← Applied to negative examples

     = 2a₁ + a₂·E_P[p] + a₃·E_P[log(p)]
           + a₂·E_N[1-p] + a₃·E_N[log(1-p)]

     = 2a₁ + a₂·(E_P[p] + E_N[1-p]) + a₃·(E_P[log(p)] + E_N[log(1-p)])
```

## Binary Cross-Entropy (BCE)

Standard BCE is:
```
L_BCE = E_P[-log(p)] + E_N[-log(1-p)]
      = -E_P[log(p)] - E_N[log(1-p)]
```

## Making Them Equivalent

To match BCE, we need:

```
L_PN = 2a₁ + a₂·(E_P[p] + E_N[1-p]) + a₃·(E_P[log(p)] + E_N[log(1-p)])
L_BCE =                                  -  (E_P[log(p)] + E_N[log(1-p)])
```

**Setting the parameters**:
- `a₁ = 0` (constant doesn't affect gradients)
- `a₂ = 0` (no linear term)
- `a₃ = -1` (matches BCE)

**Result**:
```
L_PN = 0 + 0 + (-1)·(E_P[log(p)] + E_N[log(1-p)])
     = -E_P[log(p)] - E_N[log(1-p)]
     = L_BCE  ✓
```

## What Meta-Learning Learned

After 1000 iterations, the parameters converged to:
```
a₁ =  0.02  ≈ 0
a₂ = -0.95  ≈ -1  ← Key difference!
a₃ = -0.97  ≈ -1
```

So the learned loss is:
```
L_learned = 2(0.02) + (-0.95)·(E_P[p] + E_N[1-p]) + (-0.97)·(E_P[log(p)] + E_N[log(1-p)])
          ≈ 0 - E_P[p] - E_N[1-p] - E_P[log(p)] - E_N[log(1-p)]
          = -(E_P[p + log(p)] + E_N[(1-p) + log(1-p)])
```

## Visualizing the Difference

Let's see what each term contributes:

### Pure BCE (a₂=0, a₃=-1)

For a positive example with `p = 0.9`:
```
f(p) = -log(0.9) = 0.105
```

For a negative example with `p = 0.1`:
```
f(1-p) = f(0.9) = -log(0.9) = 0.105
```

### Learned Loss (a₂=-1, a₃=-1)

For a positive example with `p = 0.9`:
```
f(p) = -0.9 - log(0.9) = -0.9 - 0.105 = -0.995
```

For a negative example with `p = 0.1`:
```
f(1-p) = f(0.9) = -0.9 - log(0.9) = -0.995
```

**Notice**: The linear term adds a **confidence penalty**!

## The Effect of Each Term

| Term | What it does | BCE | Learned |
|------|--------------|-----|---------|
| `a₁` (constant) | Baseline offset | 0 | 0 |
| `a₂·p` (linear) | Penalize confidence | 0 | **-p** |
| `a₃·log(p)` (log) | Core BCE signal | **-log(p)** | **-log(p)** |

## Complete Loss Comparison Table

| Prediction p | Pure BCE: -log(p) | Learned: -p - log(p) | Difference |
|--------------|-------------------|----------------------|------------|
| p = 0.1 | 2.30 | **2.40** | +0.1 (penalty) |
| p = 0.3 | 1.20 | **1.50** | +0.3 (penalty) |
| p = 0.5 | 0.69 | **1.19** | +0.5 (penalty) |
| p = 0.7 | 0.36 | **1.06** | +0.7 (penalty) |
| p = 0.9 | 0.11 | **1.01** | +0.9 (penalty) |
| p → 1.0 | 0.00 | **1.00** | +1.0 (maximum penalty) |

**Key insight**: As confidence increases, the learned loss adds increasing penalty!

## Why This Helps

### 1. **Prevents Overconfidence**

Pure BCE allows `p → 1` with zero loss. The learned loss penalizes this:
```
BCE(p=0.99) = 0.01        ← Nearly zero
Learned(p=0.99) = 0.99    ← Still significant penalty
```

### 2. **Better Gradients**

BCE gradients vanish for confident predictions:
```
∂BCE/∂logit = p - y  → 0  when p ≈ y
```

Learned loss maintains gradients:
```
∂L_learned/∂logit = (a₂ + a₃/p)·p·(1-p) - y
                   ≈ (-1 - 1/p)·p·(1-p) - y
```

Even when `p ≈ 1`, the `-1` term keeps gradients non-zero!

### 3. **Implicit Regularization**

The linear penalty acts like L2 regularization on probabilities:
- Encourages `p ∈ (0.7, 0.9)` instead of `p → 1`
- Better calibration
- Less overfitting to support set

## Summary

**The basis function `f(x) = a₁ + a₂·x + a₃·log(x)` forms a loss by:**

1. Applying it to positive examples: `f(p)`
2. Applying it to flipped negatives: `f(1-p)`
3. Averaging both: `L = E_P[f(p)] + E_N[f(1-p)]`

**BCE equivalence**: Set `a₁=0, a₂=0, a₃=-1`

**Meta-learned improvement**: Add linear term `a₂=-1` for confidence regularization!

This simple 3-parameter basis can express:
- Pure BCE (information-theoretic optimal)
- Confidence-regularized variants (optimization-optimal for few-shot)
- And anything in between!
