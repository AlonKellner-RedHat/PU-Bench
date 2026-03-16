# Interpretation of Learned Hierarchical PU Loss Parameters

## Learned Parameters

```
HierarchicalPULoss(
  Positive Group:
    f_p1 (outer): a1=0.0159, a2=0.0173, a3=0.0027
    f_p2 (mid):   a1=-0.0067, a2=-0.0190, a3=0.0070
    f_p3 (inner): a1=0.0034, a2=-0.0108, a3=-0.0057
  Unlabeled Group:
    f_u1 (outer): a1=0.0048, a2=-0.0134, a3=-0.0078
    f_u2 (mid):   a1=-0.0226, a2=-0.0092, a3=0.0008
    f_u3 (inner): a1=-0.0108, a2=-0.0034, a3=0.0037
  All Samples Group:
    f_a1 (outer): a1=0.0088, a2=-0.0136, a3=-0.0061
    f_a2 (mid):   a1=-0.0063, a2=0.0222, a3=0.0137
    f_a3 (inner): a1=-0.0152, a2=0.0035, a3=-0.0172
)
```

## Basis Function Reminder

Each function has form: **f(x) = a1 + a2·x + a3·log(x)**

Where:
- `a1` = constant offset
- `a2` = linear coefficient
- `a3` = logarithmic coefficient

## Loss Structure

```
L_PU = f_p1(E_P[f_p2(f_p3(p))])
     + f_u1(E_U[f_u2(f_u3(p))])
     + f_a1(E_A[f_a2(f_a3(p))])
```

Flow for each group:
1. **Innermost (f_3)**: Transforms each sample's probability `p`
2. **Middle (f_2)**: Transforms each intermediate value
3. **Mean (E)**: Averages across group
4. **Outermost (f_1)**: Transforms the mean

---

## Group 1: Labeled Positives (Clean Labels)

### f_p3 (innermost - on each probability)
```python
f_p3(p) = 0.0034 - 0.0108·p - 0.0057·log(p)
```

**Interpretation:**
- Small negative linear term (-0.0108): Slightly penalizes high predictions
- Small negative log term (-0.0057): Mild logarithmic penalty
- **Net effect**: Very weak transformation, nearly identity
- **Meaning**: Labeled positives are trustworthy → don't need aggressive transformation

### f_p2 (middle - on transformed values)
```python
f_p2(z) = -0.0067 - 0.0190·z + 0.0070·log(z)
```

**Interpretation:**
- Negative linear term (-0.0190): Penalizes larger intermediate values
- Positive log term (+0.0070): Slight concavity
- **Net effect**: Compresses the range before averaging

### f_p1 (outer - on group mean)
```python
f_p1(mean) = 0.0159 + 0.0173·mean + 0.0027·log(mean)
```

**Interpretation:**
- Positive offset (+0.0159): Adds constant penalty
- Positive linear term (+0.0173): Encourages higher group average
- Small positive log term: Slight convexity
- **Net effect**: Nearly linear, mildly increasing

**Overall Positive Group Behavior:**
- Weak transformations throughout → trust the labeled positives
- Slight penalty that increases with group mean
- Approximates: **E_P[small_constant - small_penalty(p)]**

---

## Group 2: Unlabeled Samples (Noisy Mixture)

### f_u3 (innermost - on each probability)
```python
f_u3(p) = -0.0108 - 0.0034·p + 0.0037·log(p)
```

**Interpretation:**
- Negative offset (-0.0108): Baseline penalty
- Small negative linear term (-0.0034): Weak penalty on high predictions
- Small positive log term (+0.0037): Encourages avoiding very low predictions
- **Net effect**: Minimal transformation, slight push away from extremes

### f_u2 (middle - on transformed values)
```python
f_u2(z) = -0.0226 - 0.0092·z + 0.0008·log(z)
```

**Interpretation:**
- Large negative offset (-0.0226): **Largest single coefficient!**
- Negative linear term (-0.0092): Penalty on positive values
- Tiny positive log term: Nearly negligible
- **Net effect**: Strong negative baseline → **reduces contribution of unlabeled**

### f_u1 (outer - on group mean)
```python
f_u1(mean) = 0.0048 - 0.0134·mean - 0.0078·log(mean)
```

**Interpretation:**
- Small positive offset
- Negative linear term (-0.0134): Penalizes higher group average
- Negative log term (-0.0078): Additional concave penalty
- **Net effect**: Decreasing function → **lower is better**

**Overall Unlabeled Group Behavior:**
- Strong negative offset in f_u2 → **reduces overall contribution**
- f_u1 decreases with mean → **penalizes high unlabeled predictions**
- Key insight: **Unlabeled term acts as REGULARIZER, not main objective**
- Approximates: **-small_constant - penalty(E_U[p])**

---

## Group 3: All Samples Combined

### f_a3 (innermost - on each probability)
```python
f_a3(p) = -0.0152 + 0.0035·p - 0.0172·log(p)
```

**Interpretation:**
- Negative offset (-0.0152)
- Small positive linear term (+0.0035)
- Negative log term (-0.0172): **Second largest log coefficient**
- **Net effect**: Strong logarithmic penalty → penalizes low predictions

### f_a2 (middle - on transformed values)
```python
f_a2(z) = -0.0063 + 0.0222·z + 0.0137·log(z)
```

**Interpretation:**
- Positive linear term (+0.0222): **Largest positive linear coefficient**
- Positive log term (+0.0137): **Largest positive log coefficient**
- **Net effect**: Amplifies positive intermediate values

### f_a1 (outer - on group mean)
```python
f_a1(mean) = 0.0088 - 0.0136·mean - 0.0061·log(mean)
```

**Interpretation:**
- Negative linear and log terms → decreasing function
- **Net effect**: Penalizes higher overall average

**Overall All-Samples Group Behavior:**
- f_a3 applies strong log penalty to raw probabilities
- f_a2 amplifies intermediate values with positive coefficients
- f_a1 then penalizes the result
- **Meaning**: Complex non-linear transformation on global distribution
- Acts as **global consistency regularizer**

---

## Key Insights

### 1. Asymmetry Discovered
The loss learned **different strategies for different groups**:
- **Positives**: Trust them, use directly with minimal transformation
- **Unlabeled**: Strong negative offset → reduce their influence
- **All samples**: Complex non-linear transformation for global consistency

### 2. Magnitude Analysis

**Largest coefficients:**
1. `a1_u2 = -0.0226` (unlabeled middle offset) → **Most important!**
2. `a2_a2 = +0.0222` (all-samples middle linear)
3. `a2_p2 = -0.0190` (positive middle linear)
4. `a3_a3 = -0.0172` (all-samples inner log)

**Interpretation**:
- The middle layer (f_2) has the strongest effects
- Unlabeled offset is most negative → **downweights unlabeled contribution**
- All-samples middle layer has strong positive amplification

### 3. Comparison to Standard PU Losses

**Standard BCE on PU data:**
```
L = -E_P[log(p)] - E_U[log(1-p)]
```
Treats labeled positives and unlabeled equally (both get -log).

**Learned loss:**
```
L ≈ E_P[small_penalty(p)] - large_constant + E_U[small_transform(p)] + E_A[complex_transform(p)]
```

**Key differences:**
1. **Unlabeled downweighting**: Large negative offset (-0.0226) reduces unlabeled influence
2. **No explicit label flipping**: Doesn't convert to (1-p), uses p directly
3. **Global term**: All-samples group adds consistency constraint
4. **Weak transformations**: Most coefficients small → regularized solution

### 4. Why It Works Better Than Naive

**Naive BCE problem:**
- Treats all unlabeled as negative → **heavily biased** when 70% are hidden positives
- Loss: `-log(p)` for positives, `-log(1-p)` for unlabeled
- Equal weighting despite unequal trust

**Learned solution:**
- Recognizes unlabeled are unreliable → **downweights with negative offset**
- Doesn't flip labels naively → uses `p` directly with small penalties
- Global term provides consistency across whole dataset
- **Result**: 10.5% better than naive!

### 5. Parameter Movement Analysis

Comparing initial (random) to final:

**Initial example:**
```
f_p1: a1=0.0218, a2=-0.0081, a3=-0.0042
f_u2: a1=-0.0102, a2=0.0193, a3=0.0054
```

**Final:**
```
f_p1: a1=0.0159, a2=0.0173, a3=0.0027
f_u2: a1=-0.0226, a2=-0.0092, a3=0.0008
```

**Changes:**
- f_u2.a1: -0.0102 → -0.0226 (more negative offset)
- f_p1.a2: -0.0081 → +0.0173 (sign flip!)
- **Magnitude**: Parameters moved ~2× from initialization
- **Problem**: Small movement → suggests learning rate too low or early stopping

---

## Functional Form Approximation

Simplifying to dominant terms:

```python
L_PU ≈ E_P[0.016 + 0.017·((-0.019)·p)] +     # Positives: small linear penalty
       E_U[(-0.023) + small_transform(p)] +   # Unlabeled: strong negative offset
       E_A[complex_nonlinear(p)]              # All: global consistency
```

Even simpler:
```
L_PU ≈ small_constant + E_P[linear(p)] + E_U[constant - small(p)] + E_A[nonlinear(p)]
```

**Dominant behavior:**
- Unlabeled contribute mainly through large negative constant (-0.023)
- Positives contribute through small linear transformation
- All-samples adds global non-linear regularization

---

## Limitations & Observations

### 1. Small Parameter Magnitudes
All coefficients are in range [-0.023, +0.022] → **very regularized**

**Possible reasons:**
- Meta learning rate too low (0.001)
- Only 500 iterations
- L1_lambda=0 but gradients naturally small
- Early convergence plateau (parameters stopped moving after iter 100)

### 2. Dominance of Constant Terms
Many `a1` (offset) terms are largest in their function → **additive bias dominates**

**Implication**: Loss is more about **weighting groups** than **transforming probabilities**

### 3. Limited Use of Logarithmic Terms
Most `a3` coefficients small → **not fully exploiting log basis**

**Interpretation**: Linear approximations sufficient for this problem

### 4. Middle Layer Importance
Largest coefficients are in middle layer (f_2) → **most learning happens between samples and mean**

**Design insight**: Three-level hierarchy may be overkill; two levels might suffice

---

## Recommendations

### To Improve Learning

1. **Increase meta learning rate**: Try 0.01 instead of 0.001
   - Parameters barely moved (2× initialization)
   - Stuck in local minimum

2. **More meta-iterations**: 500 iterations may be insufficient
   - Parameters plateaued at iteration 100
   - Try 1000-2000 iterations

3. **More checkpoints**: 24 checkpoints → small training set
   - Use 48 or 560 checkpoint pool
   - More diversity → better generalization

4. **L1 regularization**: Try l1_lambda = 1e-4
   - Encourage sparsity → identify unnecessary parameters
   - May simplify to fewer key coefficients

5. **Different initialization**: Try 'bce_equivalent'
   - Start from known good solution
   - See how far meta-learning moves from BCE

### To Analyze Structure

1. **Ablation study**: Remove each group, measure impact
   - Does all-samples group help?
   - What if we only use positives + unlabeled?

2. **Simplify to 2-level**: Remove middle layer
   - Does f_2 add value or just complexity?

3. **Compare to 6-param asymmetric**:
   - Run same experiments with simpler asymmetric loss
   - See if 27 params worth the complexity

---

## Conclusion

The learned hierarchical loss discovered a **downweighting strategy** for handling noisy unlabeled data:

**Key mechanism**: Large negative offset in unlabeled middle function (-0.0226) reduces contribution from unreliable unlabeled samples, while maintaining trust in clean labeled positives.

This is more sophisticated than naive label flipping, and **10.5% better** empirically. However, the small parameter magnitudes suggest the optimization may have converged prematurely, and there may be room for further improvement with better hyperparameters.

The hierarchical structure provides flexibility, but most learning concentrated in middle layer → **simpler 2-level or 6-parameter asymmetric loss might achieve similar results with less complexity**.
