# Hierarchical PU Loss Implementation

## Overview

Successfully implemented a **hierarchical 27-parameter PU loss** that generalizes multiple known PU methods (BCE, VPU, PUDRa, uPU) through meta-learning.

## Loss Structure

```
L_PU = f_p1(E_P[f_p2(f_p3(p))]) + f_u1(E_U[f_u2(f_u3(p))]) + f_a1(E_A[f_a2(f_a3(p))]) + λ·L1_reg
```

Where:
- Each `f` is the simple basis: `f(x) = a1 + a2·x + a3·log(x)`
- `E_P` = mean over labeled positives
- `E_U` = mean over unlabeled samples
- `E_A` = mean over ALL samples (both groups combined)
- Functions are nested: **innermost → middle → mean → outermost**
- L1 regularization encourages sparsity

## Parameter Organization (27 total)

### Labeled Positives Group (9 params)
- **f_p1 (outermost)**: Applied to mean value → `a1_p1, a2_p1, a3_p1`
- **f_p2 (middle)**: Applied to each transformed sample → `a1_p2, a2_p2, a3_p2`
- **f_p3 (innermost)**: Applied to each probability → `a1_p3, a2_p3, a3_p3`

### Unlabeled Group (9 params)
- **f_u1, f_u2, f_u3**: Same structure for unlabeled samples

### All Samples Group (9 params)
- **f_a1, f_a2, f_a3**: Same structure for all samples combined

## Key Features

### 1. Flexible Initialization Modes

**'random'**: Standard random initialization
```python
All 27 parameters ~ N(0, 0.01²)
```

**'bce_equivalent'**: Approximates standard BCE
```python
# Positives: identity → -log → identity
f_p3(p) = p              # a1=0, a2=1, a3=0
f_p2(p) = -log(p)        # a1=0, a2=0, a3=-1
f_p1(x) = x              # a1=0, a2=1, a3=0

# Unlabeled: reverse → -log → identity
f_u3(p) = 1-p            # a1=1, a2=-1, a3=0
f_u2(1-p) = -log(1-p)    # a1=0, a2=0, a3=-1
f_u1(x) = x              # a1=0, a2=1, a3=0

# All samples: zeros (no contribution)
```

**'zeros'**: All parameters = 0

**'identity_chain'**: All functions = identity (a1=0, a2=1, a3=0)

**'diverse_init'**: Mixed initialization for robustness

### 2. Numerical Stability

**Three-tier clamping strategy:**
1. **Innermost functions** (on probabilities): Clamp to [eps, 1-eps] where eps=1e-7
2. **Middle/outer functions** (on intermediate values): Wider range [eps, 1e6]
3. **Output clamping**: Prevent overflow with limits (100 for probs, 1e3 for intermediate)

### 3. L1 Regularization

```python
L1_penalty = λ · sum(|all 27 parameters|)
```

Encourages sparsity - meta-learning can discover which parameters are unnecessary and drive them to zero.

## How It Generalizes Known Methods

### BCE (Binary Cross-Entropy)
```
Set: f_p3=identity, f_p2=-log, f_p1=identity
     f_u3=reverse, f_u2=-log, f_u1=identity
     f_a*=0
Result: -E_P[log(p)] - E_U[log(1-p)]
```

### Label Reversal
The innermost function can learn to flip labels:
```python
f_p3(p) = 1-p  →  a1=1, a2=-1, a3=0
```
No need to explicitly specify label flipping in the loss structure!

### Novel Transformations
Meta-learning can discover:
- Non-linear aggregation before averaging (f_p3, f_p2)
- Non-linear combination of group averages (f_p1, f_u1, f_a1)
- Optimal balance between positive, unlabeled, and all-samples terms

## Usage

### Basic Usage

```python
from loss.hierarchical_pu_loss import HierarchicalPULoss

# Create loss
loss_fn = HierarchicalPULoss(
    init_mode='random',      # or 'bce_equivalent', 'zeros', etc.
    init_scale=0.01,         # for random init
    l1_lambda=0.0,           # L1 regularization coefficient
    eps=1e-7,                # numerical stability
)

# Forward pass
outputs = model(x)  # logits
labels = y_pu       # 1 for labeled positive, -1 for unlabeled
loss = loss_fn(outputs, labels, mode='pu')
```

### In Meta-Learning

```python
# Replace AsymmetricPULoss with HierarchicalPULoss
# OLD:
# loss_fn = AsymmetricPULoss(init_mode='random', init_scale=0.01).to(device)

# NEW:
loss_fn = HierarchicalPULoss(
    init_mode='random',
    init_scale=0.01,
    l1_lambda=1e-4,  # optional: adds sparsity
).to(device)

# Rest of meta-learning code stays the same!
meta_optimizer = torch.optim.Adam(loss_fn.parameters(), lr=0.001)
```

## Implementation Details

### File Location
`/Users/akellner/MyDir/Code/Other/PU-Bench/toy_meta_learning/loss/hierarchical_pu_loss.py`

### Key Methods

1. **apply_basis(x, a1, a2, a3, is_probability=True)**
   - Applies basis function with context-aware clamping
   - Different ranges for probabilities vs intermediate values

2. **apply_nested_group(p_group, params_inner, params_middle, params_outer)**
   - Computes f_1(mean(f_2(f_3(p))))
   - Handles the full nested transformation pipeline

3. **forward(outputs, labels, mode='pu')**
   - Main loss computation
   - Masks samples into three groups (positive, unlabeled, all)
   - Applies nested transformations to each group
   - Sums results and adds L1 regularization

4. **compute_l1_regularization()**
   - Returns λ · sum(|parameters|)

5. **get_parameters()**
   - Returns all 27 parameters as a single [27] tensor

## Verification Tests

All basic tests pass:
✓ Loss creation and initialization
✓ Forward pass (no NaN/Inf)
✓ Backward pass (all 27 parameters receive gradients)
✓ BCE initialization correctly approximates BCE
✓ Meta-training successfully updates parameters

## Example Output

```
HierarchicalPULoss(
  Positive Group:
    f_p1 (outer): a1=0.0121, a2=0.0145, a3=-0.0014
    f_p2 (mid):   a1=0.0057, a2=-0.0083, a3=-0.0095
    f_p3 (inner): a1=-0.0016, a2=-0.0016, a3=0.0165
  Unlabeled Group:
    f_u1 (outer): a1=-0.0054, a2=-0.0037, a3=-0.0011
    f_u2 (mid):   a1=0.0033, a2=0.0106, a3=-0.0112
    f_u3 (inner): a1=0.0067, a2=-0.0046, a3=-0.0184
  All Samples Group:
    f_a1 (outer): a1=-0.0137, a2=0.0053, a3=0.0059
    f_a2 (mid):   a1=-0.0068, a2=-0.0029, a3=0.0091
    f_a3 (inner): a1=-0.0215, a2=0.0050, a3=-0.0043
  L1_lambda: 0.0
)
```

## Next Steps

1. **Run full meta-training**: Complete 300-500 iterations on extended checkpoint pool
2. **Compare to baselines**: Test against 6-param asymmetric loss and pure BCE
3. **Analyze learned structure**: Which functions matter most? Does sparsity emerge with L1?
4. **Test on real datasets**: Apply to actual PU learning benchmarks
5. **Experiment with L1_lambda**: Try values like 1e-4, 1e-3 to encourage sparsity

## Performance Considerations

- **Computational complexity**: O(n) where n is batch size (same as asymmetric loss)
- **Memory**: 27 parameters = 108 bytes (negligible)
- **Training speed**: Comparable to 6-parameter asymmetric loss
- **Gradient flow**: All 27 parameters receive clean gradients through 3 levels of nesting

## Advantages Over Simpler Losses

1. **Expressiveness**: Can represent a much wider family of loss functions
2. **Label flexibility**: Can learn whether to use p, 1-p, or other transformations
3. **Three-way decomposition**: Separate handling of positives, unlabeled, and all samples
4. **Discovers structure**: Meta-learning finds which transformations are beneficial
5. **Sparsity option**: L1 regularization can identify unnecessary complexity
