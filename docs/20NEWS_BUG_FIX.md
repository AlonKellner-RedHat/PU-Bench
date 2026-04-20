# 20News Label-Embedding Alignment Bug Fix

## Summary

Fixed a critical bug in the 20News dataset loader that caused label-embedding misalignment for all random seeds except seed 42, resulting in near-random performance (AUC ~0.50 instead of ~0.95).

## The Bug

### Root Cause

**File:** `data/News20_PU.py`

The bug occurred because:

1. **Precomputed embeddings** were generated with `fetch_20newsgroups(shuffle=True, random_state=42)`
2. **Runtime data loading** used `fetch_20newsgroups(shuffle=True, random_state=random_seed)` with the experiment seed
3. When `random_seed != 42`, the labels and embeddings were shuffled in different orders, causing complete misalignment

### Evidence

**Before fix:**

| Seed | Oracle AUC | Oracle Acc | Status |
|------|-----------|-----------|--------|
| 42   | 0.9722    | 0.9128    | ✓ Good (aligned) |
| 456  | 0.4953    | 0.5262    | ✗ Bad (misaligned) |
| 789  | 0.4948    | 0.5134    | ✗ Bad (misaligned) |
| 1024 | 0.5018    | 0.5198    | ✗ Bad (misaligned) |
| 2048 | 0.4902    | 0.5104    | ✗ Bad (misaligned) |
| 3000 | 0.4956    | 0.5149    | ✗ Bad (misaligned) |
| 4096 | 0.5039    | 0.5165    | ✗ Bad (misaligned) |
| 5555 | 0.5003    | 0.5183    | ✗ Bad (misaligned) |
| 6789 | 0.4958    | 0.5123    | ✗ Bad (misaligned) |
| 8192 | 0.4917    | 0.5445    | ✗ Bad (misaligned) |

**Mean:** AUC = 0.544 ± 0.143 (9/10 seeds affected)

## The Fix

### Changes Made

Modified `data/News20_PU.py` to:

1. **Always fetch with seed 42** (matching embeddings generation):
   ```python
   train_bunch = fetch_20newsgroups(
       subset="train",
       shuffle=True,
       random_state=42,  # Fixed seed matching embeddings file
   )
   ```

2. **Shuffle with experiment seed AFTER** labels and embeddings are aligned:
   ```python
   # Shuffle train and test data with experiment seed
   train_shuffle_idx = rng.permutation(len(X_train))
   X_train = X_train[train_shuffle_idx]
   y_train_bin = y_train_bin[train_shuffle_idx]

   test_shuffle_idx = rng.permutation(len(X_test))
   X_test = X_test[test_shuffle_idx]
   y_test_bin = y_test_bin[test_shuffle_idx]
   ```

### Verification

**After fix:**

| Seed | Oracle AUC | Oracle Acc | Status |
|------|-----------|-----------|--------|
| 42   | 0.9528    | 0.8902    | ✓ Good |
| 456  | 0.9529    | 0.8887    | ✓ Good |
| 789  | 0.9526    | 0.8887    | ✓ Good |
| 1024 | 0.9528    | 0.8898    | ✓ Good |
| 2048 | 0.9531    | 0.8895    | ✓ Good |
| 3000 | 0.9532    | 0.8893    | ✓ Good |
| 4096 | 0.9533    | 0.8889    | ✓ Good |
| 5555 | 0.9528    | 0.8899    | ✓ Good |
| 6789 | 0.9533    | 0.8886    | ✓ Good |
| 8192 | 0.9533    | 0.8883    | ✓ Good |

**Mean:** AUC = 0.9530 ± 0.0003 (all seeds working correctly)

## Impact

### Affected Experiments

- **Phase 1 Extended** (results_phase1_extended/): All 20News experiments
  - Total affected: 210 experiments (7 datasets × 10 seeds × 3 c values)
  - Only seed 42 results were valid (21 experiments)
  - Remaining 189 experiments showed random performance

### Action Required

**Phase 1 Extended must be re-run** for 20News dataset only:
- 1 dataset (20News)
- 10 seeds (42, 456, 789, 1024, 2048, 3000, 4096, 5555, 6789, 8192)
- 3 c values (0.1, 0.3, 0.5)
- 18 methods (excluding VAE-PU, PUET which were incomplete)
- **Total:** ~540 experiments to re-run

### Unaffected Datasets

The following datasets were **NOT affected** by this bug:
- **IMDB**: Uses HuggingFace dataset loading which is deterministic
- **MNIST**: No precomputed embeddings
- **FashionMNIST**: No precomputed embeddings
- **Mushrooms**: No precomputed embeddings
- **Spambase**: No precomputed embeddings
- **Connect4**: No precomputed embeddings

## Prevention

To prevent similar bugs in future:

1. **Document embedding generation**: Record the exact parameters used to generate precomputed embeddings
2. **Validate alignment**: Add assertion checks that embeddings and labels match
3. **Test multiple seeds**: Always test with at least 2-3 different seeds to catch alignment issues early
4. **Consistent loading**: Use deterministic data loading (shuffle=False or fixed seed) before any experiment-specific shuffling

## Date

Fixed: 2026-04-20
