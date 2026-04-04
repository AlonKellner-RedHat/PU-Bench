# Comprehensive Experiment Suite Implementation

**Date:** 2026-04-03
**Status:** ✅ Implementation Complete, Ready for Execution

---

## Overview

Implemented a comprehensive, publication-ready experiment suite for comparing all PU learning methods across 9 datasets with extensive parameter coverage.

**Experiment Scale:**
- **Total Experiments:** 6,750
- **Datasets:** 9 (MNIST, FashionMNIST, IMDB, 20News, Mushrooms, Spambase, Connect4, CIFAR10, AlzheimerMRI)
- **Methods:** 10 total
  - 8 VPU variants: vpu, vpu_nomixup, vpu_mean_prior×3 (0.5/auto/1.0), vpu_nomixup_mean_prior×3 (0.5/auto/1.0)
  - 2 baselines: nnpu, distpu
- **Seeds:** 5 [42, 456, 789, 1024, 2048]
- **Label Frequencies (c):** 3 [0.01, 0.1, 0.5]
- **True Priors:** 5 [0.1, 0.3, 0.5, 0.7, 0.9]
- **Estimated Runtime:** ~21 hours with 4 parallel workers

**Calculation:** 9 datasets × 5 seeds × 3 c × 5 true_priors × (2 base VPU + 6 VPU with prior grid + 2 baselines) = 6,750

---

## Implementation Phases

### ✅ Phase 1: Dataset Loader Modifications

**Files Modified:**
1. `/Users/akellner/MyDir/Code/Other/PU-Bench/data/Connect4_PU.py`
2. `/Users/akellner/MyDir/Code/Other/PU-Bench/data/CIFAR10_PU.py`
3. `/Users/akellner/MyDir/Code/Other/PU-Bench/data/AlzheimerMRI_PU.py`

**Changes:**
- Added `target_prevalence_train: float | None = None` parameter to all three loaders
- Implemented training set resampling BEFORE PU split creation
- Resampling occurs after train/val split but before `create_pu_training_set()`
- This ensures unlabeled set U inherits the target prior

**Special Fix for Connect4:**
- OpenML Connect-4 dataset has numeric labels (-1, 0, 1) not strings
- Fixed loader to properly map numeric labels to binary classes
- Created mapping: loss→-1.0, draw→0.0, win→1.0

**Code Pattern:**
```python
# Split validation set
train_features, train_labels, val_features, val_labels = split_train_val(...)

# NEW: Resample training set to target true prior (if specified)
if target_prevalence_train is not None and target_prevalence_train > 0:
    train_features, train_labels = resample_by_prevalence(
        train_features, train_labels, target_prevalence_train, random_seed
    )

# Create PU split (now with resampled data)
pu_train_features, ... = create_pu_training_set(...)
```

### ✅ Phase 2: Custom Prior Testing

**Test Configurations Created:**
- 7 test configs in `config/test/`
  - Connect4: priors [0.1, 0.5, 0.9]
  - CIFAR10: prior [0.5]
  - AlzheimerMRI: priors [0.1, 0.5, 0.9]

**Test Script:** `scripts/test_custom_priors.sh`
- Tests each problematic dataset with multiple priors
- Verifies no crashes during data loading
- Checks actual measured prior matches target (within ±0.05)
- Confirms PU split completes successfully
- Validates training runs without errors

**Status:** Tests currently running in background

### ✅ Phase 3: Comprehensive Configs

**Configs Created:** 9 files in `config/comprehensive/`
1. `mnist_comprehensive.yaml`
2. `fashionmnist_comprehensive.yaml`
3. `imdb_comprehensive.yaml`
4. `20news_comprehensive.yaml`
5. `mushrooms_comprehensive.yaml`
6. `spambase_comprehensive.yaml`
7. `connect4_comprehensive.yaml`
8. `cifar10_comprehensive.yaml`
9. `alzheimermri_comprehensive.yaml`

**Common Structure:**
```yaml
dataset_class: <DATASET>
data_dir: ./datasets
random_seeds: [42, 456, 789, 1024, 2048]  # 5 seeds
c_values: [0.01, 0.1, 0.5]  # 3 label frequencies

# Cartesian grid dimensions
target_prevalence_train_values: [0.1, 0.3, 0.5, 0.7, 0.9]  # 5 true priors
method_prior_values: [0.5, null, 1.0]  # For VPU mean_prior variants (null = "auto")

# PU setup
scenarios: [case-control]
selection_strategies: [random]
val_ratio: 0.01
target_prevalence: [null]  # Don't resample test set
with_replacement: true
case_control_mode: "naive_mode"
also_print_dataset_stats: false
```

**Note:** Text datasets (IMDB, 20News) include SBERT embeddings paths

### ✅ Phase 4: Randomized Execution

**File Modified:** `/Users/akellner/MyDir/Code/Other/PU-Bench/run_train.py`

**Changes:**

1. **New Command-Line Arguments:**
   ```python
   --shuffle-seed: Seed for randomizing experiment order (default: 42)
   --num-workers: Number of parallel workers (default: 1)
   --worker-id: Worker ID for distributed execution (0 to num_workers-1)
   ```

2. **Flattened Experiment Queue:**
   - Replaced nested loops with single flat list of experiment configs
   - Each experiment is a dict with: dataset_class, data_cfg, target_prev_train, method_prior, method, seed

3. **Randomization:**
   ```python
   random.seed(args.shuffle_seed)
   random.shuffle(all_experiments)
   ```

4. **Worker Distribution:**
   ```python
   if args.worker_id is not None:
       worker_experiments = [
           exp for i, exp in enumerate(all_experiments)
           if i % args.num_workers == args.worker_id
       ]
   ```

5. **Progress Tracking:**
   - Output format: `[i/total] ▶ RUN method on exp_name`
   - Clear status indicators: ⏭ SKIP, ▶ RUN, ✔ DONE, ✗ FAILED

**Benefits:**
- Randomized order prevents dataset-specific bottlenecks
- Balanced load distribution across workers
- Graceful resume after interruption
- Each worker processes every Nth experiment (modulo distribution)

### ✅ Phase 5: Execution Script

**File Created:** `scripts/run_comprehensive.sh`

**Features:**
- Launches 4 parallel workers
- Uses GNU parallel if available, otherwise background processes
- Deterministic randomization with fixed shuffle seed (12345)
- Each worker logs to separate file: `logs/comprehensive/worker_N.log`
- Resume capability via `--resume` flag
- Automatic result counting at completion

**Usage:**
```bash
bash scripts/run_comprehensive.sh
```

**Worker Distribution:**
- Worker 0: experiments 0, 4, 8, 12, ...
- Worker 1: experiments 1, 5, 9, 13, ...
- Worker 2: experiments 2, 6, 10, 14, ...
- Worker 3: experiments 3, 7, 11, 15, ...

All workers use the same shuffled order, so distribution is deterministic and resumable.

### ✅ Phase 6: Verification Scripts

**Scripts Created:**

1. **`scripts/verify_completeness.py`**
   - Counts experiments per method
   - Compares actual vs expected counts
   - Expected: vpu=675, vpu_nomixup=675, vpu_mean_prior=2,025, vpu_nomixup_mean_prior=2,025, nnpu=675, distpu=675
   - Total: 6,750

2. **`scripts/verify_oracle_ce.py`**
   - Checks all results have `test_oracle_ce` metric
   - Validates not None or NaN
   - Ensures Oracle CE is present for all methods

3. **`scripts/verify_data_quality.py`**
   - Checks for NaN, Inf values
   - Validates metrics in valid ranges:
     - AP, AUC, F1, Accuracy, Precision, Recall: [0.0, 1.0]
     - ECE, MCE: [0.0, 1.0]
     - Brier: [0.0, 2.0]
   - Reports any data quality issues

**Usage:**
```bash
python scripts/verify_completeness.py
python scripts/verify_oracle_ce.py
python scripts/verify_data_quality.py
```

---

## Key Technical Decisions

### 1. Method Prior Grid
- **method_prior_values: [0.5, null, 1.0]**
- `null` → method_prior="auto" → computed from labeled data
- Creates 3 variants for vpu_mean_prior and vpu_nomixup_mean_prior
- Total: 2 base VPU + 6 with prior grid = 8 VPU variants

### 2. Randomization Strategy
- Fixed shuffle_seed=12345 ensures reproducibility
- All workers see same randomized order
- Modulo distribution ensures balanced load
- Can restart with same seed to resume

### 3. True Prior Implementation
- Resample training set BEFORE PU split
- Unlabeled set U inherits target prior naturally
- Labeled set P size determined by c (labeled_ratio)
- Clean separation between true_prior and label_frequency

### 4. Oracle CE Metric
- Already implemented universally in `train/train_utils.py` (lines 718-743)
- Uses true labels to compute binary cross-entropy
- Formula: `-[y*log(p) + (1-y)*log(1-p)]`
- No code changes needed - metric already universal

---

## Files Created/Modified Summary

### Created (15 files):

**Test Configs (7):**
- `config/test/connect4_test_prior{0.1,0.5,0.9}.yaml`
- `config/test/cifar10_test_prior0.5.yaml`
- `config/test/alzheimermri_test_prior{0.1,0.5,0.9}.yaml`

**Comprehensive Configs (9):**
- `config/comprehensive/{mnist,fashionmnist,imdb,20news,mushrooms,spambase,connect4,cifar10,alzheimermri}_comprehensive.yaml`

**Scripts (4):**
- `scripts/test_custom_priors.sh`
- `scripts/run_comprehensive.sh`
- `scripts/verify_completeness.py`
- `scripts/verify_oracle_ce.py`
- `scripts/verify_data_quality.py`

### Modified (4 files):

**Dataset Loaders (3):**
- `data/Connect4_PU.py` - Added target_prevalence_train + fixed label mapping
- `data/CIFAR10_PU.py` - Added target_prevalence_train
- `data/AlzheimerMRI_PU.py` - Added target_prevalence_train

**Execution (1):**
- `run_train.py` - Added randomization + worker distribution

---

## Execution Instructions

### Step 1: Verify Test Results
```bash
# Check test suite completed successfully
tail -n 50 logs/full_prior_test.log
```

### Step 2: Run Comprehensive Experiments
```bash
# Launch 4 workers (will run for ~21 hours)
bash scripts/run_comprehensive.sh

# Monitor progress
tail -f logs/comprehensive/worker_0.log
```

### Step 3: Verify Results
```bash
# Check completeness
python scripts/verify_completeness.py

# Verify Oracle CE present
python scripts/verify_oracle_ce.py

# Check data quality
python scripts/verify_data_quality.py
```

### Step 4: Analyze Results
```bash
# Count results
find results_comprehensive -name "*.json" | wc -l

# Expected: 6,750 total experiments
# With 10 methods per experiment: ~6,750 result entries across files
```

---

## Success Criteria

✅ **Implementation Complete:**
1. ✅ All 3 dataset loaders support target_prevalence_train
2. ✅ Connect4 label mapping fixed
3. ✅ 9 comprehensive configs created
4. ✅ run_train.py supports randomized execution
5. ✅ Execution script with 4 workers created
6. ✅ 3 verification scripts created

**Execution Criteria (To Be Verified):**
1. ⏳ Test suite passes for all problematic datasets (running)
2. ⏳ 6,750 experiments complete successfully
3. ⏳ All experiments include Oracle CE metric
4. ⏳ No NaN, Inf, or out-of-range values
5. ⏳ Execution completes within ~24 hours

---

## Timeline Estimate

- **Development:** ✅ Complete (~3 hours actual)
- **Testing:** ⏳ In progress (~30 minutes)
- **Execution:** Not started (~21 hours with 4 workers)
- **Verification:** Not started (~30 minutes)
- **Analysis:** Not started (~4-6 hours)

**Total end-to-end:** ~29-31 hours (1.2-1.3 days)

---

## Next Steps

1. **Wait for test suite to complete** - verify all custom prior tests pass
2. **Launch comprehensive experiments** - run `bash scripts/run_comprehensive.sh`
3. **Monitor execution** - check worker logs periodically
4. **Verify results** - run all three verification scripts
5. **Analyze data** - create comparison tables and visualizations
6. **Write paper section** - comprehensive results with all 10 methods

---

## Contact

For issues or questions:
- Check logs in `logs/comprehensive/worker_*.log`
- Review test results in `logs/test_priors/`
- Verify configs in `config/comprehensive/`
