# Adding method_prior=0.69 to Phase 1 Extended and Phase 3

## Motivation

The value **0.69** minimizes overall error assuming a uniform distribution over true priors. This represents an optimal fixed prior choice when the true prior is unknown but assumed to be uniformly distributed across [0, 1].

## Impact Summary

**Phase 1 Extended:**
- Additional methods: `vpu_mean_prior(0.69)`, `vpu_nomixup_mean_prior(0.69)`
- Additional runs: 420 (7 datasets × 10 seeds × 3 c values × 2 methods)
- Estimated time: ~40 minutes with 6 workers

**Phase 3:**
- Additional methods: `vpu_mean_prior(0.69)`, `vpu_nomixup_mean_prior(0.69)`
- Additional runs: 686 (7 datasets × 1 seed × 7 c × 7 priors × 2 methods)
- Estimated time: ~55 minutes with 8 workers

**Total additional time:** ~1.5 hours

---

## Recommended Execution Plan (Option A)

### Step 1: Wait for Current Phase 3 to Complete

**Status:** Phase 3 currently at 83% completion  
**ETA:** ~35 minutes  

**Action:** Let current Phase 3 finish to preserve 4.7 hours of completed work.

### Step 2: Update Configuration Files

**Files to modify (14 total):**

#### Phase 1 Extended configs (7 files):
- `config/comprehensive/mnist_comprehensive.yaml`
- `config/comprehensive/fashionmnist_comprehensive.yaml`
- `config/comprehensive/imdb_comprehensive.yaml`
- `config/comprehensive/20news_comprehensive.yaml`
- `config/comprehensive/mushrooms_comprehensive.yaml`
- `config/comprehensive/spambase_comprehensive.yaml`
- `config/comprehensive/connect4_comprehensive.yaml`

#### Phase 3 configs (7 files):
- `config/phase3/mnist_phase3.yaml`
- `config/phase3/fashionmnist_phase3.yaml`
- `config/phase3/imdb_phase3.yaml`
- `config/phase3/20news_phase3.yaml`
- `config/phase3/mushrooms_phase3.yaml`
- `config/phase3/spambase_phase3.yaml`
- `config/phase3/connect4_phase3.yaml`

**Change to make in each file:**

```yaml
# OLD:
method_prior_values: [null, "auto", 0.5]

# NEW:
method_prior_values: [null, "auto", 0.5, 0.69]
```

### Step 3: Run Phase 3 with --resume

**Command:**
```bash
bash scripts/run_phase3.sh
```

**What happens:**
- Script already includes `--resume` flag
- Workers will skip all 2,744 already-completed experiments
- Workers will run only the 686 new experiments with method_prior=0.69
- Results saved to same `results_phase3/` directory

**Expected time:** ~55 minutes

### Step 4: Run Phase 1 Extended with --resume

**Option 4a: Use existing script (if it has --resume)**
```bash
bash scripts/run_phase1_extended.sh
```

**Option 4b: Create targeted script**

If `run_phase1_extended.sh` doesn't exist or lacks --resume, create:

```bash
#!/bin/bash
# Run Phase 1 Extended with method_prior=0.69 additions

OUTPUT_DIR="results_phase1_extended"
SHUFFLE_SEED=54321
NUM_WORKERS=6

# VPU mean-prior methods only (will create 0.69 variants)
METHODS="vpu_mean_prior,vpu_nomixup_mean_prior"

CONFIGS=(
    "config/comprehensive/mnist_comprehensive.yaml"
    "config/comprehensive/fashionmnist_comprehensive.yaml"
    "config/comprehensive/imdb_comprehensive.yaml"
    "config/comprehensive/20news_comprehensive.yaml"
    "config/comprehensive/mushrooms_comprehensive.yaml"
    "config/comprehensive/spambase_comprehensive.yaml"
    "config/comprehensive/connect4_comprehensive.yaml"
)

function run_worker() {
    worker_id=$1
    python run_train.py \
        --dataset-config ${CONFIGS[@]} \
        --methods "$METHODS" \
        --output-dir "$OUTPUT_DIR" \
        --shuffle-seed "$SHUFFLE_SEED" \
        --num-workers "$NUM_WORKERS" \
        --worker-id "$worker_id" \
        --resume \
        > "logs/phase1_extended_069/worker_${worker_id}.log" 2>&1
}

export -f run_worker
export OUTPUT_DIR SHUFFLE_SEED NUM_WORKERS METHODS CONFIGS

mkdir -p logs/phase1_extended_069

parallel -j "$NUM_WORKERS" run_worker ::: $(seq 0 $((NUM_WORKERS - 1)))
```

**Expected time:** ~40 minutes

### Step 5: Verification

**Verify Phase 3 completeness:**
```bash
python scripts/verify_phase3_config.py
```

Expected output: 3,430 total experiments (2,744 original + 686 new)

**Verify Phase 1 Extended completeness:**
```bash
python analysis/generate_comprehensive_table.py --validate-only
```

Expected methods:
- 19 baseline methods (210 runs each)
- 2 base VPU methods (210 runs each)
- 6 VPU mean-prior variants (210 runs each):
  - vpu_mean_prior(auto), vpu_mean_prior(0.5), **vpu_mean_prior(0.69)**
  - vpu_nomixup_mean_prior(auto), vpu_nomixup_mean_prior(0.5), **vpu_nomixup_mean_prior(0.69)**

Total: 27 methods × 210 runs = 5,670 experiments

---

## Alternative Options (Not Recommended)

### Option B: Stop and Restart Phase 3 Now

**Pros:**
- Single unified run

**Cons:**
- Wastes 4.7 hours of completed work (2,279 runs)
- Total time: 3.4 hours vs 1.5 hours for Option A

**Not recommended** unless there's a critical issue with current Phase 3 results.

### Option C: Separate Directory Structure

Run 0.69 experiments in separate directories:
- `results_phase3_069/`
- `results_phase1_extended_069/`

**Pros:**
- Clear separation of original vs new runs

**Cons:**
- Analysis scripts need to merge results from multiple directories
- More complex bookkeeping

**Use only if:** You want to keep results completely separate for comparison.

---

## Implementation Checklist

- [ ] Wait for current Phase 3 to complete
- [ ] Update 7 Phase 1 Extended config files (add 0.69 to method_prior_values)
- [ ] Update 7 Phase 3 config files (add 0.69 to method_prior_values)
- [ ] Create execution script for Phase 1 Extended 0.69 additions
- [ ] Run Phase 3 with resume (~55 min)
- [ ] Run Phase 1 Extended with resume (~40 min)
- [ ] Verify completeness of Phase 3 (3,430 total experiments)
- [ ] Verify completeness of Phase 1 Extended (5,670 total experiments)
- [ ] Regenerate comprehensive tables with new method variants
- [ ] Update analysis to compare auto vs 0.5 vs 0.69

---

## Expected Analysis Insights

Adding 0.69 will enable:

1. **Optimal fixed prior comparison:** Compare 0.69 (theoretically optimal under uniform prior distribution) vs 0.5 (balanced assumption)

2. **Prior mismatch robustness:** 
   - When true prior π = 0.5: 0.5 should win
   - When true prior π ≠ 0.5: 0.69 may outperform 0.5 on average
   - When true prior is unknown: 0.69 minimizes worst-case error

3. **Validation of theoretical optimality:** Empirical verification that 0.69 minimizes average error across uniform prior distribution

4. **Practical guidance:** Determine when practitioners should use:
   - `auto` (when confident in labeled set representativeness)
   - `0.69` (when prior is completely unknown)
   - `0.5` (when assuming balanced classes)

---

## Timeline

**Immediate:** Wait for Phase 3 completion (~35 min remaining)  
**After Phase 3:** Update configs and run Phase 3 0.69 additions (~55 min)  
**After Phase 3 0.69:** Run Phase 1 Extended 0.69 additions (~40 min)  

**Total additional time from now:** ~2.1 hours (including wait time)

---

## Files Created

This plan will be executed by modifying existing configs and running existing scripts with `--resume` flag. No new code required.
