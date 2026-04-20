# Phase 3 Multi-Seed Expansion

## Overview

Expand Phase 3 from 1 seed to 5 seeds to improve statistical robustness of VPU prior/label-frequency analysis.

## Motivation

**Current Phase 3:**
- 1 seed (42) provides point estimates
- Cannot assess variance or confidence intervals
- Limited ability to detect statistically significant differences

**5-Seed Phase 3:**
- Enables variance decomposition (dataset vs seed effects)
- Provides confidence intervals for all metrics
- Allows statistical significance testing between method_prior variants
- Matches Phase 1 Extended seed count for consistency

## Configuration

### Seeds
```yaml
random_seeds: [42, 456, 789, 1024, 2048]  # 5 seeds
```

Matches first 5 seeds from Phase 1 Extended (which uses 10 seeds total).

### Total Experiments

**Complete configuration:**
- 7 datasets: MNIST, FashionMNIST, IMDB, 20News, Mushrooms, Spambase, Connect4
- 5 seeds: [42, 456, 789, 1024, 2048]
- 7 label frequencies (c): [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
- 7 true priors (π): [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
- 10 method configurations:
  - vpu, vpu_nomixup (2 base methods)
  - vpu_mean_prior (auto, 0.5, 0.69) (3 variants)
  - vpu_nomixup_mean_prior (auto, 0.5, 0.69) (3 variants)
  - oracle_bce, pn_naive (2 baselines)

**Total:** 7 × 5 × 7 × 7 × 10 = **17,150 experiments**

### Incremental Execution

**Already completed (seed 42):**
- 7 × 1 × 7 × 7 × 10 = 3,430 experiments
- Includes all 10 method configs (with 0.69 variants)

**Additional runs needed (4 new seeds):**
- 7 × 4 × 7 × 7 × 10 = 13,720 experiments
- Seeds: 456, 789, 1024, 2048

**Execution approach:**
- Use `--resume` flag to skip seed 42 (already complete)
- Run only the 4 new seeds
- Same `results_phase3/` output directory

## Estimated Runtime

**Per-experiment average:** ~35 seconds (based on current Phase 3 performance)

**With 8 workers:**
- Total time: 13,720 × 35 / 8 / 3600 = **16.7 hours**
- Overnight run: Start ~23:00, complete by ~15:40 next day

**Resource requirements:**
- Compute: 8-core machine
- Memory: ~32 GB (8 workers × 4 GB each)
- Storage: ~200-300 GB additional (raw scores + metrics)

## Expected Benefits

### 1. Variance Decomposition

With 5 seeds per configuration, can decompose variance:
```
Var(metric) = Var(dataset) + Var(seed) + Var(interaction)
```

**From Phase 1 Extended (10 seeds):**
- Dataset heterogeneity ~7.5× larger than seed variance
- With 5 seeds, can verify this holds for Phase 3 grid

### 2. Confidence Intervals

**Current (1 seed):** Point estimates only
- Example: AUC = 0.95 (no uncertainty)

**With 5 seeds:** Mean ± CI
- Example: AUC = 0.95 ± 0.02 (95% CI)
- Can identify when differences are statistically significant

### 3. Method Comparison Robustness

**Key question:** Does method_prior=0.69 outperform 0.5 across prior grid?

**With 5 seeds:**
- Compute mean difference: Δ_AUC(0.69 vs 0.5)
- Test significance: paired t-test across seeds
- Report: "0.69 outperforms 0.5 by 0.02 AUC (p < 0.01)" vs just "0.69 > 0.5"

### 4. Failure Mode Detection

With 5 seeds, can identify:
- **Consistent failures:** All 5 seeds fail (method problem)
- **Occasional failures:** 1-2 seeds fail (initialization sensitivity)
- **Seed-dependent success:** Some seeds work, others don't (robustness issue)

## Analysis Enhancements

### Heatmaps with Error Bars

**Current:** AUC(π, c) heatmap (single value per cell)

**With 5 seeds:** AUC(π, c) ± std heatmap
- Color: mean performance
- Error bars or annotations: standard deviation
- Identifies (π, c) regions with high variance

### Prior Sensitivity Curves

**Fixed c, vary π:**
```
AUC(π) = f(π) ± CI, for c ∈ {0.1, 0.5, 0.9}
```

With 5 seeds, can plot confidence bands around curves.

### Method Prior Comparison

For each (π, c) configuration:
- Compare auto vs 0.5 vs 0.69
- Statistical significance test (repeated measures ANOVA or paired t-tests)
- Report: "At π=0.3, c=0.1: auto > 0.5 (p=0.02), auto ≈ 0.69 (p=0.12)"

## Execution Plan

### Step 1: Wait for Phase 3 (0.69) Completion
- Current status: 613/686 complete (~5 min remaining)
- ETA: 22:40

### Step 2: Update Configs (DONE)
✅ All 7 Phase 3 configs updated to include 5 seeds

### Step 3: Start Multi-Seed Run
```bash
bash scripts/run_phase3.sh
```

**What happens:**
- Script uses same configs (now with 5 seeds)
- `--resume` flag skips all seed 42 experiments
- Runs 13,720 new experiments (seeds 456, 789, 1024, 2048)
- Results saved to same `results_phase3/` directory

### Step 4: Monitor Progress

**Check progress:**
```bash
# Count completed experiments per seed
find results_phase3 -name "*.json" | grep "seed_456" | wc -l
find results_phase3 -name "*.json" | grep "seed_789" | wc -l
find results_phase3 -name "*.json" | grep "seed_1024" | wc -l
find results_phase3 -name "*.json" | grep "seed_2048" | wc -l
```

**Expected:**
- Each seed should have 1,029 experiment files
- Total: 4 seeds × 1,029 = 4,116 new files
- Plus seed_42: 1,029 existing files
- Grand total: 5,145 experiment files

### Step 5: Verification

After completion, verify:
```python
python scripts/verify_phase3_multiseed.py
```

Expected output:
- 17,150 total method runs (5 seeds × 3,430 per seed)
- All 10 methods × 5 seeds × 343 configs = complete coverage

## Success Criteria

1. ✅ All 17,150 method runs complete
2. ✅ Each seed has 3,430 method runs (343 configs × 10 methods)
3. ✅ All (dataset, c, π, method, seed) combinations present
4. ✅ No duplicate runs (same config run multiple times)
5. ✅ Timing metrics available for all runs

## Timeline

**Immediate:**
- Phase 3 (0.69) completion: ~22:40 (5 min)

**Tonight:**
- Start multi-seed run: ~22:45
- Overnight execution: 16-17 hours

**Tomorrow:**
- Completion: ~15:30-16:00
- Verification: +30 min
- Initial analysis: +2 hours

**Total:** ~1 day wall-clock time (mostly overnight)

## Storage Impact

**Current results_phase3/:**
- ~175 MB (seed 42 with 10 method configs)

**After multi-seed:**
- ~700 MB (5 seeds × 140 MB average)
- Plus raw scores if saving: +300 MB
- **Total:** ~1 GB for Phase 3 multi-seed

Plenty of room after cleanup (~7 GB free).

## Files Modified

- `config/phase3/*.yaml` (7 files) - Updated random_seeds to [42, 456, 789, 1024, 2048]

## Files Created

- `docs/PHASE3_MULTISEED_PLAN.md` (this file)

## Execution Script

Same script as before:
```bash
scripts/run_phase3.sh
```

Script already supports `--resume` and will automatically:
- Detect existing seed 42 results
- Skip those experiments
- Run only seeds 456, 789, 1024, 2048

---

**Status:** Ready to execute when Phase 3 (0.69) completes (~5 minutes)
