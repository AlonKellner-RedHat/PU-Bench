# Batch Size Sensitivity Experiments

## Hypothesis

**VPU's `log(mean(φ(x)))` transformation is more sensitive to batch size than VPU-Mean's `mean(φ(x))`** because:

1. **Small batches → noisy mean estimates** → unstable log values
2. **Log amplifies variance** from small-sample means
3. **Linear averaging (VPU-Mean) is robust** to batch size variations

---

## Phase 1: Quick Validation (RUNNING)

**Status:** ⏳ In progress
**Expected completion:** ~1 hour

### Configuration

```yaml
Batch sizes: [16, 64, 256]
Datasets: [MNIST, IMDB]
Methods: [vpu, vpu_mean] (at each batch size)
Seeds: [42, 123]
Label frequency: c = 0.1
Total runs: 24
```

### Expected Outcomes

**If hypothesis is correct:**
- VPU F1 at batch 16: -10% to -20% degradation
- VPU-Mean F1 at batch 16: -2% to -5% degradation
- **4-10× larger degradation for VPU**

**Threshold-independent metrics (NEW):**
- **AP (Average Precision)**: More reliable than F1 for uncalibrated methods
- **Max F1**: Shows best achievable performance with optimal threshold
- Both metrics will reveal true discrimination ability independent of threshold

### Analysis

Run after completion:
```bash
python scripts/analyze_batch_size_sensitivity.py
```

This will generate:
- Performance degradation by batch size
- Statistical test (VPU vs VPU-Mean sensitivity)
- AP vs F1 comparison (reveals calibration effects)
- Recommendations for next steps

---

## Phase 2: Full Batch Size Sweep

**Trigger:** If Phase 1 validates hypothesis

### Configuration

```yaml
Batch sizes: [8, 16, 32, 64, 128, 256, 512]
Datasets: [MNIST, IMDB, Mushrooms]
Methods: [vpu, vpu_mean, vpu_nomixup, vpu_nomixup_mean, oracle_bce]
Seeds: [42, 123, 456]
Total runs: 315
Estimated time: 10-12 hours
```

### Key Questions

1. **At what batch size does VPU break down?**
   - Expected: Performance cliff at batch size < 32

2. **Is VPU-Mean truly stable across all sizes?**
   - Expected: < 5% degradation even at batch 8

3. **Does Oracle BCE (control) show sensitivity?**
   - Expected: Some degradation at very small batches

4. **How does MixUp affect sensitivity?**
   - Test vpu_nomixup vs vpu

---

## Phase 3: Interaction with Label Frequency

**Trigger:** After Phase 2

### Configuration

```yaml
Batch sizes: [16, 64, 256]
Label frequency: c = [0.01, 0.1, 0.5]
Datasets: [MNIST, IMDB]
Methods: [vpu, vpu_mean, vpu_nomixup, vpu_nomixup_mean]
Seeds: [42, 123]
Total runs: 144
Estimated time: 5-6 hours
```

### Key Question

**Is VPU MORE sensitive at low label frequency?**

Hypothesis: Yes, because:
- Fewer labeled samples per batch
- Mean over fewer points → higher variance
- Log transformation amplifies this

Expected pattern:
- c=0.01, batch 16: VPU catastrophic degradation
- c=0.5, batch 16: VPU mild degradation

---

## Implementation Details

### New Method Configs Created

**Batch size 16:**
- `config/methods/vpu_batch16.yaml`
- `config/methods/vpu_mean_batch16.yaml`

**Batch size 64:**
- `config/methods/vpu_batch64.yaml`
- `config/methods/vpu_mean_batch64.yaml`

**Batch size 256 (default):**
- `config/methods/vpu.yaml` (existing)
- `config/methods/vpu_mean.yaml` (existing)

### Dataset Configs

**Phase 1:**
- `config/datasets_batch_size/batch_validation_mnist.yaml`
- `config/datasets_batch_size/batch_validation_imdb.yaml`

### Run Scripts

**Phase 1:**
```bash
./scripts/run_batch_size_validation.sh
```

### Analysis Scripts

**Phase 1:**
```bash
python scripts/analyze_batch_size_sensitivity.py
```

---

## Metrics Tracked

### Primary Metrics

1. **F1 Score (threshold 0.5)**
   - Standard metric for comparison
   - May be affected by calibration differences

2. **Average Precision (AP)** ⭐ NEW
   - Threshold-independent
   - More reliable for batch size comparison
   - Not affected by calibration shifts

3. **Max F1** ⭐ NEW
   - Best F1 with optimal threshold
   - Shows true discrimination ability
   - Gap from F1 reveals calibration quality

### Secondary Metrics

4. **AUC** - Discrimination ability
5. **Precision/Recall** - Component analysis
6. **Epochs to convergence** - Training stability

---

## Why This Matters

### Practical Applications

**Memory-Constrained Environments:**
- Small GPUs (consumer, edge devices)
- Embedded systems
- Mobile deployment
- If batch size < 64 required → Use VPU-Mean

**Small Datasets:**
- Limited samples per epoch
- Effective batch size is smaller
- VPU may be unstable

**Low Label Frequency Scenarios:**
- Few labeled samples per batch
- VPU's log(mean()) over sparse data is noisy
- VPU-Mean more robust

### Theoretical Insights

1. **Confirms variance reduction mechanism**
   - VPU's log-of-mean relies on large-sample approximation
   - Breaks down at small batch sizes

2. **Explains calibration differences**
   - Batch size affects gradient estimates
   - VPU more sensitive → worse calibration at small batches

3. **Guides hyperparameter tuning**
   - If VPU performs poorly, check batch size first
   - May need larger batches for VPU stability

---

## Current Status

### Completed

✅ Hypothesis formulated
✅ Experimental design created
✅ Phase 1 configs created
✅ Method variants registered
✅ Analysis scripts prepared
✅ **Phase 1 benchmark launched** (24 runs, running now)

### In Progress

⏳ **Phase 1: Quick Validation**
- 24 runs executing
- Expected completion: ~1 hour
- Will use new AP/max_f1 metrics

### Next Steps

After Phase 1 completes:

1. **Run analysis:**
   ```bash
   python scripts/analyze_batch_size_sensitivity.py
   ```

2. **Check results:**
   - VPU degradation at batch 16?
   - Statistical significance?
   - AP vs F1 differences?

3. **Decision point:**
   - **If validated:** Proceed to Phase 2 (full sweep)
   - **If unclear:** Adjust hypothesis or test smaller batches
   - **If refuted:** Reconsider assumptions

---

## Expected Timeline

**Phase 1 (Quick Validation):**
- ✓ Setup: Complete
- ⏳ Execution: ~1 hour (in progress)
- Analysis: 10 minutes
- Total: ~1.5 hours

**Phase 2 (Full Sweep):**
- Setup: 30 minutes
- Execution: 10-12 hours
- Analysis: 30 minutes
- Total: 11-13 hours

**Phase 3 (Interaction):**
- Setup: 20 minutes
- Execution: 5-6 hours
- Analysis: 20 minutes
- Total: 6-7 hours

**Grand Total: ~19-22 hours** (if all phases executed)

---

## Literature Context

### Known Results

- **Variance of sample mean:** σ²/n
  - Small batches (small n) → high variance
  - Log transformation amplifies variance

- **Central Limit Theorem:**
  - Mean converges to normal with rate √n
  - Log(mean) has non-trivial sampling distribution for small n

### Novel Contribution

**First systematic study of batch size sensitivity in PU learning:**
- No prior work on VPU batch size requirements
- Standard PU methods (nnPU, etc.) not analyzed this way
- Practical guidance for deployment missing

**Use of threshold-independent metrics:**
- AP and max_f1 for fair comparison
- Separates discrimination from calibration effects
- More rigorous than F1-only analysis

---

## Monitoring Progress

**Check batch size validation status:**
```bash
tail -f /private/tmp/claude-501/-Users-akellner-MyDir-Code-Other-PU-Bench/tasks/buqwzabzm.output
```

**Count completed runs:**
```bash
find results/seed_* -name "*batch*" | wc -l
```

**Generate preliminary analysis (after some runs complete):**
```bash
python scripts/analyze_batch_size_sensitivity.py
```

---

## Success Criteria

### Phase 1 Validation

**Hypothesis confirmed if:**
1. VPU degradation at batch 16 > 10%
2. VPU-Mean degradation at batch 16 < 5%
3. Difference is statistically significant (p < 0.05)
4. AP shows similar pattern (confirms not just threshold issue)

**Proceed to Phase 2 if:** Any 3/4 criteria met

### Full Study Success

**Complete understanding if:**
1. Degradation curve measured (batch 8 to 512)
2. Breakdown point identified (where VPU fails)
3. Stability range established (where VPU-Mean reliable)
4. Interaction with c quantified
5. Practical guidelines derived

---

## Files Created

### Configuration Files
- `config/methods/vpu_batch16.yaml`
- `config/methods/vpu_batch64.yaml`
- `config/methods/vpu_mean_batch16.yaml`
- `config/methods/vpu_mean_batch64.yaml`
- `config/datasets_batch_size/batch_validation_mnist.yaml`
- `config/datasets_batch_size/batch_validation_imdb.yaml`

### Scripts
- `scripts/run_batch_size_validation.sh` (executable, running)
- `scripts/analyze_batch_size_sensitivity.py`

### Code Updates
- `run_train.py`: Added batch size method mappings

### Documentation
- `BATCH_SIZE_EXPERIMENTS.md` (this file)

---

## Summary

We're testing whether **VPU's log transformation makes it more sensitive to batch size** than VPU-Mean. Phase 1 quick validation (24 runs, ~1 hour) is currently running. If validated, we'll proceed to comprehensive batch size sweep and interaction studies.

**Key innovation:** Using **AP and max_f1** (threshold-independent metrics) for fair comparison, separating discrimination from calibration effects.

**Practical impact:** Will provide guidance on when to use VPU vs VPU-Mean based on memory/batch size constraints.
