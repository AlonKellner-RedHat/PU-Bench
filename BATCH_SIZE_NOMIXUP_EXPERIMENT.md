# Batch Size Experiment: WITHOUT MixUp

## Motivation

The previous batch size experiments (WITH MixUp) showed:
1. **Equal sensitivity** between VPU and VPU-Mean (-0.6% AP degradation at batch 2)
2. **Dataset effects dominate** over method choice
3. **Couldn't test batch size 1** (MixUp requires ≥2 samples per batch)

**Key question:** Does removing MixUp change batch size sensitivity?

## Experimental Setup

### Methods
- **vpu_nomixup**: VPU with log(mean(φ(x))) variance reduction, NO MixUp
- **vpu_nomixup_mean**: VPU-Mean with mean(φ(x)) variance reduction, NO MixUp

### Batch Sizes Tested
**1, 2, 4, 8, 16, 64, 256**

*Note: Batch size 1 is only possible without MixUp*

### Datasets
- **MNIST**: Easy vision task (AUC ~0.99)
- **IMDB**: Hard NLP task (AUC ~0.85)

### Configuration
- **Seeds:** 42, 123 (2 seeds)
- **Label frequency (c):** 0.1 (10% labeled positives)
- **Total runs:** 7 batch sizes × 2 datasets × 2 methods × 2 seeds = **56 runs**
- **Estimated time:** 8-12 hours

## Files Created

### Configuration Files
Created 12 new method config files in `config/methods/`:

**VPU-NoMixUp variants:**
- `vpu_nomixup_batch1.yaml` (batch size 1)
- `vpu_nomixup_batch2.yaml` (batch size 2)
- `vpu_nomixup_batch4.yaml` (batch size 4)
- `vpu_nomixup_batch8.yaml` (batch size 8)
- `vpu_nomixup_batch16.yaml` (batch size 16)
- `vpu_nomixup_batch64.yaml` (batch size 64)

**VPU-NoMixUp-Mean variants:**
- `vpu_nomixup_mean_batch1.yaml` (batch size 1)
- `vpu_nomixup_mean_batch2.yaml` (batch size 2)
- `vpu_nomixup_mean_batch4.yaml` (batch size 4)
- `vpu_nomixup_mean_batch8.yaml` (batch size 8)
- `vpu_nomixup_mean_batch16.yaml` (batch size 16)
- `vpu_nomixup_mean_batch64.yaml` (batch size 64)

### Scripts

**Training script:**
```bash
./scripts/run_nomixup_batch_sizes.sh
```

**Analysis script:**
```bash
python scripts/analyze_nomixup_batch_sizes.py
```

### Trainer Registration
Updated `run_train.py` to register all 12 new method variants to the appropriate trainers:
- `VPUNoMixUpTrainer` for vpu_nomixup variants
- `VPUNoMixUpMeanTrainer` for vpu_nomixup_mean variants

## Research Questions

### 1. Does batch size 1 work without MixUp?
**Previous result (WITH MixUp, batch 1):**
- Catastrophic failure: AUC 0.2789 (random is 0.5)
- Accuracy: 34.7%
- Speed: ~225 sec/epoch

**Question:** Does removing MixUp make batch 1 viable?

### 2. Does batch size sensitivity change without MixUp?
**Previous results (WITH MixUp):**
- Both VPU and VPU-Mean: -0.6% AP at batch 2 vs 256
- MNIST: Better at small batches (+0.2% AP at batch 2)
- IMDB: Better at large batches (-1.5% AP at batch 2)

**Question:** Are these patterns the same without MixUp?

### 3. What is the performance cost/benefit of removing MixUp?
**Hypothesis:** MixUp provides regularization that may:
- Improve generalization (higher AP/AUC)
- Stabilize training at small batches
- Slow down convergence (more epochs needed)

**Question:** How much does MixUp help at each batch size?

### 4. Does VPU vs VPU-Mean sensitivity differ without MixUp?
**Previous hypothesis (rejected WITH MixUp):**
- "VPU's log transformation is more sensitive to batch size than VPU-Mean"
- Result: Both equally sensitive with MixUp

**Question:** Does this change without MixUp?

## Expected Comparisons

The analysis will compare:

### Within No-MixUp
- Batch size sensitivity curve (1→256)
- VPU-NoMixUp vs VPU-NoMixUp-Mean
- MNIST vs IMDB behavior

### Across MixUp Status
- With MixUp (batch 2-256) vs Without MixUp (batch 1-256)
- Performance at each batch size
- Optimal batch size selection

### Key Metrics
All using threshold-independent metrics:
- **AP (Average Precision)** - Primary discrimination metric
- **max_f1** - Best achievable F1
- **AUC** - Overall discrimination
- **A-NICE** - Calibration quality
- **Convergence speed** - Epochs and time to best

## Running the Experiment

```bash
# Start the experiment (runs in foreground)
./scripts/run_nomixup_batch_sizes.sh

# Or run in background
nohup ./scripts/run_nomixup_batch_sizes.sh > nomixup_training.log 2>&1 &

# Monitor progress
tail -f nomixup_training.log

# After completion, analyze results
python scripts/analyze_nomixup_batch_sizes.py
```

## Expected Outputs

### During Training
Results saved to `results/seed_{42,123}/`:
- `MNIST_case-control_random_c0.1_seed{42,123}.json`
- `IMDB_case-control_random_c0.1_seed{42,123}.json`

Each file will contain runs for:
- vpu_nomixup_batch{1,2,4,8,16,64}
- vpu_nomixup
- vpu_nomixup_mean_batch{1,2,4,8,16,64}
- vpu_nomixup_mean

### After Analysis
Generated report: `results/BATCH_SIZE_NOMIXUP_ANALYSIS.md`

Sections:
1. **MixUp vs No-MixUp Comparison** at batch 256 baseline
2. **Batch Size 1 Analysis** (only possible without MixUp)
3. **Batch Size Sensitivity** curves with and without MixUp
4. **Recommendations** for optimal batch sizes

## Integration with Previous Results

This experiment complements the earlier batch size study:

| Experiment | Batch Sizes | MixUp | Methods | Runs |
|------------|-------------|-------|---------|------|
| Previous   | 2, 4, 8, 16, 64, 256 | Yes | vpu, vpu_mean | 48 |
| This       | 1, 2, 4, 8, 16, 64, 256 | No | vpu_nomixup, vpu_nomixup_mean | 56 |
| **Total**  | 1-256 | Both | 4 variants | **104 runs** |

Combined analysis will provide comprehensive understanding of:
- Batch size effects across 1-256
- MixUp's role in batch size sensitivity
- Optimal configurations for different scenarios
