# Phase 3: VPU Prior/Label-Frequency Analysis

## Overview

Phase 3 is a focused deep-dive into **VPU method performance** across a comprehensive grid of true class priors (π) and label frequencies (c). This phase complements Phase 1's broad method comparison with a detailed analysis of how VPU variants behave under different data conditions.

## Motivation

**From Phase 1 findings:**
- VPU variants (especially VPU-nomix-MP(0.5)) showed strong overall performance
- Limited exploration of prior/label-frequency space (only 3 c values, natural priors)
- Need to understand VPU sensitivity to these critical hyperparameters

**Phase 3 addresses:**
1. **Full prior range:** How does VPU perform when true prior π varies from 1% to 99%?
2. **Extreme label scarcity:** What happens at c=0.01 (only 1% of positives labeled)?
3. **High labeling rates:** Does c=0.99 approach supervised performance?
4. **Prior mismatch:** How robust is VPU-mean-prior when the assumed prior is wrong?

## Experimental Design

### Dimensions

| Dimension | Values | Count |
|-----------|--------|-------|
| **Datasets** | MNIST, FashionMNIST, IMDB, 20News, Mushrooms, Spambase, Connect4 | 7 |
| **Seeds** | 42 | 1 |
| **Label Frequency (c)** | 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99 | 7 |
| **True Prior (π)** | 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99 | 7 |
| **Methods** | VPU variants + Oracle-PN + Naive-PN | 6 base methods |

**Total experiments:** 7 × 1 × 7 × 7 × 8 = **2,744 experiments**

(Note: 8 includes method_prior variants for mean_prior methods)

### Methods

**VPU Variants (4 base methods):**
1. **vpu** - Standard VPU with mixup
2. **vpu_nomixup** - VPU without mixup
3. **vpu_mean_prior** - VPU with mean-prior regularization + mixup
   - Variants: auto (computed from labeled set), 0.5 (fixed)
4. **vpu_nomixup_mean_prior** - VPU with mean-prior, no mixup
   - Variants: auto, 0.5

**Baselines (2 methods):**
5. **oracle_bce** - Oracle trained with true labels (performance ceiling)
6. **pn_naive** - Naive PN treating unlabeled as negative (performance floor)

### Parameter Grid

#### Label Frequency (c)
```
c = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
```
- **c=0.01:** Extreme scarcity (1 in 100 positives labeled)
- **c=0.1:** Low labeling rate (typical in medical applications)
- **c=0.3, 0.5:** Moderate labeling
- **c=0.7, 0.9:** High labeling rates
- **c=0.99:** Nearly fully labeled (near-supervised setting)

#### True Prior (π)
```
π = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
```
- **π=0.01:** Extreme imbalance (1% positive)
- **π=0.1, 0.3:** Imbalanced (10-30% positive)
- **π=0.5:** Balanced classes
- **π=0.7, 0.9:** Imbalanced (70-90% positive)
- **π=0.99:** Extreme imbalance (99% positive)

### Key Differences from Phase 1

| Aspect | Phase 1 | Phase 3 |
|--------|---------|---------|
| **Focus** | Broad method comparison | VPU deep-dive |
| **Methods** | 17 methods | 6 methods (VPU + baselines) |
| **Seeds** | 10 seeds | 1 seed |
| **Label freq (c)** | 3 values [0.1, 0.3, 0.5] | 7 values [0.01-0.99] |
| **True prior (π)** | Natural (dataset-dependent) | 7 values [0.01-0.99] |
| **Total experiments** | ~3,570 | 2,744 |
| **Goal** | Which method is best overall? | How does VPU handle extreme conditions? |

## Research Questions

### 1. Prior Sensitivity
- **Q:** How sensitive are VPU variants to true class prior π?
- **Analysis:** Fix c, vary π, measure performance degradation
- **Hypothesis:** VPU-mean-prior should be more robust to prior shifts

### 2. Label Scarcity
- **Q:** At what labeling rate does VPU performance collapse?
- **Analysis:** Fix π, vary c from 0.01 to 0.99
- **Hypothesis:** Performance degrades gracefully until c < 0.05

### 3. Method Prior Mismatch
- **Q:** How much does wrong mean_prior assumption hurt?
- **Analysis:** Compare auto vs. 0.5 across different true priors
- **Hypothesis:** Auto should outperform fixed 0.5 when π ≠ 0.5

### 4. Mixup Benefit
- **Q:** Does mixup help more at extreme priors or extreme c values?
- **Analysis:** Compare vpu vs. vpu_nomixup across grid
- **Hypothesis:** Mixup more beneficial at low c (data scarcity)

### 5. Oracle Gap
- **Q:** How close can VPU get to Oracle at high c?
- **Analysis:** Compare VPU vs. Oracle-PN at c=0.99
- **Hypothesis:** Gap < 2% AUC when c=0.99

## Expected Outputs

### 1. Heatmaps
- **AUC vs. (π, c)** for each VPU variant and dataset
- Identify "safe regions" where VPU performs well
- Highlight failure modes (e.g., π=0.01, c=0.01)

### 2. Sensitivity Curves
- **Performance vs. π** (fixing c=0.1, 0.5, 0.9)
- **Performance vs. c** (fixing π=0.1, 0.5, 0.9)
- Error bars from bootstrapping single seed

### 3. Method Comparison Tables
- **Best method by regime:**
  - Extreme imbalance: π < 0.1 or π > 0.9
  - Balanced: 0.3 < π < 0.7
  - Low labeling: c < 0.1
  - High labeling: c > 0.7

### 4. Prior Mismatch Analysis
- **Error when mean_prior=0.5 but true π ≠ 0.5**
- Quantify: `|AUC(auto) - AUC(0.5)|` across π values

## Execution

### Run Phase 3

```bash
bash scripts/run_phase3.sh
```

**Estimated time:** ~15-20 hours with 8 workers

**Progress monitoring:**
```bash
# Count completed experiments
find results_phase3 -name "*.json" | wc -l

# Watch worker logs
tail -f logs/phase3/worker_0.log
```

### Resource Requirements

- **Compute:** ~16-20 hours on 8-core machine
- **Storage:** ~50-100 GB (raw scores + metrics)
- **Memory:** ~32 GB recommended (8 workers × 4 GB each)

## Analysis Scripts (To Be Created)

### 1. Generate Heatmaps
```bash
python scripts/plot_phase3_heatmaps.py
```
Creates 7 datasets × 6 methods = 42 heatmaps showing AUC(π, c)

### 2. Prior Sensitivity Analysis
```bash
python scripts/analyze_phase3_priors.py
```
Generates curves showing performance vs. π for different c values

### 3. Label Frequency Analysis
```bash
python scripts/analyze_phase3_label_freq.py
```
Generates curves showing performance vs. c for different π values

### 4. Method Comparison Report
```bash
python scripts/generate_phase3_report.py
```
Creates comprehensive markdown report with:
- Best method by (π, c) regime
- Failure mode analysis
- Prior mismatch quantification
- Mixup benefit analysis

## Success Criteria

Phase 3 is successful if it produces:

1. ✅ **Complete results:** All 2,744 experiments finish successfully
2. ✅ **Clear patterns:** Heatmaps reveal interpretable performance regions
3. ✅ **Actionable insights:** Identify when to use which VPU variant
4. ✅ **Robustness quantification:** Measure sensitivity to π and c
5. ✅ **Method selection guidance:** Recommend method based on (π, c) estimates

## Timeline

- **Setup:** 1 hour (configs and scripts already created)
- **Execution:** 15-20 hours (parallel workers)
- **Analysis:** 4-6 hours (generate plots and reports)
- **Writing:** 2-3 hours (summarize findings)

**Total:** ~1 day execution + 1 day analysis

## Configuration Files

All Phase 3 configs created in `config/phase3/`:
- `mnist_phase3.yaml`
- `fashionmnist_phase3.yaml`
- `imdb_phase3.yaml`
- `20news_phase3.yaml`
- `mushrooms_phase3.yaml`
- `spambase_phase3.yaml`
- `connect4_phase3.yaml`

Execution script: `scripts/run_phase3.sh`

## Date

Planned: 2026-04-20
