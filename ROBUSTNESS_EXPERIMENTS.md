# Prior Robustness Experiments for VPU-NoMixup-Mean-Prior

## Overview

This document describes the experimental setup for testing the robustness of `vpu_nomixup_mean_prior` to prior (class prevalence) misspecification.

**Research Question**: How does `vpu_nomixup_mean_prior` perform when the prior (π) passed to the method differs from the true class prevalence in the training data?

## Experimental Design

### Independent Variables

1. **Dataset** (6 levels): MNIST, FashionMNIST, IMDB, 20News, Mushrooms, Spambase
2. **Label Frequency (c)** (3 levels): 0.1, 0.5, 0.9
3. **Random Seed** (3 levels): 42, 456, 789
4. **Method Prior** (7 levels): 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, "auto"
   - "auto" uses the true prior computed from training data (baseline)
5. **Method** (3 variants):
   - `vpu_nomixup`: Baseline without prior
   - `vpu_nomixup_mean`: With mean but no prior
   - `vpu_nomixup_mean_prior`: Full method with prior (focus of robustness study)

### Total Experiments

- Base configurations: 6 datasets × 3 seeds × 3 c-values = 54
- Method runs per configuration:
  - vpu_nomixup: 1 (no prior)
  - vpu_nomixup_mean: 1 (no prior)
  - vpu_nomixup_mean_prior: 7 (6 fixed priors + 1 auto)
- **Total method runs: 54 × 9 = 486**

### Data Isolation

**Critical**: Robustness experiments are completely isolated from the original 1,260 experiments.

- Original experiments: `results/` directory
  - Contains main VPU analysis with 1,260 experiments
  - Original analysis scripts read from here only
  - **NOT affected** by robustness experiments

- Robustness experiments: `results_robustness/` directory
  - New experiments testing prior misspecification
  - Dedicated analysis scripts
  - Completely independent from `results/`

## Implementation

### Code Modifications

**1. `train/vpu_nomixup_mean_prior_trainer.py`**
- Modified `create_criterion()` to use `method_prior` if specified in config
- Falls back to computed prior from training data if not specified

**2. `train/base_trainer.py`**
- Extracts `method_prior` from params
- Stores both `true_prior` (from training data) and `prior` (used by method)
- Uses `output_dir` from params for results storage

**3. `run_train.py`**
- Added `--output-dir` CLI parameter (default: "results")
- Expands `method_prior_values` grid from config
- Updates experiment naming to include `methodprior{value}`
- Passes method_prior to trainer

### Configuration Files

Location: `config/vpu_rerun/robustness/*.yaml`

Each config specifies:
```yaml
method_prior_values: [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, "auto"]
target_prevalence: [null]  # Keep test set at natural prevalence
```

Files:
- `mnist_robustness.yaml`
- `fashionmnist_robustness.yaml`
- `imdb_robustness.yaml` (with cached SBERT embeddings)
- `20news_robustness.yaml` (with cached SBERT embeddings)
- `mushrooms_robustness.yaml`
- `spambase_robustness.yaml`

## Running Experiments

### Full Robustness Suite

```bash
# Run all 486 experiments (~1.5-2 hours with 3 parallel workers)
bash scripts/run_prior_robustness.sh
```

This script:
- Runs 6 datasets in parallel (3 workers via GNU parallel)
- Saves results to `results_robustness/`
- Logs to `logs/prior_robustness/`

### Single Dataset Test

```bash
# Test with one dataset
python run_train.py \
    --dataset-config config/vpu_rerun/robustness/mnist_robustness.yaml \
    --methods vpu_nomixup_mean_prior \
    --output-dir results_robustness
```

### Resume Interrupted Runs

```bash
# Automatically skip completed experiments
python run_train.py \
    --dataset-config config/vpu_rerun/robustness/mnist_robustness.yaml \
    --methods vpu_nomixup vpu_nomixup_mean vpu_nomixup_mean_prior \
    --output-dir results_robustness \
    --resume
```

## Analysis

### 1. Extract and Analyze Results

```bash
python analysis/analyze_prior_robustness.py
```

This script:
- Loads all JSON files from `results_robustness/`
- Calculates prior error: |method_prior - π_true|
- Computes performance degradation vs baseline (auto prior)
- Saves:
  - `results_robustness/robustness_full_results.csv` - all experiments
  - `results_robustness/robustness_degradation.csv` - degradation metrics

### 2. Generate Visualizations

```bash
python analysis/plot_prior_robustness.py
```

Generates plots in `results_robustness/plots/`:

1. **robustness_curves_all_datasets.png**
   - Performance vs prior error for each dataset
   - Shows all 3 methods
   - Error bars show cross-seed variance

2. **degradation_heatmap.png**
   - Dataset × prior_error heatmap
   - Color-coded relative F1 drop (%)

3. **label_frequency_interaction.png**
   - Tests if label frequency (c) affects prior sensitivity
   - Separate curves for c=0.1, 0.5, 0.9

4. **degradation_by_error_bins.png**
   - Bar plot showing mean degradation at different error levels
   - Error bins: 0-0.1, 0.1-0.2, 0.2-0.3, 0.3-0.5, 0.5+

5. **method_comparison_by_error.png**
   - Compares all 3 methods at different error levels
   - Shows advantage of prior-based method even with errors

6. **robustness_summary_table.csv** + LaTeX table
   - Publication-ready summary statistics

### 3. Validation

```bash
python scripts/validate_robustness_results.py
```

Checks:
- All expected experiments completed (486)
- All JSON files have `method_prior` and `true_prior` fields
- No contamination in `results/` directory
- Prior error distribution

## Expected Results

### Hypothesis 1: Graceful Degradation

Performance drops slowly as prior error increases:
- Error ≤ 0.1: < 5% performance drop
- Error ≤ 0.2: < 10% performance drop
- Error ≤ 0.5: < 20% performance drop

**Interpretation**: If true, method is practical with approximate priors.

### Hypothesis 2: Better Than No-Prior Variants

Even with moderate prior error (≤ 0.2), `vpu_nomixup_mean_prior` still outperforms `vpu_nomixup` and `vpu_nomixup_mean`.

**Interpretation**: Validates using approximate priors over no priors.

### Hypothesis 3: Dataset-Specific Robustness

Some datasets more sensitive than others:
- Text datasets (IMDB, 20News): More robust
- Tabular (Mushrooms, Spambase): Less robust

**Interpretation**: Guides prior estimation effort based on dataset characteristics.

## File Structure

```
PU-Bench/
├── config/vpu_rerun/robustness/
│   ├── mnist_robustness.yaml
│   ├── fashionmnist_robustness.yaml
│   ├── imdb_robustness.yaml
│   ├── 20news_robustness.yaml
│   ├── mushrooms_robustness.yaml
│   └── spambase_robustness.yaml
├── scripts/
│   ├── run_prior_robustness.sh          # Main execution script
│   ├── validate_robustness_results.py   # Validation
│   └── test_robustness_setup.sh         # Quick test
├── analysis/
│   ├── analyze_prior_robustness.py      # Extract & analyze
│   └── plot_prior_robustness.py         # Visualizations
├── results_robustness/                  # Isolated results directory
│   ├── seed_42/
│   ├── seed_456/
│   ├── seed_789/
│   ├── robustness_full_results.csv
│   ├── robustness_degradation.csv
│   └── plots/
└── ROBUSTNESS_EXPERIMENTS.md            # This file
```

## Troubleshooting

### No results found

Check if experiments have run:
```bash
ls -la results_robustness/seed_*/
```

If empty, run:
```bash
bash scripts/run_prior_robustness.sh
```

### Data contamination warning

If validation detects robustness files in `results/`:
```bash
# This should NOT happen with proper --output-dir usage
# If it does, clean up:
rm results/seed_*/*methodprior*.json
```

### Incomplete experiments

Check progress:
```bash
python scripts/validate_robustness_results.py
```

Resume:
```bash
bash scripts/run_prior_robustness.sh  # Uses --resume by default
```

### Analysis scripts fail

Ensure analysis has been run first:
```bash
python analysis/analyze_prior_robustness.py  # Must run before plotting
python analysis/plot_prior_robustness.py
```

## Timeline

- **Development**: 2-3 hours (already complete)
- **Experiment Execution**: 1.5-2 hours (parallel, 3 workers)
- **Analysis**: 10-15 minutes
- **Total**: ~2-3 hours end-to-end

## References

See main paper results in `results/vpu_nomixup_variants_comparison.csv` for the original performance analysis that motivated these robustness experiments.
