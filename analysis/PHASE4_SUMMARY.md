# Phase 4 Summary: Optimal Method Prior for PU Learning

## Experiment Setup

- **Methods**: VPU-MP (with mixup), VPU-nomix-MP (without mixup)
- **Datasets**: 7 (MNIST, FashionMNIST, IMDB, 20News, Connect4, Mushrooms, Spambase)
- **Seeds**: 10
- **Label frequencies (c)**: 3 [0.01, 0.5, 0.99]
- **True priors (π)**: 7 [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
- **Method prior values**: 14 [auto, true, (ep+1)/3, 0.01, 0.1111, ..., 0.99]
- **Total unique runs**: 41,160 (evenly split: 4,116 per seed)

## Key Finding: The Optimal Prior is a Constant

The optimal method_prior does **not** depend on the effective prior (c × π).
Beta-distribution fits to metric(method_prior) across 18 effective prior values
(R² > 0.9 for AUC, R² > 0.97 for calibration metrics) yield best-fit lines with
near-zero slopes:

| Metric | Best-Fit Line | Slope | Intercept |
|--------|---------------|-------|-----------|
| AUC ↑ | 0.06·ep + 0.51 | ~0 | ~0.5 |
| AP ↑ | 0.07·ep + 0.59 | ~0 | ~0.6 |
| Brier ↓ | -0.02·ep + 0.67 | ~0 | ~0.67 |
| Oracle CE ↓ | -0.04·ep + 0.60 | ~0 | ~0.6 |
| ECE ↓ | -0.14·ep + 0.69 | ~0 | ~0.7 |
| A-NICE ↓ | -0.06·ep + 0.88 | ~0 | ~0.85 |
| S-NICE ↓ | -0.03·ep + 0.86 | ~0 | ~0.85 |

**A fixed constant outperforms data-dependent priors** ("auto" = c × π, "true" = π)
across all metrics and effective prior ranges.

## The Optimal Constant Depends on What You Optimize

| Metric Family | Optimal Prior | Why |
|---------------|---------------|-----|
| **Ranking** (AUC, AP, Max F1) | ~0.5 | Ranking only needs correct ordering, not calibrated probabilities. A balanced prior gives the purest ranking signal. |
| **Proper scoring** (Brier, Oracle CE) | ~0.6–0.67 | These reward both ranking AND calibration. PU learning has a systematic negative bias (hidden positives treated as negatives), and a prior of ~0.6 corrects this bias. |
| **Neighborhood calibration** (A-NICE, S-NICE) | ~0.85 | NICE metrics are sensitive to the bimodal probability distribution that PU methods produce. A high prior compresses this bimodality. |

**Recommended default: 0.6667 (2/3)** — it wins the uniform-weighted ranking across
all 11 metrics (avg rank 3.55), balancing ranking performance (where it's competitive
with 0.5) against calibration (where it clearly dominates).

## Bias Correction: Uniform Effective-Prior Weighting

The experimental grid overrepresents low effective priors (57% of data has ep < 0.2).
Naive averaging biases toward priors that work well at low ep (e.g., 0.5), hiding the
advantage of higher priors at mid-to-high ep.

**Correction**: Bin by effective prior, compute mean within each bin, average across
bins with equal weight.

| method_prior | Naive Rank | Uniform Rank |
|--------------|------------|--------------|
| (ep+1)/3 | 1 | 3 |
| 0.5000 | 2 | 5 |
| **0.6667** | 5 | **1** |

The bias correction changes the winner from 0.5 (naive) to **0.6667** (uniform).

## Mixup vs No-Mixup

Head-to-head across 20,580 paired configurations:

| Metric | Mixup win% | No-mixup win% |
|--------|------------|----------------|
| A-NICE ↓ | **72.1%** | 25.8% |
| S-NICE ↓ | **71.8%** | 26.1% |
| Brier ↓ | **61.9%** | 36.1% |
| Oracle CE ↓ | **59.9%** | 38.1% |
| AUC ↑ | 48.6% | 49.3% |

**Mixup is a calibration enhancer** (60–72% win rate on calibration metrics) while
being neutral on ranking. Recommended: **use mixup**.

## Per-Dataset Variation

The optimal prior varies across datasets (intercepts of per-dataset best-fit lines):

| Dataset | Type | AUC intercept | Oracle CE intercept |
|---------|------|---------------|---------------------|
| MNIST | Image | 0.77 | 0.78 |
| FashionMNIST | Image | 0.70 | 0.78 |
| Connect4 | Tabular | 0.49 | 0.73 |
| Mushrooms | Tabular | 0.74 | 0.77 |
| 20News | Text | 0.64 | 0.53 |
| IMDB | Text | 0.55 | 0.44 |
| Spambase | Tabular | 0.43 | 0.45 |

**Image datasets prefer higher priors (~0.75), text datasets prefer lower (~0.50).**

### Why 0.6667 Is Still Good Enough for Low-Prior Datasets

Two datasets (IMDB, Spambase) achieve best Oracle CE at lower priors (0.33–0.44).
However, the regret from using 0.6667 instead is **asymmetrically small**:

| Dataset | Best Prior | CE @ Best | CE @ 0.67 | Regret |
|---------|-----------|-----------|-----------|--------|
| 20News | 0.6667 | 0.554 | 0.554 | 0.0% |
| Connect4 | 0.6667 | 0.376 | 0.376 | 0.0% |
| FashionMNIST | 0.6667 | 0.258 | 0.258 | 0.0% |
| MNIST | 0.6667 | 0.298 | 0.298 | 0.0% |
| Mushrooms | 0.6667 | 0.471 | 0.471 | 0.0% |
| IMDB | 0.3333 | 0.611 | 0.725 | +18.6% |
| Spambase | 0.4444 | 0.528 | 0.603 | +14.2% |

**The reverse regret is far worse:** Using 0.3333 as default would cost **+31% Oracle CE**
on the 4 datasets where 0.6667 is optimal, to save ~15% on 2 datasets.

Furthermore, on IMDB and Spambase the Oracle CE curve is flat across the 0.33–0.56
range — the difference between 0.33 and 0.55 is only ~5%. The steep degradation
only begins above 0.6. This means:

1. **0.6 is a viable compromise** — it sits at the edge of the flat region for
   IMDB/Spambase while staying near-optimal for the other 5 datasets.
2. **The 0.33–0.67 range is a plateau**, not a cliff. Performance degrades
   gracefully, unlike the extremes (0.01 or 0.99) which fail catastrophically.
3. **A practitioner using 0.6667 on IMDB loses ~19% on calibration but gains
   access to a prior that works well on 5/7 datasets without tuning.** This is
   the Stein's paradox argument: shrinking toward a common constant beats
   per-dataset optimization in aggregate.

## Theoretical Explanation

| Framework | Explains |
|-----------|----------|
| **Stein's Paradox** | A fixed constant outperforms per-configuration estimation (auto/true) when aggregating across ≥3 settings |
| **Implicit Regularization** | PU learning's negative bias (hidden positives → negative labels) requires a correction toward higher priors |
| **Proper Scoring Decomposition** | Different metrics weight ranking vs calibration differently, explaining the metric-dependent optimal |
| **Minimax Regret** | The optimal constant minimizes worst-case loss across effective priors, explaining the independence from ep |

## Practical Recommendations

1. **Default**: Use method_prior = **0.6667 (2/3)** with mixup enabled
2. **If ranking is all you care about**: method_prior = **0.5** is marginally better for AUC/AP
3. **If you know your dataset type**:
   - Image data: 0.6667–0.77 (higher priors work well)
   - Text data: 0.5–0.6 (lower priors are safer)
   - Tabular: 0.6667 is a reasonable default but varies
4. **Never use "auto" (c × π)** — it's dominated by fixed constants across all metrics
5. **Always use mixup** — neutral for ranking, substantially better for calibration
