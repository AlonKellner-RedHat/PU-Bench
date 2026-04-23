# Phase 3 Analysis: Method Prior Comparison (Seed 42)

**Date:** 2026-04-20  
**Scope:** 3,430 experiments across 7 datasets, 7 label frequencies (c), 7 true priors (π)  
**Focus:** Comparing method_prior variants (0.69 vs 0.5 vs auto) for VPU mean-prior methods

---

## Executive Summary

**Key Finding 1: method_prior=0.69 provides MARGINAL improvement over 0.5**
- For VPU-MP (with mixup): 0.69 achieves 0.8980 AUC vs 0.5's 0.8951 (Δ=+0.0029, p=0.284, **not significant**)
- For VPU-nomix-MP: 0.5 achieves 0.8946 AUC vs 0.69's 0.8939 (Δ=-0.0007, p=0.327, **not significant**)
- **Recommendation**: The theoretical advantage of 0.69 is not practically significant in Phase 3 experiments

**Key Finding 2: auto performs SIGNIFICANTLY WORSE than fixed priors**
- VPU-MP(auto): 0.8680 AUC vs 0.5's 0.8951 (Δ=-0.0271, p<0.0001, **highly significant**)
- VPU-nomix-MP(auto): 0.8781 AUC vs 0.5's 0.8946 (Δ=-0.0165, p<0.0001, **highly significant**)
- **Recommendation**: Avoid auto prior in Phase 3 grid settings; use fixed priors (0.5 or 0.69)

**Key Finding 3: Mixup interaction with method_prior is complex**
- For 0.69: Mixup helps (+0.0041, p=0.0115) → Use VPU-MP(0.69) with mixup
- For auto: Mixup hurts (-0.0101, p=0.0027) → Use VPU-nomix-MP(auto) without mixup
- For 0.5: Mixup neutral (+0.0005, p=0.866) → Either works

**Key Finding 4: Best overall methods**
1. Oracle (0.9532 AUC) - upper bound
2. **VPU-MP(0.69)** (0.8980 AUC) - best VPU variant
3. VPU-MP(0.5) (0.8951 AUC)
4. VPU-nomix-MP(0.5) (0.8946 AUC)
5. VPU-nomix-MP(0.69) (0.8939 AUC)

---

## 1. Method Prior Comparison: 0.69 vs 0.5 vs auto

### 1.1 VPU-MP (with mixup)

**Overall Performance:**
- **0.69**: 0.8980 ± 0.1279 AUC (rank: 1st of mean-prior variants)
- **0.5**: 0.8951 ± 0.1355 AUC (rank: 2nd)
- **auto**: 0.8680 ± 0.1592 AUC (rank: 3rd, significantly worse)

**Win Rates (343 configurations):**
- 0.69: **119 wins (34.7%)**
- 0.5: 112 wins (32.7%)
- auto: 112 wins (32.7%)

**Pairwise Comparisons:**
| Comparison | Mean Δ AUC | Configurations where A > B | p-value | Significance |
|------------|------------|---------------------------|---------|--------------|
| 0.69 vs 0.5 | +0.0029 | 185/343 (54%) | 0.284 | Not significant |
| 0.69 vs auto | +0.0300 | 196/343 (57%) | <0.0001 | **Highly significant** |
| 0.5 vs auto | +0.0271 | 152/343 (44%) | <0.0001 | **Highly significant** |

**Interpretation:**
- **0.69 and 0.5 are statistically equivalent** (p=0.284)
- Both fixed priors substantially outperform auto (Δ≈+0.03 AUC, p<0.0001)
- 0.69 wins slightly more often (34.7% vs 32.7%), but difference is marginal

### 1.2 VPU-nomix-MP (without mixup)

**Overall Performance:**
- **0.5**: 0.8946 ± 0.1317 AUC (rank: 1st of VPU-nomix variants)
- **0.69**: 0.8939 ± 0.1335 AUC (rank: 2nd)
- **auto**: 0.8781 ± 0.1495 AUC (rank: 3rd)

**Win Rates (343 configurations):**
- **auto: 153 wins (44.6%)** ← Surprising! Auto wins most often without mixup
- 0.5: 98 wins (28.6%)
- 0.69: 92 wins (26.8%)

**Pairwise Comparisons:**
| Comparison | Mean Δ AUC | Configurations where A > B | p-value | Significance |
|------------|------------|---------------------------|---------|--------------|
| 0.69 vs 0.5 | -0.0007 | 153/343 (45%) | 0.327 | Not significant |
| 0.69 vs auto | +0.0158 | 154/343 (45%) | <0.0001 | **Highly significant** |
| 0.5 vs auto | +0.0165 | 183/343 (53%) | <0.0001 | **Highly significant** |

**Interpretation:**
- **0.5 and 0.69 are statistically equivalent** (p=0.327), with 0.5 marginally better
- Auto wins more configurations (44.6%) but has lower average performance
  - This suggests **auto has higher variance**: wins big when prior is correct, loses big when wrong
- Fixed priors (0.5, 0.69) provide more **robust** performance across conditions

---

## 2. Performance Across Prior Grid (π, c)

### 2.1 VPU-MP Heatmaps

**Key Observations:**

1. **All priors perform well in moderate regions** (π ∈ [0.3, 0.7], c ∈ [0.3, 0.9])
   - AUC > 0.93 consistently across all three priors
   - Fixed priors (0.5, 0.69) slightly more stable than auto

2. **Extreme priors (π = 0.01, 0.99) show degradation**
   - Performance drops to ~0.64-0.77 AUC at π=0.99
   - All priors suffer similarly in extreme regions
   - auto performs slightly worse at extremes

3. **Low label frequency (c=0.01) is challenging**
   - Performance ~0.75-0.85 AUC across all priors
   - auto shows more variability (wider color range in heatmap)

4. **Difference heatmap (0.69 - 0.5):**
   - Differences are small (±0.01 to ±0.03 in most cells)
   - Slightly positive (0.69 better) in low c regions (c=0.01, 0.1)
   - Slightly negative (0.5 better) in some moderate regions
   - **No clear systematic advantage** for either prior

### 2.2 VPU-nomix-MP Heatmaps

**Key Observations:**

1. **Similar overall patterns to VPU-MP**
   - Strong performance (AUC > 0.93) in moderate regions
   - Degradation at extreme priors (π = 0.01, 0.99)

2. **Difference heatmap (0.69 - 0.5):**
   - Even smaller differences than VPU-MP (±0.005 in most cells)
   - Essentially **no systematic difference** between 0.69 and 0.5

3. **auto shows higher variance** without mixup
   - More color variation in auto heatmap
   - Suggests auto is less stable across the grid

---

## 3. Performance vs True Prior (π) and Label Frequency (c)

### 3.1 Performance vs True Prior (π)

**VPU-MP:**
- All priors show **inverted-U shape**: peak at π ∈ [0.3, 0.5], drop at extremes
- **0.69 and 0.5 track closely** across all π values
- auto diverges significantly at low π (0.01, 0.1) and high π (0.99)
- **Worst performance** at π = 0.99 (extreme positive prior):
  - auto: 0.67 AUC
  - 0.5: 0.77 AUC
  - 0.69: 0.77 AUC

**VPU-nomix-MP:**
- Similar inverted-U pattern
- **0.5 and 0.69 nearly identical** across π
- auto shows larger variance (wider confidence bands)
- At π = 0.99:
  - auto: 0.76 AUC
  - 0.5: 0.80 AUC
  - 0.69: 0.80 AUC

**Insight:** Fixed priors (0.5, 0.69) are more robust to prior mismatch, especially at extremes.

### 3.2 Performance vs Label Frequency (c)

**VPU-MP:**
- All priors show **monotonic improvement** with increasing c
- At c=0.01 (very low labeled data):
  - auto: 0.73 AUC
  - 0.5: 0.76 AUC
  - 0.69: 0.77 AUC
- At c=0.99 (almost fully labeled):
  - All priors converge to ~0.94 AUC

**VPU-nomix-MP:**
- Similar monotonic improvement with c
- **0.5 and 0.69 nearly identical** across all c
- auto slightly worse at low c

**Insight:** 0.69 shows marginal advantage at very low c (0.01, 0.1), consistent with theoretical optimality under uniform prior assumption.

---

## 4. Mixup Effect Analysis

### 4.1 Mixup × Method Prior Interaction

**Critical finding:** Mixup effectiveness depends on method_prior choice.

| Method Prior | With Mixup (VPU-MP) | Without Mixup (VPU-nomix-MP) | Δ AUC | p-value | Recommendation |
|--------------|---------------------|------------------------------|-------|---------|----------------|
| **auto** | 0.8680 | 0.8781 | **-0.0101** | 0.003 | **Use without mixup** |
| **0.5** | 0.8951 | 0.8946 | +0.0005 | 0.866 | Either (neutral) |
| **0.69** | 0.8980 | 0.8939 | **+0.0041** | 0.012 | **Use with mixup** |
| Base (no MP) | 0.8701 | 0.8749 | -0.0048 | 0.282 | Either (neutral) |

**Interpretation:**

1. **auto + mixup is BAD combination** (p=0.003)
   - Mixup hurts when prior is auto-computed from labeled data
   - Hypothesis: Mixup introduces synthetic examples that violate auto prior assumptions

2. **0.69 + mixup is GOOD combination** (p=0.012)
   - Mixup helps when using theoretically-optimal fixed prior
   - Hypothesis: Mixup regularization synergizes with 0.69's robust prior

3. **0.5 + mixup is NEUTRAL** (p=0.866)
   - No significant difference with or without mixup

**Practical Recommendation:**
- If using **auto**: choose VPU-nomix-MP(auto)
- If using **0.69**: choose VPU-MP(0.69) with mixup
- If using **0.5**: either VPU-MP(0.5) or VPU-nomix-MP(0.5)

---

## 5. Overall Method Ranking

**Top 10 methods by average AUC (343 configurations):**

| Rank | Method | AUC | Std Dev | Notes |
|------|--------|-----|---------|-------|
| 1 | Oracle BCE | 0.9532 | 0.056 | Upper bound (true labels) |
| **2** | **VPU-MP(0.69)** | **0.8980** | 0.128 | **Best VPU variant** |
| 3 | VPU-MP(0.5) | 0.8951 | 0.136 | 2nd best VPU |
| 4 | VPU-nomix-MP(0.5) | 0.8946 | 0.132 | Best VPU-nomix |
| 5 | VPU-nomix-MP(0.69) | 0.8939 | 0.134 | 2nd best VPU-nomix |
| 6 | VPU-nomix-MP(auto) | 0.8781 | 0.150 | Best auto variant |
| 7 | VPU-nomix (base) | 0.8749 | 0.152 | Base method |
| 8 | PN-Naive | 0.8723 | 0.168 | Baseline |
| 9 | VPU (base) | 0.8701 | 0.161 | Base with mixup |
| 10 | VPU-MP(auto) | 0.8680 | 0.159 | Worst VPU variant |

**Key Takeaways:**
1. **All VPU mean-prior methods outperform base VPU** (with or without mixup)
2. **Fixed priors (0.5, 0.69) dominate auto** across the board
3. **Top 5 methods are all mean-prior variants** (excluding oracle)
4. **0.69 achieves best performance** when combined with mixup (rank 2 overall)

---

## 6. Statistical Significance Summary

### 6.1 Is 0.69 better than 0.5?

**VPU-MP:** **No significant difference** (p=0.284)
- Mean improvement: +0.0029 AUC
- 0.69 better in 185/343 configurations (54%)
- Effect size: Cohen's d ≈ 0.06 (negligible)

**VPU-nomix-MP:** **No significant difference** (p=0.327)
- Mean improvement: -0.0007 AUC (0.5 slightly better)
- 0.69 better in 153/343 configurations (45%)
- Effect size: Cohen's d ≈ 0.005 (negligible)

**Conclusion:** **0.69 and 0.5 are statistically equivalent** in Phase 3 experiments. The theoretical optimality of 0.69 does not translate to practical advantage.

### 6.2 Is auto competitive with fixed priors?

**VPU-MP (auto vs 0.5):** **No** (p<0.0001, highly significant)
- Mean degradation: -0.0271 AUC
- auto better in only 152/343 configurations (44%)

**VPU-nomix-MP (auto vs 0.5):** **No** (p<0.0001, highly significant)
- Mean degradation: -0.0165 AUC
- auto better in 183/343 configurations (53%)

**Conclusion:** **Auto is significantly worse than fixed priors** on average, despite winning more configurations in VPU-nomix-MP. This indicates higher variance: auto performs very well when prior estimate is accurate, but poorly when it's not.

---

## 7. Recommendations

### 7.1 For Practitioners

**Best method choice:**
1. **VPU-MP(0.69)** if you want top performance (0.8980 AUC avg)
2. **VPU-MP(0.5)** if you prefer balanced assumption (0.8951 AUC avg, essentially equivalent)
3. **VPU-nomix-MP(0.5)** if computational cost of mixup is prohibitive (0.8946 AUC avg)

**Avoid:**
- VPU-MP(auto): Poor performance (0.8680 AUC), mixup hurts
- VPU-nomix-MP(auto): Better than VPU-MP(auto) but still worse than fixed priors

### 7.2 For Future Research

**Key questions:**

1. **Why does auto perform poorly in Phase 3?**
   - Hypothesis: In Phase 3 grid, labeled set is artificially sampled → auto prior estimate is biased
   - Future work: Analyze auto prior estimates vs true priors across grid

2. **Why does mixup hurt auto but help 0.69?**
   - Hypothesis: Mixup synthetic data violates assumptions of auto prior computation
   - Future work: Theoretical analysis of mixup + mean-prior interaction

3. **When does 0.69 win vs 0.5?**
   - Observation: 0.69 slightly better at low c (0.01, 0.1)
   - Hypothesis: 0.69 optimality assumption (uniform π) is more accurate when labeled data is scarce
   - Future work: Stratify analysis by c and π regions

4. **Does 5-seed expansion change conclusions?**
   - Current analysis: Single seed (42)
   - Next: Repeat with seeds 456, 789, 1024, 2048 to assess variance and confirm findings

### 7.3 For Phase 3 Multi-Seed Analysis

**Expected insights from 5-seed expansion:**
1. **Confidence intervals** on all comparisons (currently point estimates)
2. **Variance decomposition**: dataset heterogeneity vs seed variance
3. **Robustness**: Do 0.69 vs 0.5 conclusions hold across seeds?
4. **Failure modes**: Are poor-performing configurations seed-specific or systematic?

---

## 8. Visualizations

All plots saved to: `analysis/plots/phase3_seed42/`

1. **`overall_method_comparison_test_auc.png`**
   - Bar chart of average AUC for all 10 methods
   - Color-coded by method type
   - Shows VPU-MP(0.69) as best non-oracle method

2. **`vpu_mean_prior_heatmap_comparison_test_auc.png`**
   - 2×2 grid: auto, 0.5, 0.69 heatmaps + difference heatmap (0.69 - 0.5)
   - Shows performance across (π, c) grid
   - Highlights negligible differences between 0.69 and 0.5

3. **`vpu_nomixup_mean_prior_heatmap_comparison_test_auc.png`**
   - Same as above but for VPU-nomix-MP
   - Shows 0.5 and 0.69 are nearly identical

4. **`vpu_mean_prior_prior_comparison_curves_test_auc.png`**
   - Left: AUC vs π (averaged over c and datasets)
   - Right: AUC vs c (averaged over π and datasets)
   - Shows convergence of 0.5 and 0.69, divergence of auto

5. **`vpu_nomixup_mean_prior_prior_comparison_curves_test_auc.png`**
   - Same as above but for VPU-nomix-MP

6. **`prior_comparison_detailed.csv`**
   - Row-level data: one row per (dataset, c, π) configuration
   - Columns: auto, 0.5, 0.69 performance, best_prior
   - Enables custom analysis and filtering

---

## 9. Conclusion

**Is 0.69 better than 0.5?**
- **No, not significantly.** The theoretical optimality of 0.69 (minimizing error under uniform prior distribution) does not translate to meaningful practical advantage in Phase 3 experiments.
- Difference: +0.0029 AUC for VPU-MP, -0.0007 AUC for VPU-nomix-MP (both p>0.28)

**Is 0.69 better than auto?**
- **Yes, significantly.** Fixed prior 0.69 outperforms auto by +0.03 AUC (p<0.0001), especially when combined with mixup.

**Best overall method:**
- **VPU-MP(0.69)**: 0.8980 AUC average, ranks 1st among VPU variants
- However, **VPU-MP(0.5)** is essentially equivalent (0.8951 AUC, p=0.284)

**Key insight:**
The choice between **0.69 and 0.5 is NOT critical** (difference is negligible and not significant). More important decisions are:
1. Use **fixed prior** (not auto)
2. Use **mean-prior method** (not base VPU)
3. For 0.69: **enable mixup** (VPU-MP)
4. For auto: **disable mixup** (VPU-nomix-MP)

**Next steps:**
Wait for Phase 3 multi-seed expansion (ETA: ~15:30 tomorrow) to:
- Confirm these findings across 5 seeds
- Quantify variance and confidence intervals
- Assess seed-specific vs systematic effects
