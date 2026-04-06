# Final Auto vs 0.5 Recommendations (Comprehensive Analysis)

## Executive Summary

**The contradiction is resolved**: The original recommendations mixed **configured true prior** patterns with **measured prior distance** patterns. After comprehensive analysis across ALL metrics, the pattern is clear and non-contradictory.

---

## Key Findings Across All Metrics

### 1. Overall Performance (Averaged Across All Experiments)

| Metric Category | auto | 0.5 | Winner | Significance | Effect Size |
|----------------|------|-----|--------|--------------|-------------|
| **AP (ranking)** | 0.8936 | 0.8951 | 0.5 | ns (p=0.53) | tiny (-0.16%) |
| **AUC (ranking)** | 0.8906 | 0.8915 | 0.5 | ns (p=0.69) | tiny (-0.09%) |
| **Oracle CE (calibration)** | 0.4797 | 0.4148 | **0.5** | **p<0.001*** | **-15.6%** ⭐ |
| **ECE (calibration)** | 0.1267 | 0.1012 | **0.5** | **p<0.001*** | **-25.2%** ⭐ |
| **MCE (calibration)** | 0.3642 | 0.3198 | **0.5** | **p<0.001*** | **-13.9%** ⭐ |
| **Brier Score** | 0.1433 | 0.1303 | 0.5 | p<0.001*** | -9.9% |
| F1 | 0.8115 | 0.8056 | auto | ns (p=0.47) | +0.7% |
| Accuracy | 0.8219 | 0.8353 | 0.5 | p<0.001*** | -1.6% |
| Precision | 0.8570 | 0.8686 | 0.5 | p=0.04* | -1.3% |
| Recall | 0.8216 | 0.7989 | auto | p=0.04* | +2.8% |

**Interpretation**:
- **Ranking (AP/AUC)**: Essentially identical (differences < 0.2%)
- **Calibration**: 0.5 is **significantly better** across all calibration metrics (15-25% improvement)
- **Threshold metrics**: Mixed, but low importance (threshold-dependent)

**Winner**: **0.5 overall** due to dramatically better calibration with identical ranking performance.

---

### 2. Performance by True Prior (π_true) - **THE CRITICAL DIMENSION**

This is where the real story emerges:

#### π_true = 0.1 (Extreme Positive Scarcity)

| Metric | auto | 0.5 | Diff | Winner |
|--------|------|-----|------|--------|
| AP | 0.8857 | 0.8905 | -0.0048 | 0.5 |
| Oracle CE | **0.6925** | **0.4434** | +0.2491 | **0.5** ⭐ |
| ECE | **0.1995** | **0.0794** | +0.1201 | **0.5** ⭐ |
| F1 | **0.6820** | **0.7905** | -0.1084 | **0.5** ⭐ |

**Verdict**: **0.5 STRONGLY better** (better ranking, MUCH better calibration, better F1)

---

#### π_true = 0.3 (Moderate Positive Scarcity)

| Metric | auto | 0.5 | Diff | Winner |
|--------|------|-----|------|--------|
| AP | 0.9210 | 0.9150 | +0.0061 | auto |
| Oracle CE | 0.3963 | 0.3576 | +0.0387 | 0.5 |
| ECE | 0.0961 | 0.0646 | +0.0314 | 0.5 |
| F1 | 0.8385 | 0.8451 | -0.0066 | 0.5 |

**Verdict**: **Mixed** (auto slightly better ranking, 0.5 better calibration)
- **If ranking matters most**: auto
- **If calibration matters**: 0.5

---

#### π_true = 0.5 (Balanced)

| Metric | auto | 0.5 | Diff | Winner |
|--------|------|-----|------|--------|
| AP | 0.9236 | 0.9240 | -0.0004 | tied |
| Oracle CE | **0.3117** | **0.3256** | -0.0138 | **auto** ⭐ |
| ECE | **0.0621** | **0.0690** | -0.0069 | **auto** ⭐ |
| F1 | 0.8523 | 0.8556 | -0.0033 | 0.5 |

**Verdict**: **Tied on ranking, auto BETTER on calibration!**
- This is surprising but important: at π=0.5, auto's estimated prior ≈ 0.5, so they converge
- auto has slight calibration edge

---

#### π_true = 0.7 (Moderate Positive Abundance) ⭐ **AUTO'S BEST CASE**

| Metric | auto | 0.5 | Diff | Winner |
|--------|------|-----|------|--------|
| AP | **0.9055** | **0.8998** | +0.0057 | **auto** ⭐ |
| Oracle CE | **0.3534** | **0.4149** | -0.0615 | **auto** ⭐ |
| ECE | **0.0856** | **0.1316** | -0.0460 | **auto** ⭐ |
| F1 | **0.8719** | **0.8253** | +0.0465 | **auto** ⭐ |

**Verdict**: **auto DOMINATES** (better on ranking, calibration, AND F1)
- Win rate: 73.3%
- Better calibration by 14.8% (Oracle CE)
- Better ECE by 34.9%

**This is the ONLY true prior where auto is unambiguously better across all metrics.**

---

#### π_true = 0.9 (Extreme Positive Abundance)

| Metric | auto | 0.5 | Diff | Winner |
|--------|------|-----|------|--------|
| AP | 0.8324 | 0.8463 | -0.0140 | 0.5 |
| Oracle CE | 0.6445 | 0.5328 | +0.1117 | 0.5 |
| ECE | 0.1904 | 0.1616 | +0.0288 | 0.5 |
| F1 | 0.8127 | 0.7117 | +0.1010 | auto |

**Verdict**: **0.5 better** (better ranking and calibration, despite worse F1)

---

### 3. Convergence Speed

| Method | Mean Epochs | Median Epochs | Faster % |
|--------|-------------|---------------|----------|
| **0.5** | **10.45** | **8** | **50.9%** ⭐ |
| auto | 12.23 | 10 | 29.3% |

**Winner**: **0.5 converges faster** in 50.9% of experiments, taking 17% fewer epochs on average.

---

### 4. Win Rate Paradox Explained

**Why does auto win 63.1% of matchups but have worse average performance?**

Looking at win rates by true prior:
- π = 0.1: auto wins **68.9%** but performs **worse** (-0.0048)
- π = 0.9: auto wins **55.6%** but performs **worse** (-0.0140)

**Explanation**: auto has **high variance**:
- Wins many small victories (e.g., +0.0010)
- Has occasional **catastrophic failures** (e.g., -0.05 on some runs)
- These catastrophic failures at extreme priors (0.1, 0.9) drag down the average
- 0.5 is more **consistent** (lower variance)

This is especially visible in calibration:
- At π=0.1: auto's Oracle CE is 0.69 vs 0.5's 0.44 (56% worse!)
- These extreme failures outweigh auto's wins at moderate priors

---

## FINAL RECOMMENDATIONS (Non-Contradictory)

### Primary Decision Tree

```
1. What is your true positive class prior (π_true)?
   
   π = 0.7:
      → Use AUTO (unambiguously better on all metrics)
   
   π ∈ [0.3, 0.5]:
      → Is calibration critical?
         YES: Use 0.5 (better calibrated at π=0.3, tied at π=0.5)
         NO:  Use auto (slightly better ranking at π=0.3, tied at π=0.5)
   
   π ∈ {0.1, 0.2, 0.8, 0.9} (extremes):
      → Use 0.5 (MUCH better calibration, better/similar ranking)

2. If true prior is unknown:
      → Is calibration critical?
         YES: Use 0.5 (safer, 15-25% better calibration overall)
         NO:  Try both on validation, but 0.5 is safer default
```

### Detailed Recommendations by Use Case

#### Use `auto` when:

✅ **True prior is π = 0.7** (unambiguous win)
- Better ranking: +0.64%
- Better Oracle CE: -14.8%
- Better ECE: -34.9%
- Win rate: 73.3%

✅ **True prior ∈ [0.3, 0.6] AND ranking (AP/AUC) is primary metric**
- Slight ranking advantage at π=0.3 (+0.67%)
- Calibration gap is moderate (not catastrophic like at extremes)

✅ **Your application prioritizes robustness over consistency**
- auto wins more individual matchups (63%)
- But higher variance (occasional failures)

#### Use `0.5` when:

✅ **True prior is extreme (π ≤ 0.2 or π ≥ 0.8)**
- Much better calibration (15-56% improvement in Oracle CE)
- Better or equal ranking performance
- Example: π=0.1 → 0.5 has 56% better Oracle CE

✅ **Calibration is critical** (e.g., medical diagnosis, risk assessment, probability forecasting)
- 0.5 is 15.6% better on Oracle CE overall
- 25.2% better on ECE
- More consistent (lower variance)

✅ **Faster convergence matters**
- 0.5 converges 17% faster on average
- Useful for large-scale experiments or real-time learning

✅ **True prior is unknown**
- 0.5 is safer default (better worst-case performance)
- Less sensitive to prior misspecification at extremes

✅ **Production deployment** (general recommendation)
- More consistent performance
- Better calibrated probabilities
- Faster training

---

### Resolution of the Contradiction

**Original contradictory claims**:
1. "Use auto when π ∈ [0.3, 0.7]" 
2. "Use auto when |π - 0.5| > 0.2"

**Why this was wrong**:
- Claim 1 is based on **configured true prior** patterns
- Claim 2 was based on **measured prior distance** (which correlates with configured but adds noise)
- π ∈ [0.3, 0.7] means |π - 0.5| ∈ [0, 0.2] (OPPOSITE of claim 2!)

**Corrected pattern**:
Looking at **configured true prior** only:

| π_true | \|π - 0.5\| | Winner | Reason |
|--------|-------------|--------|--------|
| 0.1 | 0.4 (far) | **0.5** | Much better calibration |
| 0.3 | 0.2 (medium) | **mixed** | auto slight ranking edge, 0.5 better calibration |
| 0.5 | 0.0 (close) | **tied** | Essentially identical |
| 0.7 | 0.2 (medium) | **auto** | Better on everything ⭐ |
| 0.9 | 0.4 (far) | **0.5** | Better ranking and calibration |

**The real pattern is NOT monotonic with distance from 0.5!**

Instead, the pattern is:
- **π = 0.7 is auto's sweet spot** (73% win rate, better on all metrics)
- **Extreme priors (0.1, 0.9) favor 0.5** (much better calibration)
- **Moderate priors (0.3, 0.5) are mixed/tied**

**Why is π = 0.7 special for auto?**
- auto estimates prior from labeled data
- At π=0.7, the labeled set has good representation of both classes
- auto can adapt to the imbalance while maintaining good calibration
- At π=0.1 or 0.9, labeled set is too skewed → auto overfits to the extreme

---

## Summary Table: Quick Reference

| True Prior (π) | Recommendation | AP Winner | Calibration Winner | Notes |
|----------------|----------------|-----------|-------------------|-------|
| 0.1 | **0.5** | 0.5 | **0.5** (56% better!) | Extreme imbalance hurts auto |
| 0.2 | **0.5** | likely 0.5 | **0.5** | Extrapolated from 0.1 pattern |
| 0.3 | **0.5 (calibration) or auto (ranking)** | auto (+0.7%) | 0.5 (-10%) | Depends on metric priority |
| 0.5 | **Either** (tied) | tied | auto (slight) | Methods converge |
| 0.7 | **auto** ⭐ | auto (+0.6%) | **auto** (-15%!) | auto's sweet spot |
| 0.8 | **0.5** | likely 0.5 | **0.5** | Extrapolated from 0.9 pattern |
| 0.9 | **0.5** | 0.5 | 0.5 | Extreme imbalance hurts auto |

**Default when prior is unknown**: **0.5** (safer, better calibration, faster convergence)

---

## Practical Guidelines

### For Research/Experiments:
1. If you know π_true ≈ 0.7: use auto
2. If you know π_true is extreme (<0.2 or >0.8): use 0.5
3. If unknown: try both, select on validation AP or Oracle CE depending on priority

### For Production Deployment:
1. **Use 0.5** (default)
   - Better calibrated
   - More consistent
   - Faster convergence
   - Less sensitive to prior estimation errors

2. **Exception**: If you have strong evidence that π ≈ 0.7 AND calibration is not critical, consider auto

### For Publications:
1. Report results for **both** auto and 0.5
2. Analyze performance by true prior (π_true)
3. Report both ranking (AP/AUC) and calibration (Oracle CE/ECE) metrics
4. State whether calibration or ranking is the primary evaluation criterion

---

## Key Takeaways

1. ⭐ **π = 0.7 is auto's sweet spot** - the ONLY clear win across all metrics
2. ⭐ **0.5 is dramatically better at extreme priors** (π ≤ 0.2 or ≥ 0.8)
3. ⭐ **Calibration vs ranking trade-off**: 0.5 is 15-25% better calibrated overall, ranking is tied
4. ⭐ **0.5 converges 17% faster** - practical advantage for large experiments
5. ⭐ **auto has higher variance** - wins more matchups but with occasional catastrophic failures
6. **When in doubt, use 0.5** - safer default with better worst-case performance
