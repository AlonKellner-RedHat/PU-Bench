# Multi-Seed Statistical Analysis Reports

This directory contains comprehensive statistical analysis of VPU vs VPU-Mean methods across 5 random seeds (42, 123, 456, 789, 2024) with 1,812 total experiments.

## 📄 Analysis Reports

### 🎯 Start Here
- **[ANALYSIS_EXECUTIVE_SUMMARY.md](ANALYSIS_EXECUTIVE_SUMMARY.md)** - Quick overview and decision guide

### 📊 Detailed Reports
1. **[MULTI_SEED_ANALYSIS_SUMMARY.md](MULTI_SEED_ANALYSIS_SUMMARY.md)** - F1-focused analysis with MixUp
   - Performance by dataset, c, and prior
   - Paired t-tests across seeds
   - Confidence assessment
   - Win rates and consistency analysis

2. **[COMPREHENSIVE_METRICS_ANALYSIS.md](COMPREHENSIVE_METRICS_ANALYSIS.md)** - All metrics including calibration
   - AUC, precision, recall analysis
   - Calibration metrics (A-NICE, ECE, MCE, S-NICE, Brier)
   - Performance-calibration tradeoff
   - Dataset-specific patterns

3. **[NOMIXUP_ANALYSIS_SUMMARY.md](NOMIXUP_ANALYSIS_SUMMARY.md)** - Analysis without MixUp
   - How results change without augmentation
   - MixUp dependency analysis
   - Clear recommendations for no-MixUp scenario

4. **[../results/PRIOR_VARIANTS_SUMMARY.md](../results/PRIOR_VARIANTS_SUMMARY.md)** - Prior-weighted variants analysis
   - VPU-Mean-Prior vs VPU-Mean
   - VPU-NoMixUp-Mean-Prior vs VPU-NoMixUp-Mean
   - Impact of π·E_P[log(φ(x))] term on performance and calibration
   - **Conclusion: Prior weighting hurts performance, do not use**

## 🔑 Key Findings

### With MixUp Available
- **VPU and VPU-Mean are statistically comparable** overall (p=0.147)
- **VPU-Mean:** Better recall (+5.1%, p<0.001), better at low c
- **VPU:** Better calibration (all metrics p<0.01), better at low prior
- **Choose based on application needs** (performance vs calibration)

### Without MixUp
- **VPU-NoMixUp-Mean is the CLEAR winner** (+8% F1, p<0.001)
- VPU loses 6.3% performance when MixUp is removed
- VPU-Mean loses only 1.1% performance
- **VPU is highly MixUp-dependent**

## 📊 Benchmark Scope

- **Seeds:** 5 (42, 123, 456, 789, 2024)
- **Datasets:** 6 (MNIST, FashionMNIST, IMDB, 20News, Spambase, Mushrooms)
- **Methods:** 5 (oracle_bce, vpu, vpu_mean, vpu_nomixup, vpu_nomixup_mean)
- **Total Experiments:** 1,812
- **Statistical Tests:** Paired t-tests, Welch's t-tests, Cohen's d effect sizes
- **Significance Level:** α = 0.05

## 🔬 Analysis Scripts

Located in `scripts/`:
- `analyze_multiseed_statistics.py` - Multi-seed F1 analysis with paired t-tests
- `analyze_multiseed_all_metrics.py` - Comprehensive metrics including calibration
- `analyze_nomixup_variants.py` - No-MixUp variants analysis
- `analyze_vpu_variants.py` - Original VPU variants comparison
- `analyze_statistical_significance.py` - Statistical robustness analysis
- `analyze_prior_variants.py` - Prior-weighted variants analysis (720 experiments)

## 📅 Analysis Date

March 18, 2026

## 🎓 Quick Decision Guide

**I have MixUp and...**
- Need reliable probabilities → **VPU**
- Need maximum recall → **VPU-Mean**
- Have low c (≤0.1) → **VPU-Mean**
- Have low prior (≤0.3) → **VPU**
- General use → **VPU-Mean** (slight preference)

**I don't have MixUp:**
- Any scenario → **VPU-NoMixUp-Mean**

**⚠️ AVOID:**
- **VPU-Mean-Prior** (no F1 benefit, only calibration)
- **VPU-NoMixUp-Mean-Prior** (F1 -5.8%, catastrophic failures on some datasets)
