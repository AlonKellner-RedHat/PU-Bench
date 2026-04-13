# Phase 1 Experimental Results Summary

**Generated:** 2026-04-13

**Total Experiments:** 1575
**Datasets:** 20News, Connect4, FashionMNIST, IMDB, MNIST, Mushrooms, Spambase
**Methods Evaluated:** 10

## Methods Analyzed

- **distpu**: 534 experiments
- **nnpu**: 535 experiments
- **vpu**: 538 experiments
- **vpu_mean_prior_0.5**: 525 experiments
- **vpu_mean_prior_1.0**: 525 experiments
- **vpu_mean_prior_auto**: 525 experiments
- **vpu_nomixup**: 541 experiments
- **vpu_nomixup_mean_prior_0.5**: 525 experiments
- **vpu_nomixup_mean_prior_1.0**: 525 experiments
- **vpu_nomixup_mean_prior_auto**: 525 experiments

## Overall Method Rankings

### Calibration Metrics

#### ECE ↓

| Rank | Method | Mean | Std | Median |
|------|--------|------|-----|--------|
| 1 | vpu_nomixup_mean_prior_0.5 | 0.1002 | 0.0806 | 0.0761 |
| 2 | vpu_mean_prior_0.5 | 0.1307 | 0.0877 | 0.1038 |
| 3 | vpu_nomixup_mean_prior_auto | 0.1483 | 0.1460 | 0.0903 |
| 4 | vpu_mean_prior_auto | 0.1488 | 0.1137 | 0.1106 |
| 5 | vpu | 0.1528 | 0.1261 | 0.1019 |
| 6 | vpu_nomixup | 0.2305 | 0.1885 | 0.1796 |
| 7 | distpu | 0.2412 | 0.1825 | 0.1950 |
| 8 | nnpu | 0.3462 | 0.1929 | 0.3935 |
| 9 | vpu_nomixup_mean_prior_1.0 | 0.4868 | 0.0754 | 0.5000 |
| 10 | vpu_mean_prior_1.0 | 0.4869 | 0.0753 | 0.5000 |

#### BRIER ↓

| Rank | Method | Mean | Std | Median |
|------|--------|------|-----|--------|
| 1 | vpu_nomixup_mean_prior_0.5 | 0.1469 | 0.0943 | 0.1493 |
| 2 | vpu_mean_prior_0.5 | 0.1480 | 0.0975 | 0.1477 |
| 3 | vpu_mean_prior_auto | 0.1593 | 0.1282 | 0.1362 |
| 4 | vpu | 0.1673 | 0.1374 | 0.1482 |
| 5 | vpu_nomixup_mean_prior_auto | 0.1699 | 0.1455 | 0.1457 |
| 6 | vpu_nomixup | 0.2379 | 0.1817 | 0.2282 |
| 7 | distpu | 0.2512 | 0.1806 | 0.2314 |
| 8 | nnpu | 0.3495 | 0.1914 | 0.3939 |
| 9 | vpu_nomixup_mean_prior_1.0 | 0.4868 | 0.0754 | 0.5000 |
| 10 | vpu_mean_prior_1.0 | 0.4869 | 0.0753 | 0.5000 |

#### ANICE ↓

| Rank | Method | Mean | Std | Median |
|------|--------|------|-----|--------|
| 1 | vpu_mean_prior_auto | 0.6530 | 0.4782 | 0.5414 |
| 2 | vpu_mean_prior_0.5 | 0.6631 | 0.4586 | 0.5735 |
| 3 | vpu | 0.7089 | 0.5373 | 0.5431 |
| 4 | vpu_nomixup_mean_prior_0.5 | 0.7332 | 0.4028 | 0.6710 |
| 5 | vpu_nomixup_mean_prior_auto | 0.8498 | 0.6663 | 0.6451 |
| 6 | distpu | 1.0813 | 0.7025 | 0.9158 |
| 7 | vpu_nomixup | 1.2846 | 0.9743 | 0.9045 |
| 8 | nnpu | 1.7592 | 1.1616 | 1.4626 |
| 9 | vpu_nomixup_mean_prior_1.0 | 1.8244 | 0.3197 | 1.9670 |
| 10 | vpu_mean_prior_1.0 | 1.8245 | 0.3196 | 1.9670 |

### Ranking Metrics

#### AUC ↑

| Rank | Method | Mean | Std | Median |
|------|--------|------|-----|--------|
| 1 | vpu_mean_prior_0.5 | 0.8489 | 0.1693 | 0.9229 |
| 2 | vpu_nomixup_mean_prior_0.5 | 0.8458 | 0.1729 | 0.9202 |
| 3 | vpu_nomixup_mean_prior_auto | 0.8447 | 0.1750 | 0.9210 |
| 4 | vpu_mean_prior_auto | 0.8436 | 0.1733 | 0.9124 |
| 5 | vpu | 0.8370 | 0.1775 | 0.9109 |
| 6 | vpu_nomixup | 0.8326 | 0.1770 | 0.8846 |
| 7 | distpu | 0.7622 | 0.2244 | 0.8427 |
| 8 | nnpu | 0.7211 | 0.2109 | 0.7150 |
| 9 | vpu_mean_prior_1.0 | 0.5312 | 0.0956 | 0.5073 |
| 10 | vpu_nomixup_mean_prior_1.0 | 0.5312 | 0.0956 | 0.5073 |

#### AP ↑

| Rank | Method | Mean | Std | Median |
|------|--------|------|-----|--------|
| 1 | vpu_mean_prior_0.5 | 0.8621 | 0.1535 | 0.9184 |
| 2 | vpu_nomixup_mean_prior_0.5 | 0.8572 | 0.1617 | 0.9230 |
| 3 | vpu_nomixup_mean_prior_auto | 0.8556 | 0.1671 | 0.9281 |
| 4 | vpu_mean_prior_auto | 0.8554 | 0.1628 | 0.9183 |
| 5 | vpu | 0.8497 | 0.1674 | 0.9145 |
| 6 | vpu_nomixup | 0.8440 | 0.1700 | 0.9079 |
| 7 | distpu | 0.7838 | 0.1997 | 0.8503 |
| 8 | nnpu | 0.7389 | 0.1924 | 0.7388 |
| 9 | vpu_nomixup_mean_prior_1.0 | 0.5500 | 0.1093 | 0.5617 |
| 10 | vpu_mean_prior_1.0 | 0.5500 | 0.1093 | 0.5617 |

#### MAX_F1 ↑

| Rank | Method | Mean | Std | Median |
|------|--------|------|-----|--------|
| 1 | vpu_mean_prior_0.5 | 0.8524 | 0.1118 | 0.8615 |
| 2 | vpu_nomixup_mean_prior_0.5 | 0.8500 | 0.1151 | 0.8617 |
| 3 | vpu_nomixup_mean_prior_auto | 0.8497 | 0.1174 | 0.8607 |
| 4 | vpu_mean_prior_auto | 0.8490 | 0.1157 | 0.8567 |
| 5 | vpu | 0.8459 | 0.1181 | 0.8553 |
| 6 | vpu_nomixup | 0.8380 | 0.1194 | 0.8397 |
| 7 | distpu | 0.8150 | 0.1148 | 0.7984 |
| 8 | nnpu | 0.7818 | 0.1141 | 0.7843 |
| 9 | vpu_nomixup_mean_prior_1.0 | 0.6813 | 0.0654 | 0.6667 |
| 10 | vpu_mean_prior_1.0 | 0.6813 | 0.0653 | 0.6667 |

### Classification Metrics

#### ACCURACY ↑

| Rank | Method | Mean | Std | Median |
|------|--------|------|-----|--------|
| 1 | vpu_mean_prior_0.5 | 0.8106 | 0.1560 | 0.8480 |
| 2 | vpu_nomixup_mean_prior_0.5 | 0.7954 | 0.1690 | 0.8400 |
| 3 | vpu_mean_prior_auto | 0.7921 | 0.1762 | 0.8237 |
| 4 | vpu | 0.7876 | 0.1757 | 0.8105 |
| 5 | vpu_nomixup_mean_prior_auto | 0.7838 | 0.1811 | 0.8178 |
| 6 | vpu_nomixup | 0.7402 | 0.1948 | 0.7587 |
| 7 | distpu | 0.7174 | 0.1935 | 0.7167 |
| 8 | nnpu | 0.6613 | 0.1829 | 0.6203 |
| 9 | vpu_nomixup_mean_prior_1.0 | 0.5132 | 0.0754 | 0.5000 |
| 10 | vpu_mean_prior_1.0 | 0.5131 | 0.0753 | 0.5000 |

#### F1 ↑

| Rank | Method | Mean | Std | Median |
|------|--------|------|-----|--------|
| 1 | vpu_mean_prior_0.5 | 0.8195 | 0.1740 | 0.8511 |
| 2 | vpu_mean_prior_auto | 0.7990 | 0.1938 | 0.8346 |
| 3 | vpu | 0.7961 | 0.2030 | 0.8335 |
| 4 | vpu_nomixup_mean_prior_auto | 0.7754 | 0.2177 | 0.8267 |
| 5 | vpu_nomixup_mean_prior_0.5 | 0.7690 | 0.2473 | 0.8363 |
| 6 | distpu | 0.7422 | 0.1965 | 0.7645 |
| 7 | vpu_nomixup | 0.7024 | 0.2699 | 0.7657 |
| 8 | vpu_nomixup_mean_prior_1.0 | 0.6750 | 0.0648 | 0.6667 |
| 9 | vpu_mean_prior_1.0 | 0.6750 | 0.0648 | 0.6667 |
| 10 | nnpu | 0.6472 | 0.2286 | 0.6540 |

#### PRECISION ↑

| Rank | Method | Mean | Std | Median |
|------|--------|------|-----|--------|
| 1 | vpu_nomixup_mean_prior_0.5 | 0.8344 | 0.1643 | 0.8973 |
| 2 | vpu_nomixup_mean_prior_auto | 0.8227 | 0.1792 | 0.8956 |
| 3 | vpu_mean_prior_0.5 | 0.8226 | 0.1636 | 0.8711 |
| 4 | vpu_nomixup | 0.8179 | 0.1928 | 0.9080 |
| 5 | vpu_mean_prior_auto | 0.8124 | 0.1784 | 0.8770 |
| 6 | vpu | 0.7998 | 0.1861 | 0.8482 |
| 7 | distpu | 0.7380 | 0.1892 | 0.7627 |
| 8 | nnpu | 0.7112 | 0.1786 | 0.7168 |
| 9 | vpu_nomixup_mean_prior_1.0 | 0.5132 | 0.0754 | 0.5000 |
| 10 | vpu_mean_prior_1.0 | 0.5131 | 0.0753 | 0.5000 |

#### RECALL ↑

| Rank | Method | Mean | Std | Median |
|------|--------|------|-----|--------|
| 1 | vpu_nomixup_mean_prior_1.0 | 1.0000 | 0.0000 | 1.0000 |
| 2 | vpu_mean_prior_1.0 | 1.0000 | 0.0000 | 1.0000 |
| 3 | vpu_mean_prior_0.5 | 0.8510 | 0.1774 | 0.8970 |
| 4 | vpu | 0.8454 | 0.2196 | 0.9229 |
| 5 | vpu_mean_prior_auto | 0.8315 | 0.2131 | 0.9029 |
| 6 | distpu | 0.8139 | 0.2288 | 0.8957 |
| 7 | vpu_nomixup_mean_prior_auto | 0.7914 | 0.2522 | 0.9017 |
| 8 | vpu_nomixup_mean_prior_0.5 | 0.7670 | 0.2545 | 0.8481 |
| 9 | vpu_nomixup | 0.7155 | 0.3171 | 0.8748 |
| 10 | nnpu | 0.6920 | 0.3021 | 0.8311 |

## Key Method Comparisons

### No-Mixup: VPU vs VPU-Mean-Prior (auto) vs VPU-Mean-Prior (0.5)

| Metric | VPU | VPU-MP(auto) | VPU-MP(0.5) | Best |
|--------|-----|--------------|-------------|------|
| auc | 0.8326 | 0.8447 | **0.8458** | 0.5 |
| ap | 0.8440 | 0.8556 | **0.8572** | 0.5 |
| ece | 0.2305 | 0.1483 | **0.1002** | 0.5 |
| brier | 0.2379 | 0.1699 | **0.1469** | 0.5 |

### Mixup Impact: VPU

| Metric | No Mixup | With Mixup | Difference | Winner |
|--------|----------|------------|------------|--------|
| auc | 0.8326 | 0.8370 | +0.0044 | Mixup |
| ap | 0.8440 | 0.8497 | +0.0057 | Mixup |
| ece | 0.2305 | 0.1528 | -0.0777 | Mixup |
| brier | 0.2379 | 0.1673 | -0.0705 | Mixup |

### Mixup Impact: VPU-Mean-Prior (auto)

| Metric | No Mixup | With Mixup | Difference | Winner |
|--------|----------|------------|------------|--------|
| auc | 0.8447 | 0.8436 | -0.0012 | No Mixup |
| ap | 0.8556 | 0.8554 | -0.0002 | No Mixup |
| ece | 0.1483 | 0.1488 | +0.0005 | No Mixup |
| brier | 0.1699 | 0.1593 | -0.0106 | Mixup |

### Comparison with Baselines (nnPU, Dist-PU)

Best VPU variant vs baselines (averaged across all experiments):

| Metric | Best VPU | nnPU | Dist-PU |
|--------|----------|------|---------|
| auc | **0.8489** (vpu_mean_prior_0.5) | 0.7211 | 0.7622 |
| ap | **0.8621** (vpu_mean_prior_0.5) | 0.7389 | 0.7838 |
| ece | **0.1002** (vpu_nomixup_mean_prior_0.5) | 0.3462 | 0.2412 |
| brier | **0.1469** (vpu_nomixup_mean_prior_0.5) | 0.3495 | 0.2512 |

## Key Findings

### Auto vs 0.5 Prior (VPU-Mean-Prior, No Mixup)
- Auto wins: 0/4 key metrics
- 0.5 wins: 4/4 key metrics

### Mixup Impact
- Mixup improves performance: 2 cases
- Mixup hurts performance: 0 cases

## Recommendations

Based on Phase 1 results:

1. **Best Overall Ranking (AUC):** vpu_mean_prior_0.5 (AUC=0.8489)
2. **Best Calibration (ECE):** vpu_nomixup_mean_prior_0.5 (ECE=0.1002)

For detailed visualizations, see plots in `analysis/plots/phase1_comprehensive/`
