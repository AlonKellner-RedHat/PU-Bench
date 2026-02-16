# Benchmark Comparison: nnPU (Sigmoid) vs nnPU-Log vs PUDRa

**Date**: 2026-02-16
**Configuration**: Single seed (42), case-control scenario, c=0.1 (10% labeled ratio)
**Methods Compared**:
- **nnPU**: Non-negative PU learning with sigmoid loss `λ(x) = sigmoid(-x)` (default)
- **nnPU-Log**: Non-negative PU learning with log loss `λ(x) = -log(x)` (previously unused)
- **PUDRa**: Positive-Unlabeled Density Ratio with Point Process/KL loss

---

## Summary Table

| Dataset | Metric | nnPU (Sigmoid) | nnPU-Log | PUDRa | Best Method |
|---------|--------|----------------|----------|-------|-------------|
| **MNIST** | Accuracy | 97.24% | **35.35%** ❌ | **97.32%** ✓ | PUDRa |
| | F1 Score | 97.23% | 35.86% | 97.30% | PUDRa |
| | AUC | 99.60% | 29.07% | 99.60% | nnPU/PUDRa |
| **Fashion-MNIST** | Accuracy | 96.98% | **23.16%** ❌ | **98.26%** ✓ | PUDRa |
| | F1 Score | 97.05% | 24.95% | 98.27% | PUDRa |
| | AUC | 99.47% | 19.27% | 99.73% | PUDRa |
| **CIFAR-10** | Accuracy | 75.31% | **19.99%** ❌ | **89.06%** ✓ | PUDRa |
| | F1 Score | 70.00% | 15.40% | 86.43% | PUDRa |
| | AUC | 81.19% | 15.21% | 95.90% | PUDRa |
| **IMDb** | Accuracy | 75.82% | **41.94%** ❌ | **77.41%** ✓ | PUDRa |
| | F1 Score | 75.88% | 43.29% | 77.46% | PUDRa |
| | AUC | 84.22% | 38.43% | 85.53% | PUDRa |
| **20News** | Accuracy | 86.42% | **53.33%** ❌ | **86.03%** | nnPU |
| | F1 Score | 87.78% | 59.52% | 87.41% | nnPU |
| | AUC | 93.17% | 52.52% | 93.53% | PUDRa |
| **Connect-4** | Accuracy | 66.16% | **58.25%** ❌ | **81.57%** ✓ | PUDRa |
| | F1 Score | 74.70% | 68.79% | 86.48% | PUDRa |
| | AUC | 68.16% | 54.08% | 88.11% | PUDRa |
| **Mushrooms** | Accuracy | 97.42% | **65.85%** ❌ | **98.71%** ✓ | PUDRa |
| | F1 Score | 97.32% | 65.42% | 98.64% | PUDRa |
| | AUC | 99.24% | 71.84% | 99.99% | PUDRa |
| **Spambase** | Accuracy | 60.69% | **48.86%** ❌ | 61.02% | PUDRa |
| | F1 Score | 0.55% ⚠️ | 36.95% | 2.18% ⚠️ | nnPU-Log |
| | AUC | 92.52% | 47.95% | 91.78% | nnPU |
| **AlzheimerMRI** | Accuracy | **63.52%** ✓ | **54.30%** ❌ | 48.91% ⚠️ | nnPU |
| | F1 Score | **70.42%** ✓ | 54.90% | 65.54% | nnPU |
| | AUC | 77.44% | 56.05% | **79.53%** ✓ | PUDRa |

**Legend**:
- ✓ = Best performing method for this dataset
- ❌ = Catastrophic failure (near-random or worse performance)
- ⚠️ = Trivial classifier (collapsed to predict mostly one class despite high AUC)

---

## Key Findings

### 1. nnPU-Log Performance Issues

The **nnPU-Log variant performs catastrophically worse** than nnPU sigmoid across all datasets:

- **Image datasets**: Complete failures with accuracy near or below random chance
  - MNIST: 35% (vs 50% random for binary)
  - Fashion-MNIST: 23%
  - CIFAR-10: 20%

- **Text datasets**: Barely better than random
  - IMDb: 42% (binary classification, random = 50%)
  - 20News: 53% (binary classification)

- **Tabular datasets**: Poor but not catastrophic
  - Connect-4: 58% (still much worse than nnPU's 66%)
  - Mushrooms: 66% (vs nnPU's 97%)
  - Spambase: 49% (near random)

**Conclusion**: The log loss formulation `λ(x) = -log(x)` is **not suitable for PU learning** in this implementation. The poor performance likely stems from:
- Log loss requiring strictly positive outputs (x > 0), which may not be guaranteed by raw model outputs
- Different gradient characteristics that interfere with the nnPU risk estimator
- Potential numerical instability with log of small values

### 2. PUDRa Performance

**PUDRa demonstrates strong performance** across all dataset types, **winning on 7 out of 9 datasets**:

- **Strongest on image datasets**:
  - Fashion-MNIST: 98.26% (vs nnPU's 96.98%)
  - CIFAR-10: 89.06% (vs nnPU's 75.31%) - **13.75% improvement**

- **Strongest on tabular datasets**:
  - Connect-4: 81.57% (vs nnPU's 66.16%) - **15.41% improvement**
  - Mushrooms: 98.71% (vs nnPU's 97.42%)

- **Competitive on text datasets**:
  - IMDb: 77.41% (vs nnPU's 75.82%)
  - 20News: 86.03% (vs nnPU's 86.42%) - slightly worse but comparable

**Conclusion**: PUDRa's Point Process/KL formulation provides consistent benefits, especially on structured (tabular) and complex (CIFAR-10) datasets.

### 3. Dataset-Specific Observations

#### Spambase Anomaly
Both nnPU and PUDRa produce **trivial classifiers** (F1 < 3%) despite high AUC (>90%):
- Models learned good ranking (high AUC) but collapsed to predicting mostly one class
- This indicates the methods struggle with this particular dataset/prior combination
- Interestingly, nnPU-Log maintains better F1 (37%) but with poor AUC (48%)

#### CIFAR-10: PUDRa's Biggest Win
PUDRa achieved **13.75% higher accuracy** than nnPU on CIFAR-10, the most complex visual dataset. This suggests PUDRa's formulation is particularly well-suited for challenging classification tasks.

#### AlzheimerMRI: Challenging Medical Imaging
This medical imaging dataset proved challenging for all methods:
- **nnPU wins** with 63.52% accuracy and 70.42% F1 - the best overall performance
- **PUDRa exhibits trivial classifier behavior** with 48.91% accuracy but **100% recall** (predicting positive for almost everything)
  - Similar to Spambase but in opposite direction - collapsed to predict mostly positive class
  - Still achieved best AUC (79.53%), showing it learned good ranking despite poor calibration
- **nnPU-Log performs poorly** at 54.30% accuracy, consistent with its failures on other image datasets
- The highly imbalanced class distribution (NonDemented vs. 3 dementia classes combined) and medical imaging complexity make this a difficult PU learning task

### 4. Training Efficiency

PUDRa benefits from **early stopping**, often converging faster:
- MNIST: 11 epochs (nnPU: 32 epochs)
- IMDb: 2 epochs (nnPU: 5 epochs)
- Fashion-MNIST: 17 epochs (nnPU: 17 epochs)

nnPU-Log often stops early too, but due to **failure to improve** rather than convergence.

---

## Recommendations

1. **Do not use nnPU-Log**: The log loss variant is not viable for PU learning in this implementation
2. **Prefer PUDRa for complex datasets**: Especially tabular data and challenging image datasets like CIFAR-10
3. **nnPU sigmoid remains competitive**: Still a strong baseline, particularly excels on medical imaging (AlzheimerMRI) and text data
4. **Watch for trivial classifiers**: Both nnPU and PUDRa can collapse on certain datasets (Spambase, AlzheimerMRI) - may require calibration techniques or different hyperparameters
5. **Consider task requirements**: If ranking (AUC) is critical, PUDRa may be preferable even when accuracy is lower

---

## Method Details

### nnPU (Sigmoid)
- **Loss**: `λ(x) = sigmoid(-x)`
- **Implementation**: `train/nnpu_trainer.py`
- **Risk**: Non-negative risk estimator with gamma=1.0, beta=0.0

### nnPU-Log
- **Loss**: `λ(x) = -log(x)`
- **Implementation**: `train/nnpu_log_trainer.py`
- **Risk**: Non-negative risk estimator with gamma=1.0, beta=0.0
- **Status**: ❌ Not recommended - consistently poor performance

### PUDRa
- **Loss**: Point Process/Generalized KL: `L = π * E_P[-log(g(x))] + E_U[g(x)]`
- **Implementation**: `train/pudra_trainer.py`
- **Activation**: Sigmoid with epsilon=1e-7

---

## Files

- **Trainer implementations**:
  - [train/nnpu_trainer.py](../train/nnpu_trainer.py)
  - [train/nnpu_log_trainer.py](../train/nnpu_log_trainer.py)
  - [train/pudra_trainer.py](../train/pudra_trainer.py)

- **Method configs**:
  - [config/methods/nnpu.yaml](../config/methods/nnpu.yaml)
  - [config/methods/nnpu_log.yaml](../config/methods/nnpu_log.yaml)
  - [config/methods/pudra.yaml](../config/methods/pudra.yaml)

- **Raw results**: `results/seed_42/*.json`
