# Threshold-Independent Metrics for PU Learning

## Problem: F1 Score is Threshold-Dependent

F1 score at a fixed threshold (sigmoid(logit) >= 0.5) can be **misleading** when comparing methods with different calibration properties:

1. **Uncalibrated methods** may have good discrimination but poor F1 at threshold 0.5
2. **Different methods** may have optimal thresholds at different values
3. **Calibration differences** confound performance comparisons

**Example:** A method with excellent discrimination (high AUC) but poor calibration might show low F1 at the default threshold, even though it could achieve high F1 with optimal threshold selection.

---

## Solution: Average Precision (AP) and Max F1

We've added two threshold-independent metrics:

### 1. Average Precision (AP)

**Definition:** Area under the Precision-Recall curve

**Why it's better than F1:**
- Evaluates performance **across all possible thresholds**
- Threshold-independent (like AUC)
- Standard in information retrieval
- More robust to calibration differences

**Formula:**
```
AP = ∫₀¹ Precision(r) dr
```

where the integral is over recall values.

**Interpretation:**
- AP = 1.0: Perfect precision-recall tradeoff
- AP > F1: Model can achieve better F1 with optimal threshold
- AP ≈ F1: Model is well-calibrated (threshold 0.5 is near-optimal)

### 2. Maximum F1

**Definition:** The highest F1 score achievable across all possible thresholds

**Why it's useful:**
- Shows **best-case performance** with optimal threshold
- Reveals calibration quality: `max_f1 - f1` = improvement from threshold tuning
- Complements AP by showing peak performance

**Computation:**
```python
precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
max_f1 = max(f1_scores)
```

---

## Implementation

### Code Changes

**File:** `train/train_utils.py`

**Added imports:**
```python
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
)
```

**Added computation (lines 670-687):**
```python
# Average Precision (AP) - threshold-independent F1-like metric
try:
    if len(np.unique(y_true_arr)) < 2:
        ap = float("nan")
        max_f1 = float("nan")
    else:
        # Average Precision
        ap = float(average_precision_score(y_true_arr, y_score_arr, pos_label=1))

        # Maximum F1 achievable across all thresholds
        precisions, recalls, thresholds = precision_recall_curve(y_true_arr, y_score_arr, pos_label=1)
        f1_scores = np.where(
            (precisions + recalls) > 0,
            2 * (precisions * recalls) / (precisions + recalls),
            0
        )
        max_f1 = float(np.max(f1_scores))
except Exception:
    ap = float("nan")
    max_f1 = float("nan")
```

**Added to metrics dict:**
```python
return {
    "error": 1 - acc,
    "risk": risk,
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1,
    "auc": auc,
    "ap": ap,  # NEW: Average Precision
    "max_f1": max_f1,  # NEW: Maximum F1
    **calib_metrics,
}
```

### Analysis Script

**File:** `scripts/analyze_average_precision.py`

Analyzes:
- Overall AP vs F1 comparison
- AP differences between VPU methods
- Max F1 gaps (shows calibration quality)
- Dataset-specific patterns

**Usage:**
```bash
python scripts/analyze_average_precision.py
```

---

## How to Use These Metrics

### Re-running Experiments

To generate AP and max_f1 data, re-run any experiment:

```bash
# Single experiment
python run_train.py --dataset-config config/datasets_exploration/vary_c_mnist.yaml --methods vpu vpu_mean

# Full benchmark
./scripts/run_vpu_benchmark.sh
```

New results will automatically include `test_ap` and `test_max_f1`.

### Interpreting Results

**Scenario 1: AP ≈ F1**
- Method is well-calibrated
- Fixed threshold (0.5) is near-optimal
- Example: Oracle BCE, VPU (log methods tend to be well-calibrated)

**Scenario 2: AP > F1**
- Method has good discrimination but poor calibration
- Could achieve higher F1 with threshold tuning
- Example: VPU-Mean (mean formulation may be less calibrated)

**Scenario 3: max_f1 >> F1**
- Large calibration gap
- Optimal threshold is far from 0.5
- Consider threshold tuning or calibration methods

---

## Expected Benefits

### Better Method Comparisons

**Before (F1 only):**
- Method A: F1 = 0.85
- Method B: F1 = 0.83
- **Conclusion:** A is better

**After (with AP):**
- Method A: F1 = 0.85, AP = 0.85, max_f1 = 0.86
- Method B: F1 = 0.83, AP = 0.87, max_f1 = 0.88
- **Conclusion:** B has better discrimination, just needs threshold tuning

### Calibration Insights

The gap `max_f1 - f1` reveals calibration quality:
- Small gap (< 2%): Well-calibrated, threshold 0.5 is near-optimal
- Medium gap (2-5%): Moderate calibration needed
- Large gap (> 5%): Poor calibration, significant improvement possible

### Fair Comparisons

AP levels the playing field:
- Methods with different calibration can be compared fairly
- Removes threshold selection bias
- Standard metric in information retrieval and object detection

---

## Relation to Existing Metrics

| Metric | What it Measures | Threshold-Dependent? |
|--------|------------------|---------------------|
| **F1** | Harmonic mean of precision/recall at threshold 0.5 | ✅ Yes |
| **AUC** | Area under ROC curve (TPR vs FPR) | ❌ No |
| **AP** | Area under Precision-Recall curve | ❌ No |
| **max_f1** | Best F1 achievable | ❌ No (finds optimal) |
| **Precision** | TP/(TP+FP) at threshold 0.5 | ✅ Yes |
| **Recall** | TP/(TP+FN) at threshold 0.5 | ✅ Yes |

**Why both AUC and AP?**
- **AUC**: Discrimination ability (separating positive from negative)
- **AP**: Precision-recall tradeoff (quality of positive predictions)
- They measure different aspects and both are valuable

---

## Recommendations

### For New Experiments

**Always report:**
1. F1 (for consistency with existing results)
2. AP (threshold-independent comparison)
3. AUC (discrimination ability)
4. max_f1 (calibration quality check)

### For Analysis

**Primary comparison metric:** Use **AP** instead of F1 when:
- Comparing methods with different calibration properties
- Methods are not calibrated to threshold 0.5
- You want a fair, threshold-independent comparison

**Use F1 when:**
- Comparing to existing results (consistency)
- Threshold 0.5 is the deployed operating point
- Methods are similarly calibrated

### For Method Selection

**If AP >> F1:** Consider:
1. Threshold tuning (find optimal threshold on validation set)
2. Calibration methods (Platt scaling, isotonic regression)
3. Training with calibration objectives

---

## Literature

**Average Precision:**
- Standard metric in COCO object detection
- Used in information retrieval (Manning et al., 2008)
- Recommended by scikit-learn for imbalanced datasets

**Precision-Recall Curves:**
- More informative than ROC for imbalanced data
- Show tradeoff between precision and recall
- AP summarizes this tradeoff in a single number

**References:**
- Scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
- COCO metrics: https://cocodataset.org/#detection-eval

---

## Example Output

When metrics are computed, you'll see:

```json
{
  "test_f1": 0.8621,
  "test_ap": 0.8756,
  "test_max_f1": 0.8894,
  "test_auc": 0.8946
}
```

**Interpretation:**
- F1 at threshold 0.5: 0.8621
- AP (threshold-independent): 0.8756 (+1.3% better)
- Max achievable F1: 0.8894 (+2.7% with optimal threshold)
- Model is reasonably calibrated (small gap), but could improve F1 slightly with threshold tuning

---

## Summary

**Key Points:**
1. ✅ **AP is now computed automatically** for all experiments
2. ✅ **max_f1 shows calibration quality**
3. ✅ **Use AP for fair comparisons** between methods
4. ✅ **Existing results still valid** (F1 is still reported)
5. ⚠️ **Re-run experiments to get AP data** (existing results don't have it)

**Next Steps:**
- Re-run key experiments to generate AP data
- Compare VPU methods on AP (may reveal different rankings)
- Use AP as primary metric for future work
