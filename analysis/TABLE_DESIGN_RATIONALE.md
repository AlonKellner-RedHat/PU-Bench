# Academic Paper Table Design Rationale

## Table Structure Decision

### Rows: Methods (Grouped by Type)
**Why:** Readers compare methods, not datasets. This layout facilitates:
- Direct method-to-method comparison across metrics
- Clear visual separation between no-mixup, mixup, and baseline approaches
- Easy identification of best/second-best performers

**Groups:**
1. **No Mixup Methods**: Core VPU variants without data augmentation
2. **With Mixup Methods**: Same methods with mixup augmentation
3. **Baselines**: Established PU learning methods (nnPU, Dist-PU)

### Columns: Key Metrics + Summary Statistics

**Metric Selection (6 total):**
- **AUC, AP**: Threshold-invariant ranking metrics (most important for PU learning)
- **Max F1**: Best achievable classification performance
- **ECE, Brier**: Calibration quality (crucial for probabilistic predictions)
- **Oracle CE**: Optimal cross-entropy with best threshold

**Why these 6?**
- Cover all three critical aspects: ranking, calibration, classification
- Standard in ML literature (AUC, Brier) + PU-specific (Max F1)
- Compact enough for single table, comprehensive enough for full picture
- Excluded less common metrics (ANICE, SNICE, MCE) to avoid clutter

**Summary Columns:**
- **Avg Rank**: Average ranking across all 6 metrics → overall method quality
- **Wins**: Number of metrics where method is best → dominance indicator

### Value Format: mean ± std

**Why not confidence intervals?**
- Standard in ML papers
- Std shows variability directly
- More compact than CI
- Easy to interpret variance across experimental conditions

**Why subscript in LaTeX?**
- `0.849$_{0.169}$` more compact than `0.849 ± 0.169`
- Cleaner visual presentation
- Standard in top-tier venues (NeurIPS, ICML, ICLR)

### Visual Highlighting

**Bold** = Best performance per metric
- Immediately draws eye to winner
- Clear across all metrics

**Italic** = Second-best performance
- Shows competitive alternatives
- Important when best differs only slightly

**Why not color?**
- Black & white printable
- Accessible (no color-blindness issues)
- Journal standard

## Statistical Analysis Section

### Three Key Comparisons

1. **Auto vs 0.5 Prior (No Mixup)**
   - Tests core hypothesis: Is automatic prior estimation better?
   - Result: 0.5 significantly better for calibration, no difference for ranking

2. **Mixup Impact**
   - Tests augmentation benefit
   - Result: Slight improvement in ranking (ns), worse calibration (***)

3. **Best Method vs Baselines**
   - Validates improvements over prior art
   - Result: Significant improvements across all metrics (***)

### P-value Thresholds
- *** p < 0.001 (very strong evidence)
- ** p < 0.01 (strong evidence)
- * p < 0.05 (moderate evidence)
- ns = not significant

Standard in psychology, medicine, ML

## Design Principles Applied

### 1. **Clarity**
- Grouped methods logically
- Clear direction indicators (↑/↓)
- Consistent number formatting (3 decimal places)

### 2. **Completeness**
- All major method variants included
- Coverage of ranking, calibration, classification
- Statistical significance testing

### 3. **Conciseness**
- Single table captures full story
- 8 methods × 6 metrics = manageable
- Summary columns reduce need for external analysis

### 4. **Credibility**
- Large sample size (1,575 experiments)
- Multiple datasets (7)
- Proper statistical testing
- Transparent about variance (std shown)

### 5. **Academic Standards**
- LaTeX format provided
- Standard metrics (AUC, Brier)
- Standard notation (± for uncertainty)
- Standard significance markers (***, **, *)

## Usage Recommendations

### For Paper Main Text
Use the **LaTeX version** in your results section:
- Fits in 2-column format with `\resizebox{\textwidth}`
- Professional typesetting
- Easy to reference in text: "Table 1 shows..."

### For Supplementary Material
Include **statistical analysis tables**:
- Detailed pairwise comparisons
- P-values for reproducibility
- Can reference in main text: "See Appendix A for statistical tests"

### For README/Documentation
Use **Markdown version**:
- Renders directly on GitHub
- Easy to update
- Accessible to non-LaTeX users

## Narrative the Table Supports

Looking at Table 1, the story is clear:

1. **VPU-Mean-Prior (0.5) + mixup wins overall** (avg rank 1.5, 3 wins)
   - Best ranking performance (AUC=0.849, AP=0.862)
   - Best classification (Max F1=0.852)

2. **Auto prior doesn't help for ranking** (p > 0.9, ns)
   - Nearly identical AUC/AP/F1 between auto and 0.5

3. **But 0.5 is better calibrated** (ECE: p < 0.001)
   - 0.5 prior has 48% lower ECE than auto

4. **Mixup slightly improves ranking** (AUC +0.003, ns)
   - But significantly hurts calibration (ECE +30%, p < 0.001)

5. **All VPU variants beat baselines** (p < 0.001)
   - 12-13% AUC improvement over nnPU
   - 62% ECE reduction vs nnPU

This is a **compelling, data-driven narrative** supported by a single, well-designed table.

## Alternative Designs Considered

### Option 1: Datasets as Rows
- **Rejected**: Too many rows (7 datasets), hard to compare methods
- Better suited for dataset-specific analysis in appendix

### Option 2: More Metrics (10-12 columns)
- **Rejected**: Cluttered, hard to read, many redundant
- ECE/MCE highly correlated, AP/AUC capture similar information

### Option 3: Separate Tables for Calibration vs Ranking
- **Rejected**: Fragments the story, reader can't see trade-offs
- Single table shows calibration-ranking trade-off clearly

### Option 4: Heatmap Instead of Table
- **Rejected**: Less precise, harder to extract exact values
- Better for supplementary visualizations

## Conclusion

This table design maximizes:
- **Clarity**: Easy to read and interpret
- **Information density**: Full story in compact format  
- **Academic rigor**: Proper statistics and formatting
- **Narrative power**: Clear winner, clear trade-offs

Perfect for a top-tier conference or journal submission.
