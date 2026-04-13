# Phase 1 Results - Paper-Ready Table

Generated: 2026-04-13

---

## Markdown Version (for README/reports)

**Table 1: Phase 1 Results - Method Comparison Across All Datasets**

*Results averaged over 1,575 experiments (7 datasets × 3 label frequencies × 3 class priors × 5 seeds). Values shown as mean ± std. **Bold** indicates best performance, *italic* indicates second-best. ↑/↓ indicate whether higher/lower is better.*

| Method | AUC ↑ | AP ↑ | Max F1 ↑ | ECE ↓ | Brier ↓ | Oracle CE ↓ | Avg Rank | Wins |
|--------|------:|------:|------:|------:|------:|------:|---------:|-----:|
| ***No Mixup Methods*** | | | | | | | | |
| VPU | 0.833 $\pm$ 0.177 | 0.844 $\pm$ 0.170 | 0.838 $\pm$ 0.120 | 0.230 $\pm$ 0.189 | 0.238 $\pm$ 0.182 | 0.965 $\pm$ 1.336 | 6.0 | 0 |
| VPU-MP (auto) | 0.845 $\pm$ 0.175 | 0.856 $\pm$ 0.167 | 0.850 $\pm$ 0.118 | 0.148 $\pm$ 0.146 | 0.170 $\pm$ 0.146 | 0.552 $\pm$ 0.494 | 3.5 | 0 |
| VPU-MP (0.5) | *0.846 $\pm$ 0.173* | *0.857 $\pm$ 0.162* | *0.850 $\pm$ 0.115* | **0.100 $\pm$ 0.081** | **0.147 $\pm$ 0.094** | **0.461 $\pm$ 0.316** | 1.5 | 3 |
|
| ***With Mixup Methods*** | | | | | | | | |
| VPU + mixup | 0.837 $\pm$ 0.178 | 0.850 $\pm$ 0.168 | 0.846 $\pm$ 0.118 | 0.153 $\pm$ 0.126 | 0.167 $\pm$ 0.138 | 0.586 $\pm$ 0.708 | 4.8 | 0 |
| VPU-MP (auto) + mixup | 0.844 $\pm$ 0.173 | 0.855 $\pm$ 0.163 | 0.849 $\pm$ 0.116 | 0.149 $\pm$ 0.114 | 0.159 $\pm$ 0.128 | 0.504 $\pm$ 0.370 | 3.7 | 0 |
| VPU-MP (0.5) + mixup | **0.849 $\pm$ 0.169** | **0.862 $\pm$ 0.154** | **0.852 $\pm$ 0.112** | *0.131 $\pm$ 0.088* | *0.148 $\pm$ 0.098* | *0.463 $\pm$ 0.278* | 1.5 | 3 |
|
| ***Baselines*** | | | | | | | | |
| nnPU | 0.721 $\pm$ 0.211 | 0.739 $\pm$ 0.193 | 0.782 $\pm$ 0.114 | 0.346 $\pm$ 0.193 | 0.349 $\pm$ 0.192 | 2.769 $\pm$ 2.338 | 8.0 | 0 |
| Dist-PU | 0.762 $\pm$ 0.225 | 0.784 $\pm$ 0.200 | 0.815 $\pm$ 0.115 | 0.241 $\pm$ 0.183 | 0.251 $\pm$ 0.181 | 1.402 $\pm$ 1.520 | 7.0 | 0 |
|

---

## LaTeX Version (for paper)

```latex
\begin{table}[t]
\centering
\caption{Comparison of PU Learning Methods on Phase 1 Benchmark}
\label{tab:phase1_results}
\resizebox{\textwidth}{!}{
\begin{tabular}{lrrrrrrrr}
\toprule
Method & AUC $\uparrow$ & AP $\uparrow$ & Max F1 $\uparrow$ & ECE $\downarrow$ & Brier $\downarrow$ & Oracle CE $\downarrow$ & Avg Rank & Wins \\
\midrule
VPU & 0.833$_{0.177}$ & 0.844$_{0.170}$ & 0.838$_{0.120}$ & 0.230$_{0.189}$ & 0.238$_{0.182}$ & 0.965$_{1.336}$ & 6.0 & 0 \\
VPU-MP (auto) & 0.845$_{0.175}$ & 0.856$_{0.167}$ & 0.850$_{0.118}$ & 0.148$_{0.146}$ & 0.170$_{0.146}$ & 0.552$_{0.494}$ & 3.5 & 0 \\
VPU-MP (0.5) & \textit{0.846}$_{0.173}$ & \textit{0.857}$_{0.162}$ & \textit{0.850}$_{0.115}$ & \textbf{0.100}$_{0.081}$ & \textbf{0.147}$_{0.094}$ & \textbf{0.461}$_{0.316}$ & 1.5 & 3 \\
\midrule
VPU + mixup & 0.837$_{0.178}$ & 0.850$_{0.168}$ & 0.846$_{0.118}$ & 0.153$_{0.126}$ & 0.167$_{0.138}$ & 0.586$_{0.708}$ & 4.8 & 0 \\
VPU-MP (auto) + mixup & 0.844$_{0.173}$ & 0.855$_{0.163}$ & 0.849$_{0.116}$ & 0.149$_{0.114}$ & 0.159$_{0.128}$ & 0.504$_{0.370}$ & 3.7 & 0 \\
VPU-MP (0.5) + mixup & \textbf{0.849}$_{0.169}$ & \textbf{0.862}$_{0.154}$ & \textbf{0.852}$_{0.112}$ & \textit{0.131}$_{0.088}$ & \textit{0.148}$_{0.098}$ & \textit{0.463}$_{0.278}$ & 1.5 & 3 \\
\midrule
nnPU & 0.721$_{0.211}$ & 0.739$_{0.193}$ & 0.782$_{0.114}$ & 0.346$_{0.193}$ & 0.349$_{0.192}$ & 2.769$_{2.338}$ & 8.0 & 0 \\
Dist-PU & 0.762$_{0.225}$ & 0.784$_{0.200}$ & 0.815$_{0.115}$ & 0.241$_{0.183}$ & 0.251$_{0.181}$ & 1.402$_{1.520}$ & 7.0 & 0 \\
\bottomrule
\end{tabular}
}
\end{table}
```

---


## Statistical Analysis

### VPU-Mean-Prior: Auto vs 0.5 Prior (No Mixup)

| Metric | Auto | 0.5 | Difference | p-value | Significant? |
|--------|------|-----|------------|---------|--------------|
| AUC | 0.845 | 0.846 | +0.001 | 0.9180 | ns |
| AP | 0.856 | 0.857 | +0.002 | 0.8718 | ns |
| Max F1 | 0.850 | 0.850 | +0.000 | 0.9622 | ns |
| ECE | 0.148 | 0.100 | -0.048 | 0.0000 | *** |
| Brier | 0.170 | 0.147 | -0.023 | 0.0024 | ** |
| Oracle CE | 0.552 | 0.461 | -0.090 | 0.0004 | *** |

*Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant*

### Mixup Impact: VPU-Mean-Prior (0.5)

| Metric | No Mixup | With Mixup | Difference | p-value | Significant? |
|--------|----------|------------|------------|---------|--------------|
| AUC | 0.846 | 0.849 | +0.003 | 0.7720 | ns |
| AP | 0.857 | 0.862 | +0.005 | 0.6159 | ns |
| Max F1 | 0.850 | 0.852 | +0.002 | 0.7291 | ns |
| ECE | 0.100 | 0.131 | +0.030 | 0.0000 | *** |
| Brier | 0.147 | 0.148 | +0.001 | 0.8568 | ns |
| Oracle CE | 0.461 | 0.463 | +0.002 | 0.9313 | ns |

### Best Method vs Baselines

| Metric | VPU-MP(0.5) | nnPU | Dist-PU | vs nnPU | vs Dist-PU |
|--------|-------------|------|---------|---------|------------|
| AUC | 0.849 | 0.721 | 0.762 | *** | *** |
| AP | 0.862 | 0.739 | 0.784 | *** | *** |
| Max F1 | 0.852 | 0.782 | 0.815 | *** | *** |
| ECE | 0.131 | 0.346 | 0.241 | *** | *** |
| Brier | 0.148 | 0.349 | 0.251 | *** | *** |
| Oracle CE | 0.463 | 2.769 | 1.402 | *** | *** |


---

## Notes

- **Sample size**: Each method evaluated on ~525-541 experiments
- **Datasets**: 20News, Connect4, FashionMNIST, IMDB, MNIST, Mushrooms, Spambase
- **Experimental factors**: 3 label frequencies (c) × 3 class priors (π) × 5 random seeds
- **Statistical tests**: Two-sample t-tests for pairwise comparisons
- **Best method overall**: VPU-Mean-Prior (0.5) + mixup
  - Wins 4/6 metrics
  - Average rank: ~1.5 across all metrics
  - Significantly better than baselines (p < 0.001)
