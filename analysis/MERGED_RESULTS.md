# Positive-Unlabeled Learning: Comprehensive Experimental Results

**A streamlined comparison of 17 PU learning methods across two experimental phases.**

---

## Experimental Design

### Phase 1: Fixed Prior, Variable Label Frequency

- **Datasets**: 7 (20News, Connect4, FashionMNIST, IMDB, MNIST, Mushrooms, Spambase)
- **Random seeds**: 10 [42, 456, 789, 1024, 2048, 3000, 4096, 5555, 6789, 8192]
- **Label frequency (c)**: 3 values [0.1, 0.3, 0.5]
- **True prior (π)**: Dataset natural prior (fixed per dataset)
- **Configurations**: 7 datasets × 10 seeds × 3 c = 210 per method

### Phase 3: Full Hyperparameter Grid

- **Datasets**: 7 (same as Phase 1)
- **Random seeds**: 5 [42, 456, 789, 1024, 2048]
- **Label frequency (c)**: 7 values [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
- **True prior (π)**: 7 values [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
- **Configurations**: 7 datasets × 5 seeds × 7 c × 7 π = 1,715 per method

### Methods Evaluated

**Baseline PU Methods (9):**
- nnPU, nnPU-SB, BBE-PU, LBE, Dist-PU
- Self-PU, P3Mix-E, P3Mix-C, Robust-PU

**VPU Variants (6):**
- VPU, VPU-nomix (base methods)
- VPU-MP(auto), VPU-MP(0.69) (with mixup, mean-prior regularization)
- VPU-nomix-MP(auto), VPU-nomix-MP(0.69) (without mixup, mean-prior regularization)

**Oracle Baselines (2):**
- PN-Naive (treats unlabeled as negative)
- Oracle-PN (trained with true labels)

---

## Phase 1 Results: Performance with Fixed Priors

### Overall Performance

*Mean ± Std across 210 runs per method (7 datasets × 10 seeds × 3 label frequencies). **Bold** = best, *italic* = second-best per metric.*

| Method | AUC ↑ | AP ↑ | Max F1 ↑ | Accuracy ↑ | F1 ↑ | ECE ↓ | Brier ↓ |
|--------|--------|--------|--------|--------|--------|--------|--------|
| nnPU | 0.903 ± 0.121 | 0.916 ± 0.082 | 0.893 ± 0.071 | 0.857 ± 0.111 | 0.868 ± 0.106 | 0.132 ± 0.102 | 0.129 ± 0.103 |
| nnPU-SB | 0.949 ± 0.048 | 0.951 ± 0.046 | 0.911 ± 0.063 | 0.872 ± 0.064 | 0.882 ± 0.069 | 0.108 ± 0.051 | 0.098 ± 0.044 |
| BBE-PU | 0.908 ± 0.127 | 0.926 ± 0.083 | 0.902 ± 0.080 | 0.872 ± 0.125 | 0.881 ± 0.113 | 0.114 ± 0.113 | 0.116 ± 0.113 |
| LBE | 0.945 ± 0.058 | 0.943 ± 0.064 | 0.912 ± 0.079 | 0.891 ± 0.100 | 0.888 ± 0.110 | 0.107 ± 0.068 | 0.095 ± 0.081 |
| Dist-PU | 0.905 ± 0.130 | 0.920 ± 0.085 | 0.897 ± 0.074 | 0.852 ± 0.117 | 0.871 ± 0.087 | 0.147 ± 0.097 | 0.132 ± 0.104 |
| Self-PU | 0.903 ± 0.091 | 0.903 ± 0.081 | 0.883 ± 0.067 | 0.833 ± 0.097 | 0.838 ± 0.120 | 0.145 ± 0.070 | 0.139 ± 0.071 |
| P3Mix-E | 0.821 ± 0.206 | 0.819 ± 0.221 | 0.840 ± 0.146 | 0.755 ± 0.211 | 0.778 ± 0.198 | 0.217 ± 0.119 | 0.190 ± 0.132 |
| P3Mix-C | 0.853 ± 0.191 | 0.849 ± 0.207 | 0.859 ± 0.141 | 0.798 ± 0.202 | 0.829 ± 0.161 | 0.172 ± 0.082 | 0.147 ± 0.104 |
| Robust-PU | 0.904 ± 0.123 | 0.919 ± 0.083 | 0.895 ± 0.074 | 0.860 ± 0.117 | 0.867 ± 0.109 | 0.131 ± 0.109 | 0.130 ± 0.110 |
| VPU-nomix | 0.952 ± 0.049 | 0.954 ± 0.048 | 0.916 ± 0.068 | 0.878 ± 0.115 | 0.855 ± 0.175 | 0.099 ± 0.100 | 0.096 ± 0.090 |
| VPU-nomix-MP(auto) | 0.953 ± 0.048 | 0.955 ± 0.047 | 0.917 ± 0.067 | 0.888 ± 0.106 | 0.884 ± 0.130 | 0.089 ± 0.081 | 0.088 ± 0.068 |
| VPU-nomix-MP(0.69) | *0.954 ± 0.047* | *0.956 ± 0.046* | *0.918 ± 0.066* | *0.905 ± 0.077* | *0.912 ± 0.070* | *0.052 ± 0.050* | *0.074 ± 0.060* |
| VPU | 0.953 ± 0.048 | 0.955 ± 0.048 | 0.916 ± 0.068 | 0.899 ± 0.084 | 0.901 ± 0.097 | 0.083 ± 0.053 | 0.080 ± 0.061 |
| VPU-MP(auto) | 0.952 ± 0.048 | 0.955 ± 0.047 | 0.916 ± 0.067 | 0.886 ± 0.104 | 0.888 ± 0.108 | 0.120 ± 0.072 | 0.093 ± 0.064 |
| VPU-MP(0.69) | 0.953 ± 0.049 | 0.955 ± 0.048 | 0.916 ± 0.068 | 0.895 ± 0.096 | 0.904 ± 0.082 | 0.091 ± 0.063 | 0.083 ± 0.064 |
| PN-Naive | 0.947 ± 0.052 | 0.949 ± 0.051 | 0.911 ± 0.071 | 0.864 ± 0.094 | 0.860 ± 0.158 | 0.399 ± 0.082 | 0.341 ± 0.095 |
| Oracle-PN | **0.971 ± 0.037** | **0.971 ± 0.038** | **0.936 ± 0.060** | **0.931 ± 0.063** | **0.933 ± 0.063** | **0.033 ± 0.041** | **0.054 ± 0.048** |

## Phase 3 Results: Performance Across Full Hyperparameter Grid

### Overall Performance

*Mean ± Std across 1,715 runs per method (7 datasets × 5 seeds × 7 c × 7 π). **Bold** = best, *italic* = second-best per metric.*

| Method | AUC ↑ | AP ↑ | Max F1 ↑ | Accuracy ↑ | F1 ↑ | ECE ↓ | Brier ↓ |
|--------|--------|--------|--------|--------|--------|--------|--------|
| VPU-nomix | 0.870 ± 0.158 | 0.879 ± 0.153 | 0.862 ± 0.114 | 0.724 ± 0.202 | 0.665 ± 0.321 | 0.266 ± 0.202 | 0.255 ± 0.200 |
| VPU-nomix-MP(auto) | 0.887 ± 0.142 | 0.894 ± 0.137 | 0.871 ± 0.108 | 0.795 ± 0.167 | 0.774 ± 0.227 | 0.163 ± 0.133 | 0.165 ± 0.129 |
| VPU-nomix-MP(0.69) | 0.890 ± 0.138 | 0.896 ± 0.133 | 0.873 ± 0.107 | *0.820 ± 0.157* | *0.832 ± 0.156* | **0.134 ± 0.111** | *0.141 ± 0.113* |
| VPU | 0.871 ± 0.160 | 0.879 ± 0.155 | 0.863 ± 0.114 | 0.772 ± 0.187 | 0.767 ± 0.238 | 0.206 ± 0.170 | 0.196 ± 0.175 |
| VPU-MP(auto) | 0.889 ± 0.139 | 0.896 ± 0.134 | 0.873 ± 0.105 | 0.795 ± 0.169 | 0.782 ± 0.221 | 0.174 ± 0.117 | 0.159 ± 0.120 |
| VPU-MP(0.69) | *0.895 ± 0.131* | *0.902 ± 0.127* | *0.876 ± 0.103* | 0.820 ± 0.152 | **0.838 ± 0.140** | 0.150 ± 0.097 | 0.142 ± 0.107 |
| PN-Naive | 0.873 ± 0.164 | 0.877 ± 0.161 | 0.866 ± 0.110 | 0.632 ± 0.163 | 0.473 ± 0.337 | 0.386 ± 0.124 | 0.339 ± 0.142 |
| Oracle-PN | **0.953 ± 0.057** | **0.956 ± 0.057** | **0.919 ± 0.073** | **0.841 ± 0.161** | 0.827 ± 0.221 | *0.141 ± 0.157* | **0.136 ± 0.150** |

### Performance by Prior Regime

### Low Priors (π < 0.5)

*Mean ± Std across configurations with π ∈ {0.01, 0.1, 0.3}. **Bold** = best, *italic* = second-best.*

| Method | AUC ↑ | AP ↑ | Max F1 ↑ | Accuracy ↑ | F1 ↑ | ECE ↓ | Brier ↓ |
|--------|--------|--------|--------|--------|--------|--------|--------|
| VPU-nomix | 0.905 ± 0.119 | 0.914 ± 0.115 | 0.879 ± 0.099 | 0.695 ± 0.203 | 0.499 ± 0.379 | 0.310 ± 0.206 | 0.287 ± 0.206 |
| VPU-nomix-MP(auto) | 0.905 ± 0.120 | 0.914 ± 0.116 | 0.881 ± 0.097 | 0.816 ± 0.160 | 0.774 ± 0.250 | 0.137 ± 0.132 | 0.147 ± 0.129 |
| VPU-nomix-MP(0.69) | 0.904 ± 0.120 | 0.912 ± 0.116 | 0.880 ± 0.097 | *0.837 ± 0.141* | *0.834 ± 0.164* | **0.114 ± 0.107** | *0.129 ± 0.108* |
| VPU | 0.902 ± 0.125 | 0.909 ± 0.123 | 0.878 ± 0.101 | 0.801 ± 0.167 | 0.718 ± 0.305 | 0.165 ± 0.145 | 0.160 ± 0.147 |
| VPU-MP(auto) | 0.903 ± 0.124 | 0.910 ± 0.123 | 0.880 ± 0.099 | 0.823 ± 0.156 | 0.796 ± 0.224 | 0.134 ± 0.100 | 0.132 ± 0.109 |
| VPU-MP(0.69) | 0.906 ± 0.117 | 0.914 ± 0.114 | 0.882 ± 0.097 | **0.838 ± 0.142** | **0.842 ± 0.146** | *0.124 ± 0.090* | **0.124 ± 0.100** |
| PN-Naive | *0.918 ± 0.095* | *0.925 ± 0.089* | *0.886 ± 0.089* | 0.620 ± 0.160 | 0.371 ± 0.320 | 0.422 ± 0.111 | 0.370 ± 0.142 |
| Oracle-PN | **0.952 ± 0.054** | **0.958 ± 0.049** | **0.915 ± 0.072** | 0.825 ± 0.169 | 0.750 ± 0.297 | 0.159 ± 0.165 | 0.150 ± 0.156 |

### High Priors (π ≥ 0.5)

*Mean ± Std across configurations with π ∈ {0.5, 0.7, 0.9, 0.99}. **Bold** = best, *italic* = second-best.*

| Method | AUC ↑ | AP ↑ | Max F1 ↑ | Accuracy ↑ | F1 ↑ | ECE ↓ | Brier ↓ |
|--------|--------|--------|--------|--------|--------|--------|--------|
| VPU-nomix | 0.844 ± 0.177 | 0.852 ± 0.172 | 0.849 ± 0.123 | 0.746 ± 0.199 | 0.789 ± 0.190 | 0.232 ± 0.193 | 0.230 ± 0.192 |
| VPU-nomix-MP(auto) | 0.872 ± 0.156 | 0.879 ± 0.149 | 0.864 ± 0.115 | 0.779 ± 0.171 | 0.773 ± 0.208 | 0.182 ± 0.130 | 0.179 ± 0.127 |
| VPU-nomix-MP(0.69) | 0.879 ± 0.149 | 0.885 ± 0.144 | 0.867 ± 0.114 | *0.807 ± 0.167* | 0.831 ± 0.150 | *0.149 ± 0.111* | *0.150 ± 0.116* |
| VPU | 0.848 ± 0.179 | 0.856 ± 0.171 | 0.851 ± 0.122 | 0.751 ± 0.198 | 0.804 ± 0.163 | 0.236 ± 0.181 | 0.223 ± 0.189 |
| VPU-MP(auto) | 0.879 ± 0.149 | 0.885 ± 0.141 | 0.867 ± 0.108 | 0.773 ± 0.176 | 0.772 ± 0.217 | 0.203 ± 0.121 | 0.179 ± 0.124 |
| VPU-MP(0.69) | *0.887 ± 0.141* | *0.893 ± 0.135* | *0.872 ± 0.107* | 0.806 ± 0.158 | *0.834 ± 0.136* | 0.169 ± 0.098 | 0.155 ± 0.111 |
| PN-Naive | 0.840 ± 0.194 | 0.841 ± 0.190 | 0.851 ± 0.122 | 0.642 ± 0.164 | 0.549 ± 0.330 | 0.358 ± 0.126 | 0.315 ± 0.137 |
| Oracle-PN | **0.955 ± 0.059** | **0.955 ± 0.061** | **0.921 ± 0.074** | **0.853 ± 0.153** | **0.884 ± 0.108** | **0.128 ± 0.150** | **0.125 ± 0.144** |

### Performance by Label Frequency Regime

### Low Label Frequency (c < 0.5)

*Mean ± Std across configurations with c ∈ {0.01, 0.1, 0.3}. **Bold** = best, *italic* = second-best.*

| Method | AUC ↑ | AP ↑ | Max F1 ↑ | Accuracy ↑ | F1 ↑ | ECE ↓ | Brier ↓ |
|--------|--------|--------|--------|--------|--------|--------|--------|
| VPU-nomix | 0.832 ± 0.170 | 0.843 ± 0.168 | 0.833 ± 0.120 | 0.670 ± 0.190 | 0.572 ± 0.330 | 0.319 ± 0.191 | 0.308 ± 0.190 |
| VPU-nomix-MP(auto) | 0.842 ± 0.165 | 0.853 ± 0.161 | 0.839 ± 0.117 | 0.743 ± 0.175 | 0.700 ± 0.258 | 0.181 ± 0.136 | 0.194 ± 0.131 |
| VPU-nomix-MP(0.69) | 0.843 ± 0.163 | 0.854 ± 0.160 | 0.840 ± 0.117 | 0.771 ± 0.169 | 0.777 ± 0.184 | *0.155 ± 0.118* | *0.172 ± 0.118* |
| VPU | 0.831 ± 0.176 | 0.843 ± 0.171 | 0.834 ± 0.120 | 0.726 ± 0.192 | 0.682 ± 0.295 | 0.241 ± 0.173 | 0.235 ± 0.178 |
| VPU-MP(auto) | 0.845 ± 0.162 | 0.855 ± 0.159 | 0.840 ± 0.115 | 0.752 ± 0.175 | 0.729 ± 0.237 | 0.182 ± 0.120 | 0.186 ± 0.124 |
| VPU-MP(0.69) | *0.847 ± 0.159* | *0.857 ± 0.157* | *0.841 ± 0.116* | *0.773 ± 0.168* | *0.791 ± 0.162* | 0.167 ± 0.107 | 0.173 ± 0.116 |
| PN-Naive | 0.822 ± 0.187 | 0.830 ± 0.182 | 0.831 ± 0.117 | 0.636 ± 0.153 | 0.535 ± 0.313 | 0.470 ± 0.093 | 0.449 ± 0.100 |
| Oracle-PN | **0.954 ± 0.054** | **0.957 ± 0.055** | **0.919 ± 0.072** | **0.839 ± 0.163** | **0.821 ± 0.229** | **0.143 ± 0.159** | **0.137 ± 0.152** |

### High Label Frequency (c ≥ 0.5)

*Mean ± Std across configurations with c ∈ {0.5, 0.7, 0.9, 0.99}. **Bold** = best, *italic* = second-best.*

| Method | AUC ↑ | AP ↑ | Max F1 ↑ | Accuracy ↑ | F1 ↑ | ECE ↓ | Brier ↓ |
|--------|--------|--------|--------|--------|--------|--------|--------|
| VPU-nomix | 0.898 ± 0.141 | 0.905 ± 0.135 | 0.883 ± 0.105 | 0.765 ± 0.202 | 0.734 ± 0.295 | 0.225 ± 0.201 | 0.215 ± 0.198 |
| VPU-nomix-MP(auto) | 0.920 ± 0.111 | 0.924 ± 0.106 | 0.895 ± 0.094 | 0.834 ± 0.149 | 0.829 ± 0.182 | 0.150 ± 0.129 | 0.144 ± 0.123 |
| VPU-nomix-MP(0.69) | 0.924 ± 0.103 | 0.928 ± 0.098 | 0.897 ± 0.091 | **0.857 ± 0.136** | **0.874 ± 0.115** | **0.119 ± 0.102** | **0.118 ± 0.103** |
| VPU | 0.901 ± 0.140 | 0.906 ± 0.135 | 0.884 ± 0.104 | 0.807 ± 0.176 | 0.830 ± 0.158 | 0.179 ± 0.163 | 0.167 ± 0.167 |
| VPU-MP(auto) | 0.922 ± 0.108 | 0.926 ± 0.102 | 0.897 ± 0.089 | 0.827 ± 0.158 | 0.822 ± 0.198 | 0.167 ± 0.115 | 0.139 ± 0.113 |
| VPU-MP(0.69) | *0.931 ± 0.091* | *0.936 ± 0.084* | *0.902 ± 0.082* | *0.855 ± 0.129* | *0.873 ± 0.109* | *0.137 ± 0.087* | *0.118 ± 0.093* |
| PN-Naive | 0.912 ± 0.132 | 0.912 ± 0.132 | 0.893 ± 0.097 | 0.630 ± 0.170 | 0.426 ± 0.347 | 0.322 ± 0.104 | 0.256 ± 0.109 |
| Oracle-PN | **0.953 ± 0.059** | **0.956 ± 0.058** | **0.919 ± 0.074** | 0.843 ± 0.159 | 0.830 ± 0.215 | 0.140 ± 0.155 | 0.134 ± 0.148 |

---

## Method Rankings

### Phase 1: Fixed Prior Experiments

#### Overall Performance

*11 metrics*

| Method | Wins | Avg Rank |
|--------|------|----------|
| Oracle-PN | 9/11 | 1.18 |
| VPU-nomix-MP(0.69) | 0/11 | 2.64 |
| VPU-MP(0.69) | 0/11 | 4.27 |
| VPU | 0/11 | 4.36 |
| VPU-nomix-MP(auto) | 0/11 | 5.27 |
| VPU-MP(auto) | 0/11 | 6.64 |
| LBE | 0/11 | 8.09 |
| nnPU-SB | 1/11 | 8.18 |
| VPU-nomix | 1/11 | 8.27 |
| BBE-PU | 0/11 | 10.73 |
| Dist-PU | 0/11 | 12.27 |
| PN-Naive | 0/11 | 12.27 |
| nnPU | 0/11 | 12.55 |
| Robust-PU | 0/11 | 12.82 |
| Self-PU | 0/11 | 13.18 |
| P3Mix-C | 0/11 | 14.09 |
| P3Mix-E | 0/11 | 16.18 |

#### Threshold-Invariant Metrics

*3 metrics*

| Method | Wins | Avg Rank |
|--------|------|----------|
| Oracle-PN | 3/3 | 1.00 |
| VPU-nomix-MP(0.69) | 0/3 | 2.00 |
| VPU-nomix-MP(auto) | 0/3 | 3.00 |
| VPU-MP(0.69) | 0/3 | 4.00 |
| VPU | 0/3 | 5.33 |
| VPU-MP(auto) | 0/3 | 5.67 |
| VPU-nomix | 0/3 | 7.00 |
| nnPU-SB | 0/3 | 8.33 |
| LBE | 0/3 | 9.33 |
| PN-Naive | 0/3 | 9.33 |
| BBE-PU | 0/3 | 11.00 |
| Dist-PU | 0/3 | 12.00 |
| Robust-PU | 0/3 | 13.00 |
| nnPU | 0/3 | 14.33 |
| Self-PU | 0/3 | 14.67 |
| P3Mix-C | 0/3 | 16.00 |
| P3Mix-E | 0/3 | 17.00 |

#### Threshold-Dependent Metrics

*4 metrics*

| Method | Wins | Avg Rank |
|--------|------|----------|
| Oracle-PN | 2/4 | 1.50 |
| VPU-nomix-MP(0.69) | 0/4 | 3.50 |
| VPU | 0/4 | 4.75 |
| VPU-MP(0.69) | 0/4 | 5.00 |
| LBE | 0/4 | 7.00 |
| VPU-nomix-MP(auto) | 0/4 | 7.25 |
| VPU-MP(auto) | 0/4 | 7.25 |
| nnPU-SB | 1/4 | 7.75 |
| BBE-PU | 0/4 | 9.50 |
| VPU-nomix | 1/4 | 10.00 |
| PN-Naive | 0/4 | 10.50 |
| nnPU | 0/4 | 10.75 |
| Dist-PU | 0/4 | 11.00 |
| Robust-PU | 0/4 | 12.25 |
| Self-PU | 0/4 | 14.00 |
| P3Mix-C | 0/4 | 14.25 |
| P3Mix-E | 0/4 | 16.75 |

#### Calibration Metrics

*3 metrics*

| Method | Wins | Avg Rank |
|--------|------|----------|
| Oracle-PN | 3/3 | 1.00 |
| VPU-nomix-MP(0.69) | 0/3 | 2.33 |
| VPU | 0/3 | 3.33 |
| VPU-MP(0.69) | 0/3 | 3.67 |
| VPU-nomix-MP(auto) | 0/3 | 5.00 |
| VPU-MP(auto) | 0/3 | 7.00 |
| VPU-nomix | 0/3 | 7.33 |
| LBE | 0/3 | 7.67 |
| nnPU-SB | 0/3 | 9.00 |
| BBE-PU | 0/3 | 10.33 |
| Self-PU | 0/3 | 11.33 |
| Robust-PU | 0/3 | 12.00 |
| nnPU | 0/3 | 12.33 |
| P3Mix-C | 0/3 | 13.67 |
| Dist-PU | 0/3 | 14.00 |
| P3Mix-E | 0/3 | 16.00 |
| PN-Naive | 0/3 | 17.00 |

#### Cross-Entropy Metrics

*1 metrics*

| Method | Wins | Avg Rank |
|--------|------|----------|
| Oracle-PN | 1/1 | 1.00 |
| VPU-nomix-MP(0.69) | 0/1 | 2.00 |
| VPU | 0/1 | 3.00 |
| VPU-MP(0.69) | 0/1 | 4.00 |
| VPU-nomix-MP(auto) | 0/1 | 5.00 |
| VPU-MP(auto) | 0/1 | 6.00 |
| nnPU-SB | 0/1 | 7.00 |
| VPU-nomix | 0/1 | 8.00 |
| P3Mix-C | 0/1 | 9.00 |
| LBE | 0/1 | 10.00 |
| Self-PU | 0/1 | 11.00 |
| P3Mix-E | 0/1 | 12.00 |
| Dist-PU | 0/1 | 13.00 |
| PN-Naive | 0/1 | 14.00 |
| nnPU | 0/1 | 15.00 |
| BBE-PU | 0/1 | 16.00 |
| Robust-PU | 0/1 | 17.00 |

### Phase 3: Full Grid Experiments

#### Overall Performance

*11 metrics*

| Method | Wins | Avg Rank |
|--------|------|----------|
| Oracle-PN | 6/11 | 2.18 |
| VPU-MP(0.69) | 4/11 | 2.36 |
| VPU-nomix-MP(0.69) | 1/11 | 2.55 |
| VPU-MP(auto) | 0/11 | 4.00 |
| VPU-nomix-MP(auto) | 0/11 | 4.55 |
| VPU | 0/11 | 6.09 |
| VPU-nomix | 0/11 | 6.91 |
| PN-Naive | 0/11 | 7.36 |

#### Threshold-Invariant Metrics

*3 metrics*

| Method | Wins | Avg Rank |
|--------|------|----------|
| Oracle-PN | 3/3 | 1.00 |
| VPU-MP(0.69) | 0/3 | 2.00 |
| VPU-nomix-MP(0.69) | 0/3 | 3.00 |
| VPU-MP(auto) | 0/3 | 4.00 |
| VPU-nomix-MP(auto) | 0/3 | 5.00 |
| VPU | 0/3 | 6.67 |
| PN-Naive | 0/3 | 6.67 |
| VPU-nomix | 0/3 | 7.67 |

#### Threshold-Dependent Metrics

*4 metrics*

| Method | Wins | Avg Rank |
|--------|------|----------|
| Oracle-PN | 2/4 | 2.00 |
| VPU-nomix-MP(0.69) | 0/4 | 3.00 |
| VPU-MP(0.69) | 2/4 | 3.00 |
| VPU-nomix-MP(auto) | 0/4 | 4.25 |
| VPU-MP(auto) | 0/4 | 4.25 |
| VPU | 0/4 | 6.00 |
| VPU-nomix | 0/4 | 6.25 |
| PN-Naive | 0/4 | 7.25 |

#### Calibration Metrics

*3 metrics*

| Method | Wins | Avg Rank |
|--------|------|----------|
| VPU-nomix-MP(0.69) | 1/3 | 1.67 |
| VPU-MP(0.69) | 1/3 | 2.33 |
| Oracle-PN | 1/3 | 3.00 |
| VPU-MP(auto) | 0/3 | 4.00 |
| VPU-nomix-MP(auto) | 0/3 | 4.33 |
| VPU | 0/3 | 5.67 |
| VPU-nomix | 0/3 | 7.00 |
| PN-Naive | 0/3 | 8.00 |

#### Cross-Entropy Metrics

*1 metrics*

| Method | Wins | Avg Rank |
|--------|------|----------|
| VPU-MP(0.69) | 1/1 | 1.00 |
| VPU-nomix-MP(0.69) | 0/1 | 2.00 |
| VPU-MP(auto) | 0/1 | 3.00 |
| Oracle-PN | 0/1 | 4.00 |
| VPU-nomix-MP(auto) | 0/1 | 5.00 |
| VPU | 0/1 | 6.00 |
| VPU-nomix | 0/1 | 7.00 |
| PN-Naive | 0/1 | 8.00 |

### Phase 3: By Hyperparameter Regime

#### Low Priors (π < 0.5)

*11 metrics*

| Method | Wins | Avg Rank |
|--------|------|----------|
| VPU-MP(0.69) | 6/11 | 2.27 |
| VPU-nomix-MP(0.69) | 1/11 | 3.55 |
| Oracle-PN | 4/11 | 3.55 |
| VPU-MP(auto) | 0/11 | 4.18 |
| VPU-nomix-MP(auto) | 0/11 | 4.36 |
| PN-Naive | 0/11 | 5.82 |
| VPU | 0/11 | 6.09 |
| VPU-nomix | 0/11 | 6.18 |

#### High Priors (π ≥ 0.5)

*11 metrics*

| Method | Wins | Avg Rank |
|--------|------|----------|
| Oracle-PN | 9/11 | 1.73 |
| VPU-MP(0.69) | 2/11 | 2.45 |
| VPU-nomix-MP(0.69) | 0/11 | 2.91 |
| VPU-MP(auto) | 0/11 | 4.36 |
| VPU-nomix-MP(auto) | 0/11 | 4.91 |
| VPU | 0/11 | 5.55 |
| VPU-nomix | 0/11 | 6.27 |
| PN-Naive | 0/11 | 7.82 |

#### Low Label Frequency (c < 0.5)

*11 metrics*

| Method | Wins | Avg Rank |
|--------|------|----------|
| Oracle-PN | 9/11 | 1.55 |
| VPU-MP(0.69) | 2/11 | 2.45 |
| VPU-nomix-MP(0.69) | 0/11 | 3.27 |
| VPU-MP(auto) | 0/11 | 3.64 |
| VPU-nomix-MP(auto) | 0/11 | 4.64 |
| VPU | 0/11 | 6.00 |
| VPU-nomix | 0/11 | 6.45 |
| PN-Naive | 0/11 | 8.00 |

#### High Label Frequency (c ≥ 0.5)

*11 metrics*

| Method | Wins | Avg Rank |
|--------|------|----------|
| VPU-MP(0.69) | 3/11 | 2.18 |
| VPU-nomix-MP(0.69) | 4/11 | 2.27 |
| Oracle-PN | 4/11 | 2.91 |
| VPU-MP(auto) | 0/11 | 4.27 |
| VPU-nomix-MP(auto) | 0/11 | 4.64 |
| VPU | 0/11 | 5.73 |
| PN-Naive | 0/11 | 6.82 |
| VPU-nomix | 0/11 | 7.18 |

---

## Key Findings

### Overall Performance

**Phase 1 (Fixed Prior):**
  1. **Oracle-PN**: AUC = 0.971 ± 0.037
  2. **VPU-nomix-MP(0.69)**: AUC = 0.954 ± 0.047
  3. **VPU-nomix-MP(auto)**: AUC = 0.953 ± 0.048

**Phase 3 (Full Grid):**
  1. **Oracle-PN**: AUC = 0.953 ± 0.057
  2. **VPU-MP(0.69)**: AUC = 0.895 ± 0.131
  3. **VPU-nomix-MP(0.69)**: AUC = 0.890 ± 0.138

### VPU Method Variants

**Phase 3 AUC Performance:**
  1. **VPU-MP(0.69)**: 0.895 ± 0.131
  2. **VPU-nomix-MP(0.69)**: 0.890 ± 0.138
  3. **VPU-MP(auto)**: 0.889 ± 0.139
  4. **VPU-nomix-MP(auto)**: 0.887 ± 0.142
  5. **VPU**: 0.871 ± 0.160
  6. **VPU-nomix**: 0.870 ± 0.158

### Method Stability

Standard deviation of AUC across Phase 3 configurations (lower = more stable):

  1. **Oracle-PN**: σ = 0.0571 (mean AUC = 0.953)
  2. **VPU-MP(0.69)**: σ = 0.1314 (mean AUC = 0.895)
  3. **VPU-nomix-MP(0.69)**: σ = 0.1382 (mean AUC = 0.890)
  4. **VPU-MP(auto)**: σ = 0.1392 (mean AUC = 0.889)
  5. **VPU-nomix-MP(auto)**: σ = 0.1422 (mean AUC = 0.887)
