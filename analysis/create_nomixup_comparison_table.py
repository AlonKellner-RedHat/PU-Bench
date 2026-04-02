#!/usr/bin/env python3
"""
Create comparison table for vpu_nomixup variants including convergence speed
"""

import pandas as pd
import numpy as np

# Load comprehensive metrics
df = pd.read_csv('results/comprehensive_metrics.csv')

# Filter for the three nomixup variants
methods = ['vpu_nomixup', 'vpu_nomixup_mean', 'vpu_nomixup_mean_prior']
df_nomixup = df[df['method'].isin(methods)].copy()

print(f"Creating comparison table for {len(df_nomixup)} experiments")
print(f"  - {len(df_nomixup) // 3} experiments per method")
print(f"  - Methods: {methods}")
print()

# Define metrics with their properties
# (column_name, display_name, higher_is_better)
metrics = [
    # Performance metrics
    ('test_f1', 'F1', True),
    ('test_max_f1', 'Max F1', True),
    ('test_ap', 'AP', True),
    ('test_auc', 'AUC', True),
    ('test_accuracy', 'Accuracy', True),
    ('test_precision', 'Precision', True),
    ('test_recall', 'Recall', True),

    # Calibration metrics
    ('test_anice', 'A-NICE', False),
    ('test_snice', 'S-NICE', False),
    ('test_ece', 'ECE', False),
    ('test_mce', 'MCE', False),
    ('test_brier', 'Brier', False),

    # Convergence speed
    ('convergence_epoch', 'Convergence Speed (epochs)', False),
]

# Calculate statistics for each method
results = []

for col, display_name, higher_is_better in metrics:
    row = {'Metric': display_name}

    # Calculate mean ± std for each method
    stats = {}
    for method in methods:
        method_data = df_nomixup[df_nomixup['method'] == method][col]
        mean_val = method_data.mean()
        std_val = method_data.std()

        stats[method] = {
            'mean': mean_val,
            'std': std_val,
            'formatted': f"{mean_val:.3f} ± {std_val:.3f}"
        }

        row[method] = stats[method]['formatted']

    # Calculate relative improvements
    baseline = stats['vpu_nomixup']['mean']
    mean_val = stats['vpu_nomixup_mean']['mean']
    prior_val = stats['vpu_nomixup_mean_prior']['mean']

    if higher_is_better:
        # For metrics where higher is better
        mean_improvement = ((mean_val - baseline) / baseline * 100) if baseline != 0 else 0
        prior_improvement = ((prior_val - baseline) / baseline * 100) if baseline != 0 else 0
    else:
        # For metrics where lower is better (calibration and convergence)
        mean_improvement = ((baseline - mean_val) / baseline * 100) if baseline != 0 else 0
        prior_improvement = ((baseline - prior_val) / baseline * 100) if baseline != 0 else 0

    row['Δ (mean vs baseline)'] = f"{mean_improvement:+.1f}%"
    row['Δ (prior vs baseline)'] = f"{prior_improvement:+.1f}%"

    results.append(row)

# Create DataFrame
df_table = pd.DataFrame(results)

# Save to CSV
output_path = 'results/vpu_nomixup_variants_comparison.csv'
df_table.to_csv(output_path, index=False)
print(f"✓ Saved comparison table to {output_path}")
print()

# Display table
print("="*100)
print("VPU NoMixup Variants Comparison Table")
print("="*100)
print(df_table.to_string(index=False))
print()

# Calculate aggregate improvements
print("="*100)
print("Aggregate Improvements vs vpu_nomixup baseline")
print("="*100)

# Performance metrics (first 7)
perf_metrics = df_table.iloc[:7]
perf_improvements_mean = []
perf_improvements_prior = []

for _, row in perf_metrics.iterrows():
    # Extract numeric value from percentage strings like "+1.9%"
    mean_imp = float(row['Δ (mean vs baseline)'].replace('%', ''))
    prior_imp = float(row['Δ (prior vs baseline)'].replace('%', ''))
    perf_improvements_mean.append(mean_imp)
    perf_improvements_prior.append(prior_imp)

print(f"Performance Metrics (avg of {len(perf_metrics)} metrics):")
print(f"  vpu_nomixup_mean:       {np.mean(perf_improvements_mean):+.2f}%")
print(f"  vpu_nomixup_mean_prior: {np.mean(perf_improvements_prior):+.2f}%")
print()

# Calibration metrics (next 5)
cal_metrics = df_table.iloc[7:12]
cal_improvements_mean = []
cal_improvements_prior = []

for _, row in cal_metrics.iterrows():
    mean_imp = float(row['Δ (mean vs baseline)'].replace('%', ''))
    prior_imp = float(row['Δ (prior vs baseline)'].replace('%', ''))
    cal_improvements_mean.append(mean_imp)
    cal_improvements_prior.append(prior_imp)

print(f"Calibration Metrics (avg of {len(cal_metrics)} metrics):")
print(f"  vpu_nomixup_mean:       {np.mean(cal_improvements_mean):+.2f}%")
print(f"  vpu_nomixup_mean_prior: {np.mean(cal_improvements_prior):+.2f}%")
print()

# Convergence speed
conv_metric = df_table.iloc[12]
print(f"Convergence Speed:")
print(f"  vpu_nomixup_mean:       {conv_metric['Δ (mean vs baseline)']}")
print(f"  vpu_nomixup_mean_prior: {conv_metric['Δ (prior vs baseline)']}")
print()

# Generate LaTeX table
print("="*100)
print("LaTeX Table Code")
print("="*100)
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\caption{Comparison of VPU-NoMixup variants across performance, calibration, and convergence metrics}")
print(r"\label{tab:vpu_nomixup_comparison}")
print(r"\begin{tabular}{lcccc}")
print(r"\hline")
print(r"Metric & vpu\_nomixup & vpu\_nomixup\_mean & vpu\_nomixup\_mean\_prior & $\Delta$ (prior) \\")
print(r"\hline")
print(r"\multicolumn{5}{l}{\textit{Performance Metrics}} \\")

for idx, row in df_table.iloc[:7].iterrows():
    metric = row['Metric']
    baseline = row['vpu_nomixup']
    mean_method = row['vpu_nomixup_mean']
    prior_method = row['vpu_nomixup_mean_prior']
    delta_prior = row['Δ (prior vs baseline)']
    print(f"{metric} & {baseline} & {mean_method} & {prior_method} & {delta_prior} \\\\")

print(r"\hline")
print(r"\multicolumn{5}{l}{\textit{Calibration Metrics}} \\")

for idx, row in df_table.iloc[7:12].iterrows():
    metric = row['Metric']
    baseline = row['vpu_nomixup']
    mean_method = row['vpu_nomixup_mean']
    prior_method = row['vpu_nomixup_mean_prior']
    delta_prior = row['Δ (prior vs baseline)']
    print(f"{metric} & {baseline} & {mean_method} & {prior_method} & {delta_prior} \\\\")

print(r"\hline")
print(r"\multicolumn{5}{l}{\textit{Convergence}} \\")

row = df_table.iloc[12]
metric = row['Metric']
baseline = row['vpu_nomixup']
mean_method = row['vpu_nomixup_mean']
prior_method = row['vpu_nomixup_mean_prior']
delta_prior = row['Δ (prior vs baseline)']
print(f"{metric} & {baseline} & {mean_method} & {prior_method} & {delta_prior} \\\\")

print(r"\hline")
print(r"\end{tabular}")
print(r"\end{table}")
print()
print("="*100)
