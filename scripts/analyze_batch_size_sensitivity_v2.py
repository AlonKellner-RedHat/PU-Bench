#!/usr/bin/env python3
"""
Analyze batch size sensitivity using THRESHOLD-INDEPENDENT metrics.

Focuses on:
- AP (Average Precision) - primary metric
- max_f1 - best achievable F1
- AUC - discrimination ability
- A-NICE - calibration quality
- Convergence speed (epochs, time)

F1 at fixed threshold is de-emphasized as it's unreliable for uncalibrated methods.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# Configuration
SEEDS = [42, 123, 456, 789, 2024]
DATASETS = ['MNIST', 'IMDB', 'Mushrooms']
RESULTS_DIR = Path('results')

# Batch size variants
BATCH_SIZE_METHODS = {
    'vpu': 256,
    'vpu_batch1': 1,
    'vpu_batch2': 2,
    'vpu_batch4': 4,
    'vpu_batch8': 8,
    'vpu_batch16': 16,
    'vpu_batch64': 64,
    'vpu_mean': 256,
    'vpu_mean_batch1': 1,
    'vpu_mean_batch2': 2,
    'vpu_mean_batch4': 4,
    'vpu_mean_batch8': 8,
    'vpu_mean_batch16': 16,
    'vpu_mean_batch64': 64,
}


def extract_batch_size(method_name: str) -> int:
    """Extract batch size from method name."""
    return BATCH_SIZE_METHODS.get(method_name, 256)


def extract_base_method(method_name: str) -> str:
    """Extract base method (vpu or vpu_mean) from method name."""
    if 'vpu_mean' in method_name:
        return 'vpu_mean'
    elif 'vpu' in method_name:
        return 'vpu'
    return method_name


def load_batch_size_results() -> pd.DataFrame:
    """Load results for batch size experiments."""
    records = []

    for seed in SEEDS:
        seed_dir = RESULTS_DIR / f'seed_{seed}'
        if not seed_dir.exists():
            continue

        for config_file in seed_dir.glob('*.json'):
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)
            except:
                continue

            experiment = data.get('experiment', '')
            parts = experiment.split('_')
            dataset = parts[0]

            if dataset not in DATASETS:
                continue

            # Extract c value
            c_value = None
            for part in parts:
                if part.startswith('c') and not part.startswith('case'):
                    try:
                        c_value = float(part[1:])
                    except:
                        pass

            for method_name in BATCH_SIZE_METHODS.keys():
                method_data = data.get('runs', {}).get(method_name)
                if not method_data or 'best' not in method_data:
                    continue

                if 'metrics' not in method_data['best']:
                    continue

                metrics = method_data['best']['metrics']
                best_info = method_data['best']

                # Extract batch size and base method
                batch_size = extract_batch_size(method_name)
                base_method = extract_base_method(method_name)

                record = {
                    'experiment': experiment,
                    'dataset': dataset,
                    'seed': seed,
                    'c': c_value,
                    'method': method_name,
                    'base_method': base_method,
                    'batch_size': batch_size,
                    # Threshold-independent metrics (PRIMARY)
                    'test_ap': metrics.get('test_ap'),
                    'test_max_f1': metrics.get('test_max_f1'),
                    'test_auc': metrics.get('test_auc'),
                    'test_anice': metrics.get('test_anice'),
                    # Convergence metrics
                    'best_epoch': best_info.get('epoch'),
                    'time_to_best': best_info.get('time_to_best'),
                    # Threshold-dependent (SECONDARY)
                    'test_f1': metrics.get('test_f1'),
                    'test_recall': metrics.get('test_recall'),
                    'test_precision': metrics.get('test_precision'),
                }

                records.append(record)

    df = pd.DataFrame(records)

    if len(df) == 0:
        print("WARNING: No batch size results found!")
        return df

    print(f"Loaded {len(df)} results")
    print(f"Datasets: {sorted(df['dataset'].unique())}")
    print(f"Batch sizes: {sorted(df['batch_size'].unique())}")
    print(f"Base methods: {sorted(df['base_method'].unique())}")

    # Check threshold-independent metrics
    ap_count = df['test_ap'].notna().sum()
    print(f"\n✓ Results with AP/max_f1: {ap_count}/{len(df)} ({ap_count/len(df)*100:.1f}%)")

    return df


def analyze_overall_performance(df: pd.DataFrame) -> str:
    """Analyze performance using threshold-independent metrics."""
    output = []
    output.append("=" * 90)
    output.append("BATCH SIZE PERFORMANCE - THRESHOLD-INDEPENDENT METRICS")
    output.append("=" * 90)
    output.append("")
    output.append("Primary Metrics:")
    output.append("  AP (Average Precision) - Discrimination across all thresholds")
    output.append("  max_f1 - Best achievable F1 with optimal threshold")
    output.append("  AUC - Overall discrimination ability")
    output.append("  A-NICE - Calibration quality (lower is better)")
    output.append("")

    # Performance table
    output.append("Performance by Batch Size")
    output.append("-" * 90)
    output.append(f"{'Method':<15} {'Batch':<8} {'AP':<10} {'max_f1':<10} {'AUC':<10} {'A-NICE':<10}")
    output.append("-" * 90)

    for base_method in ['vpu', 'vpu_mean']:
        for batch_size in sorted(df['batch_size'].unique()):
            df_subset = df[(df['base_method'] == base_method) &
                          (df['batch_size'] == batch_size)]

            if len(df_subset) == 0:
                continue

            ap_mean = df_subset['test_ap'].mean()
            max_f1_mean = df_subset['test_max_f1'].mean()
            auc_mean = df_subset['test_auc'].mean()
            anice_mean = df_subset['test_anice'].mean()

            output.append(
                f"{base_method:<15} {batch_size:<8} {ap_mean:>8.4f}   "
                f"{max_f1_mean:>8.4f}   {auc_mean:>8.4f}   {anice_mean:>8.4f}"
            )

    # Improvement/degradation from baseline (batch 256)
    output.append("")
    output.append("")
    output.append("Change from Baseline (Batch 256)")
    output.append("Positive values = improvement, Negative values = degradation")
    output.append("-" * 90)
    output.append(f"{'Method':<15} {'Batch':<8} {'AP Δ':<12} {'max_f1 Δ':<12} {'AUC Δ':<12}")
    output.append("-" * 90)

    for base_method in ['vpu', 'vpu_mean']:
        # Get baseline performance (batch 256)
        df_baseline = df[(df['base_method'] == base_method) &
                        (df['batch_size'] == 256)]

        if len(df_baseline) == 0:
            continue

        baseline_ap = df_baseline['test_ap'].mean()
        baseline_max_f1 = df_baseline['test_max_f1'].mean()
        baseline_auc = df_baseline['test_auc'].mean()

        for batch_size in sorted(df['batch_size'].unique()):
            if batch_size == 256:
                continue

            df_subset = df[(df['base_method'] == base_method) &
                          (df['batch_size'] == batch_size)]

            if len(df_subset) == 0:
                continue

            ap_mean = df_subset['test_ap'].mean()
            max_f1_mean = df_subset['test_max_f1'].mean()
            auc_mean = df_subset['test_auc'].mean()

            # Change (positive = improvement)
            ap_change = ((ap_mean - baseline_ap) / baseline_ap) * 100 if not np.isnan(baseline_ap) else float('nan')
            max_f1_change = ((max_f1_mean - baseline_max_f1) / baseline_max_f1) * 100 if not np.isnan(baseline_max_f1) else float('nan')
            auc_change = ((auc_mean - baseline_auc) / baseline_auc) * 100 if not np.isnan(baseline_auc) else float('nan')

            output.append(
                f"{base_method:<15} {batch_size:<8} {ap_change:>+10.1f}%   "
                f"{max_f1_change:>+10.1f}%   {auc_change:>+10.1f}%"
            )

    return "\n".join(output)


def analyze_convergence_speed(df: pd.DataFrame) -> str:
    """Analyze convergence speed by batch size."""
    output = []
    output.append("\n" + "=" * 90)
    output.append("CONVERGENCE SPEED ANALYSIS")
    output.append("=" * 90)
    output.append("")

    # Filter to only results with convergence data
    df_conv = df[df['best_epoch'].notna()].copy()

    if len(df_conv) == 0:
        output.append("No convergence data available")
        return "\n".join(output)

    output.append("Epochs to Best Performance")
    output.append("-" * 90)
    output.append(f"{'Method':<15} {'Batch':<8} {'Epochs':<12} {'Time (s)':<12}")
    output.append("-" * 90)

    for base_method in ['vpu', 'vpu_mean']:
        for batch_size in sorted(df_conv['batch_size'].unique()):
            df_subset = df_conv[(df_conv['base_method'] == base_method) &
                               (df_conv['batch_size'] == batch_size)]

            if len(df_subset) == 0:
                continue

            epoch_mean = df_subset['best_epoch'].mean()
            time_mean = df_subset['time_to_best'].mean()

            output.append(
                f"{base_method:<15} {batch_size:<8} {epoch_mean:>10.1f}   "
                f"{time_mean:>10.1f}"
            )

    return "\n".join(output)


def analyze_by_dataset(df: pd.DataFrame) -> str:
    """Analyze batch size effects by dataset."""
    output = []
    output.append("\n" + "=" * 90)
    output.append("DATASET-SPECIFIC ANALYSIS")
    output.append("=" * 90)

    for dataset in sorted(df['dataset'].unique()):
        df_dataset = df[df['dataset'] == dataset]

        output.append(f"\n{dataset}")
        output.append("-" * 90)
        output.append(f"{'Method':<15} {'Batch':<8} {'AP':<10} {'max_f1':<10} {'AUC':<10}")
        output.append("-" * 90)

        for base_method in ['vpu', 'vpu_mean']:
            for batch_size in sorted(df_dataset['batch_size'].unique()):
                df_subset = df_dataset[(df_dataset['base_method'] == base_method) &
                                      (df_dataset['batch_size'] == batch_size)]

                if len(df_subset) == 0:
                    continue

                ap_mean = df_subset['test_ap'].mean()
                max_f1_mean = df_subset['test_max_f1'].mean()
                auc_mean = df_subset['test_auc'].mean()

                output.append(
                    f"{base_method:<15} {batch_size:<8} {ap_mean:>8.4f}   "
                    f"{max_f1_mean:>8.4f}   {auc_mean:>8.4f}"
                )

    return "\n".join(output)


def statistical_analysis(df: pd.DataFrame) -> str:
    """Statistical test using threshold-independent metrics."""
    output = []
    output.append("\n" + "=" * 90)
    output.append("STATISTICAL ANALYSIS")
    output.append("=" * 90)
    output.append("")

    # Compare batch 1 vs 256 using AP
    vpu_ap_changes = []
    vpu_mean_ap_changes = []

    for dataset in df['dataset'].unique():
        for seed in df['seed'].unique():
            # VPU
            df_vpu_256 = df[(df['base_method'] == 'vpu') &
                           (df['batch_size'] == 256) &
                           (df['dataset'] == dataset) &
                           (df['seed'] == seed)]

            df_vpu_1 = df[(df['base_method'] == 'vpu') &
                          (df['batch_size'] == 1) &
                          (df['dataset'] == dataset) &
                          (df['seed'] == seed)]

            if len(df_vpu_256) > 0 and len(df_vpu_1) > 0:
                ap_256 = df_vpu_256['test_ap'].iloc[0]
                ap_1 = df_vpu_1['test_ap'].iloc[0]
                if not np.isnan(ap_256) and not np.isnan(ap_1):
                    change = ((ap_1 - ap_256) / ap_256) * 100
                    vpu_ap_changes.append(change)

            # VPU-Mean
            df_mean_256 = df[(df['base_method'] == 'vpu_mean') &
                            (df['batch_size'] == 256) &
                            (df['dataset'] == dataset) &
                            (df['seed'] == seed)]

            df_mean_1 = df[(df['base_method'] == 'vpu_mean') &
                           (df['batch_size'] == 1) &
                           (df['dataset'] == dataset) &
                           (df['seed'] == seed)]

            if len(df_mean_256) > 0 and len(df_mean_1) > 0:
                ap_256 = df_mean_256['test_ap'].iloc[0]
                ap_1 = df_mean_1['test_ap'].iloc[0]
                if not np.isnan(ap_256) and not np.isnan(ap_1):
                    change = ((ap_1 - ap_256) / ap_256) * 100
                    vpu_mean_ap_changes.append(change)

    if len(vpu_ap_changes) > 0 and len(vpu_mean_ap_changes) > 0:
        # Independent t-test
        t_stat, p_value = stats.ttest_ind(vpu_ap_changes, vpu_mean_ap_changes)

        output.append(f"AP Change: Batch 1 vs 256")
        output.append(f"  VPU:       {np.mean(vpu_ap_changes):>+6.2f}% ± {np.std(vpu_ap_changes):>5.2f}%")
        output.append(f"  VPU-Mean:  {np.mean(vpu_mean_ap_changes):>+6.2f}% ± {np.std(vpu_mean_ap_changes):>5.2f}%")
        output.append(f"  t-statistic: {t_stat:.3f}")
        output.append(f"  p-value: {p_value:.4f}")

        if p_value < 0.05:
            output.append(f"\n✓ Statistically significant difference (p<0.05)")
        else:
            output.append(f"\n○ No significant difference (p≥0.05)")
    else:
        output.append("Insufficient data for statistical test")

    return "\n".join(output)


def generate_recommendations(df: pd.DataFrame) -> str:
    """Generate recommendations based on threshold-independent analysis."""
    output = []
    output.append("\n" + "=" * 90)
    output.append("RECOMMENDATIONS")
    output.append("=" * 90)
    output.append("")

    # Find best batch size by AP
    for base_method in ['vpu', 'vpu_mean']:
        batch_perf = []
        for batch_size in sorted(df['batch_size'].unique()):
            df_subset = df[(df['base_method'] == base_method) &
                          (df['batch_size'] == batch_size)]
            if len(df_subset) > 0:
                ap_mean = df_subset['test_ap'].mean()
                max_f1_mean = df_subset['test_max_f1'].mean()
                batch_perf.append((batch_size, ap_mean, max_f1_mean))

        if batch_perf:
            best_batch, best_ap, best_max_f1 = max(batch_perf, key=lambda x: x[1])
            output.append(f"{base_method.upper()}:")
            output.append(f"  Best batch size by AP: {best_batch}")
            output.append(f"    AP: {best_ap:.4f}")
            output.append(f"    max_f1: {best_max_f1:.4f}")
            output.append("")

    return "\n".join(output)


def main():
    """Run batch size sensitivity analysis."""
    print("Loading batch size experiment results...")
    df = load_batch_size_results()

    if len(df) == 0:
        print("\nNo batch size results found!")
        print("Run: ./scripts/run_extreme_batch_sizes.sh")
        return

    print("\nGenerating analysis...")

    sections = [
        analyze_overall_performance(df),
        analyze_convergence_speed(df),
        analyze_by_dataset(df),
        statistical_analysis(df),
        generate_recommendations(df),
    ]

    full_report = "\n\n".join(sections)

    output_file = RESULTS_DIR / 'BATCH_SIZE_ANALYSIS_THRESHOLD_INDEPENDENT.md'
    with open(output_file, 'w') as f:
        f.write(full_report)

    print(f"\n✓ Analysis complete!")
    print(f"  Report saved to: {output_file}")
    print(f"\n{generate_recommendations(df)}")


if __name__ == '__main__':
    main()
