#!/usr/bin/env python3
"""
Analyze batch size sensitivity WITHOUT MixUp.

Compares:
- With MixUp (vpu, vpu_mean): batch 2-256
- Without MixUp (vpu_nomixup, vpu_nomixup_mean): batch 1-256

Key questions:
1. Does batch size 1 work without MixUp?
2. Does batch size sensitivity change without MixUp?
3. What is the performance cost/benefit of removing MixUp?
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

# Batch size variants - WITH and WITHOUT MixUp
BATCH_SIZE_METHODS = {
    # WITH MixUp
    'vpu': 256,
    'vpu_batch2': 2,
    'vpu_batch4': 4,
    'vpu_batch8': 8,
    'vpu_batch16': 16,
    'vpu_batch64': 64,
    'vpu_mean': 256,
    'vpu_mean_batch2': 2,
    'vpu_mean_batch4': 4,
    'vpu_mean_batch8': 8,
    'vpu_mean_batch16': 16,
    'vpu_mean_batch64': 64,
    # WITHOUT MixUp
    'vpu_nomixup': 256,
    'vpu_nomixup_batch1': 1,
    'vpu_nomixup_batch2': 2,
    'vpu_nomixup_batch4': 4,
    'vpu_nomixup_batch8': 8,
    'vpu_nomixup_batch16': 16,
    'vpu_nomixup_batch64': 64,
    'vpu_nomixup_mean': 256,
    'vpu_nomixup_mean_batch1': 1,
    'vpu_nomixup_mean_batch2': 2,
    'vpu_nomixup_mean_batch4': 4,
    'vpu_nomixup_mean_batch8': 8,
    'vpu_nomixup_mean_batch16': 16,
    'vpu_nomixup_mean_batch64': 64,
}


def extract_batch_size(method_name: str) -> int:
    """Extract batch size from method name."""
    return BATCH_SIZE_METHODS.get(method_name, 256)


def extract_base_method(method_name: str) -> str:
    """Extract base method and mixup status."""
    if 'vpu_nomixup_mean' in method_name:
        return 'vpu_nomixup_mean'
    elif 'vpu_nomixup' in method_name:
        return 'vpu_nomixup'
    elif 'vpu_mean' in method_name:
        return 'vpu_mean'
    elif 'vpu' in method_name:
        return 'vpu'
    return method_name


def has_mixup(method_name: str) -> bool:
    """Check if method uses MixUp."""
    return 'nomixup' not in method_name


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
                uses_mixup = has_mixup(method_name)

                record = {
                    'experiment': experiment,
                    'dataset': dataset,
                    'seed': seed,
                    'c': c_value,
                    'method': method_name,
                    'base_method': base_method,
                    'batch_size': batch_size,
                    'uses_mixup': uses_mixup,
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
    print(f"With MixUp: {df['uses_mixup'].sum()} results")
    print(f"Without MixUp: {(~df['uses_mixup']).sum()} results")

    # Check threshold-independent metrics
    ap_count = df['test_ap'].notna().sum()
    print(f"\n✓ Results with AP/max_f1: {ap_count}/{len(df)} ({ap_count/len(df)*100:.1f}%)")

    return df


def compare_mixup_vs_nomixup(df: pd.DataFrame) -> str:
    """Compare WITH vs WITHOUT MixUp."""
    output = []
    output.append("=" * 90)
    output.append("MIXUP vs NO-MIXUP COMPARISON")
    output.append("=" * 90)
    output.append("")

    # Overall comparison table
    output.append("Performance at Batch 256 (Baseline)")
    output.append("-" * 90)
    output.append(f"{'Method':<25} {'MixUp':<10} {'AP':<10} {'max_f1':<10} {'AUC':<10}")
    output.append("-" * 90)

    for base_type in ['vpu', 'vpu_mean']:
        # With MixUp
        df_mixup = df[(df['base_method'] == base_type) &
                     (df['batch_size'] == 256) &
                     (df['uses_mixup'] == True)]

        # Without MixUp
        df_nomixup = df[(df['base_method'] == f'{base_type}_nomixup') &
                       (df['batch_size'] == 256)]

        if len(df_mixup) > 0:
            ap_mixup = df_mixup['test_ap'].mean()
            max_f1_mixup = df_mixup['test_max_f1'].mean()
            auc_mixup = df_mixup['test_auc'].mean()
            output.append(
                f"{base_type:<25} {'Yes':<10} {ap_mixup:>8.4f}   "
                f"{max_f1_mixup:>8.4f}   {auc_mixup:>8.4f}"
            )

        if len(df_nomixup) > 0:
            ap_nomixup = df_nomixup['test_ap'].mean()
            max_f1_nomixup = df_nomixup['test_max_f1'].mean()
            auc_nomixup = df_nomixup['test_auc'].mean()
            output.append(
                f"{base_type:<25} {'No':<10} {ap_nomixup:>8.4f}   "
                f"{max_f1_nomixup:>8.4f}   {auc_nomixup:>8.4f}"
            )

            # Calculate difference
            if len(df_mixup) > 0:
                ap_diff = ((ap_nomixup - ap_mixup) / ap_mixup) * 100
                max_f1_diff = ((max_f1_nomixup - max_f1_mixup) / max_f1_mixup) * 100
                auc_diff = ((auc_nomixup - auc_mixup) / auc_mixup) * 100
                output.append(
                    f"{'  → Difference':<25} {'':<10} {ap_diff:>+7.1f}%  "
                    f"{max_f1_diff:>+7.1f}%  {auc_diff:>+7.1f}%"
                )

        output.append("")

    return "\n".join(output)


def analyze_batch1_performance(df: pd.DataFrame) -> str:
    """Analyze batch size 1 performance (only possible without MixUp)."""
    output = []
    output.append("\n" + "=" * 90)
    output.append("BATCH SIZE 1 ANALYSIS (No MixUp only)")
    output.append("=" * 90)
    output.append("")

    df_batch1 = df[df['batch_size'] == 1]

    if len(df_batch1) == 0:
        output.append("No batch size 1 results found")
        return "\n".join(output)

    output.append("Performance at Batch 1")
    output.append("-" * 90)
    output.append(f"{'Method':<25} {'AP':<10} {'max_f1':<10} {'AUC':<10}")
    output.append("-" * 90)

    for base_method in sorted(df_batch1['base_method'].unique()):
        df_subset = df_batch1[df_batch1['base_method'] == base_method]

        if len(df_subset) > 0:
            ap_mean = df_subset['test_ap'].mean()
            max_f1_mean = df_subset['test_max_f1'].mean()
            auc_mean = df_subset['test_auc'].mean()

            output.append(
                f"{base_method:<25} {ap_mean:>8.4f}   "
                f"{max_f1_mean:>8.4f}   {auc_mean:>8.4f}"
            )

    # Compare batch 1 vs batch 256 (no mixup)
    output.append("")
    output.append("Batch 1 vs Batch 256 Comparison (No MixUp)")
    output.append("-" * 90)
    output.append(f"{'Method':<25} {'AP Δ':<12} {'max_f1 Δ':<12} {'AUC Δ':<12}")
    output.append("-" * 90)

    for base_method in ['vpu_nomixup', 'vpu_nomixup_mean']:
        df_1 = df[(df['base_method'] == base_method) & (df['batch_size'] == 1)]
        df_256 = df[(df['base_method'] == base_method) & (df['batch_size'] == 256)]

        if len(df_1) > 0 and len(df_256) > 0:
            ap_1 = df_1['test_ap'].mean()
            ap_256 = df_256['test_ap'].mean()
            max_f1_1 = df_1['test_max_f1'].mean()
            max_f1_256 = df_256['test_max_f1'].mean()
            auc_1 = df_1['test_auc'].mean()
            auc_256 = df_256['test_auc'].mean()

            ap_change = ((ap_1 - ap_256) / ap_256) * 100
            max_f1_change = ((max_f1_1 - max_f1_256) / max_f1_256) * 100
            auc_change = ((auc_1 - auc_256) / auc_256) * 100

            output.append(
                f"{base_method:<25} {ap_change:>+10.1f}%  "
                f"{max_f1_change:>+10.1f}%  {auc_change:>+10.1f}%"
            )

    return "\n".join(output)


def analyze_batch_sensitivity_by_mixup(df: pd.DataFrame) -> str:
    """Compare batch size sensitivity with and without MixUp."""
    output = []
    output.append("\n" + "=" * 90)
    output.append("BATCH SIZE SENSITIVITY: WITH vs WITHOUT MIXUP")
    output.append("=" * 90)
    output.append("")

    # VPU variants
    output.append("VPU: Batch Size Performance")
    output.append("-" * 90)
    output.append(f"{'Batch':<10} {'With MixUp':<30} {'Without MixUp':<30}")
    output.append(f"{'Size':<10} {'AP':<10} {'max_f1':<10} {'AUC':<10} {'AP':<10} {'max_f1':<10} {'AUC':<10}")
    output.append("-" * 90)

    batch_sizes = [1, 2, 4, 8, 16, 64, 256]

    for batch_size in batch_sizes:
        df_mixup = df[(df['base_method'] == 'vpu') &
                     (df['batch_size'] == batch_size) &
                     (df['uses_mixup'] == True)]

        df_nomixup = df[(df['base_method'] == 'vpu_nomixup') &
                       (df['batch_size'] == batch_size)]

        row = f"{batch_size:<10}"

        if len(df_mixup) > 0:
            ap = df_mixup['test_ap'].mean()
            max_f1 = df_mixup['test_max_f1'].mean()
            auc = df_mixup['test_auc'].mean()
            row += f" {ap:>8.4f} {max_f1:>8.4f} {auc:>8.4f}  "
        else:
            row += " " * 30

        if len(df_nomixup) > 0:
            ap = df_nomixup['test_ap'].mean()
            max_f1 = df_nomixup['test_max_f1'].mean()
            auc = df_nomixup['test_auc'].mean()
            row += f" {ap:>8.4f} {max_f1:>8.4f} {auc:>8.4f}"

        output.append(row)

    output.append("")
    output.append("VPU-Mean: Batch Size Performance")
    output.append("-" * 90)
    output.append(f"{'Batch':<10} {'With MixUp':<30} {'Without MixUp':<30}")
    output.append(f"{'Size':<10} {'AP':<10} {'max_f1':<10} {'AUC':<10} {'AP':<10} {'max_f1':<10} {'AUC':<10}")
    output.append("-" * 90)

    for batch_size in batch_sizes:
        df_mixup = df[(df['base_method'] == 'vpu_mean') &
                     (df['batch_size'] == batch_size) &
                     (df['uses_mixup'] == True)]

        df_nomixup = df[(df['base_method'] == 'vpu_nomixup_mean') &
                       (df['batch_size'] == batch_size)]

        row = f"{batch_size:<10}"

        if len(df_mixup) > 0:
            ap = df_mixup['test_ap'].mean()
            max_f1 = df_mixup['test_max_f1'].mean()
            auc = df_mixup['test_auc'].mean()
            row += f" {ap:>8.4f} {max_f1:>8.4f} {auc:>8.4f}  "
        else:
            row += " " * 30

        if len(df_nomixup) > 0:
            ap = df_nomixup['test_ap'].mean()
            max_f1 = df_nomixup['test_max_f1'].mean()
            auc = df_nomixup['test_auc'].mean()
            row += f" {ap:>8.4f} {max_f1:>8.4f} {auc:>8.4f}"

        output.append(row)

    return "\n".join(output)


def generate_recommendations(df: pd.DataFrame) -> str:
    """Generate recommendations based on analysis."""
    output = []
    output.append("\n" + "=" * 90)
    output.append("RECOMMENDATIONS")
    output.append("=" * 90)
    output.append("")

    # Find best configurations
    for base_type in ['vpu', 'vpu_mean']:
        output.append(f"{base_type.upper()}:")

        # Best with MixUp
        df_mixup = df[(df['base_method'] == base_type) & (df['uses_mixup'] == True)]
        if len(df_mixup) > 0:
            best_idx = df_mixup.groupby('batch_size')['test_ap'].mean().idxmax()
            best_ap = df_mixup[df_mixup['batch_size'] == best_idx]['test_ap'].mean()
            best_max_f1 = df_mixup[df_mixup['batch_size'] == best_idx]['test_max_f1'].mean()
            output.append(f"  WITH MixUp - Best batch: {best_idx}")
            output.append(f"    AP: {best_ap:.4f}, max_f1: {best_max_f1:.4f}")

        # Best without MixUp
        df_nomixup = df[df['base_method'] == f'{base_type}_nomixup']
        if len(df_nomixup) > 0:
            best_idx = df_nomixup.groupby('batch_size')['test_ap'].mean().idxmax()
            best_ap = df_nomixup[df_nomixup['batch_size'] == best_idx]['test_ap'].mean()
            best_max_f1 = df_nomixup[df_nomixup['batch_size'] == best_idx]['test_max_f1'].mean()
            output.append(f"  WITHOUT MixUp - Best batch: {best_idx}")
            output.append(f"    AP: {best_ap:.4f}, max_f1: {best_max_f1:.4f}")

        output.append("")

    return "\n".join(output)


def main():
    """Run no-mixup batch size analysis."""
    print("Loading batch size experiment results (WITH and WITHOUT MixUp)...")
    df = load_batch_size_results()

    if len(df) == 0:
        print("\nNo batch size results found!")
        print("Run: ./scripts/run_nomixup_batch_sizes.sh")
        return

    print("\nGenerating analysis...")

    sections = [
        compare_mixup_vs_nomixup(df),
        analyze_batch1_performance(df),
        analyze_batch_sensitivity_by_mixup(df),
        generate_recommendations(df),
    ]

    full_report = "\n\n".join(sections)

    output_file = RESULTS_DIR / 'BATCH_SIZE_NOMIXUP_ANALYSIS.md'
    with open(output_file, 'w') as f:
        f.write(full_report)

    print(f"\n✓ Analysis complete!")
    print(f"  Report saved to: {output_file}")
    print(f"\n{generate_recommendations(df)}")


if __name__ == '__main__':
    main()
