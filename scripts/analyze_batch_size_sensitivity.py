#!/usr/bin/env python3
"""
Analyze batch size sensitivity across VPU variants.

Tests hypothesis: VPU log-transformation methods are more sensitive to batch
size than VPU-Mean variants.

Uses threshold-independent metrics (AP, max_f1) for fair comparison.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt

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
                    'test_f1': metrics.get('test_f1'),
                    'test_ap': metrics.get('test_ap'),
                    'test_max_f1': metrics.get('test_max_f1'),
                    'test_auc': metrics.get('test_auc'),
                    'test_recall': metrics.get('test_recall'),
                    'test_precision': metrics.get('test_precision'),
                    'best_epoch': method_data['best'].get('epoch'),
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

    # Check if new metrics exist
    ap_count = df['test_ap'].notna().sum()
    if ap_count == 0:
        print("\n⚠️  WARNING: No AP/max_f1 metrics found!")
        print("These experiments were run before AP was added.")
        print("F1-only analysis will be less reliable for uncalibrated methods.")
    else:
        print(f"\n✓ Results with AP: {ap_count}/{len(df)} ({ap_count/len(df)*100:.1f}%)")

    return df


def analyze_overall_sensitivity(df: pd.DataFrame) -> str:
    """Analyze overall batch size sensitivity."""
    output = []
    output.append("=" * 80)
    output.append("BATCH SIZE SENSITIVITY ANALYSIS")
    output.append("=" * 80)
    output.append("")
    output.append("Hypothesis: VPU log-transformation is more sensitive to batch size")
    output.append("than VPU-Mean's linear averaging.")
    output.append("")

    # Group by base method and batch size
    output.append("Performance by Batch Size")
    output.append("-" * 80)
    output.append(f"{'Method':<15} {'Batch':<8} {'F1':<10} {'AP':<10} {'Max F1':<10} {'AUC':<10}")
    output.append("-" * 80)

    for base_method in ['vpu', 'vpu_mean']:
        for batch_size in sorted(df['batch_size'].unique()):
            df_subset = df[(df['base_method'] == base_method) &
                          (df['batch_size'] == batch_size)]

            if len(df_subset) == 0:
                continue

            f1_mean = df_subset['test_f1'].mean()
            ap_mean = df_subset['test_ap'].mean()
            max_f1_mean = df_subset['test_max_f1'].mean()
            auc_mean = df_subset['test_auc'].mean()

            output.append(
                f"{base_method:<15} {batch_size:<8} {f1_mean:>8.4f}   "
                f"{ap_mean:>8.4f}   {max_f1_mean:>8.4f}   {auc_mean:>8.4f}"
            )

    # Compute performance degradation (batch 256 as baseline)
    output.append("")
    output.append("")
    output.append("Performance Degradation from Baseline (Batch 256)")
    output.append("-" * 80)
    output.append(f"{'Method':<15} {'Batch':<8} {'F1 Drop':<12} {'AP Drop':<12} {'Max F1 Drop':<12}")
    output.append("-" * 80)

    for base_method in ['vpu', 'vpu_mean']:
        # Get baseline performance (batch 256)
        df_baseline = df[(df['base_method'] == base_method) &
                        (df['batch_size'] == 256)]

        if len(df_baseline) == 0:
            continue

        baseline_f1 = df_baseline['test_f1'].mean()
        baseline_ap = df_baseline['test_ap'].mean()
        baseline_max_f1 = df_baseline['test_max_f1'].mean()

        for batch_size in sorted(df['batch_size'].unique()):
            if batch_size == 256:
                continue

            df_subset = df[(df['base_method'] == base_method) &
                          (df['batch_size'] == batch_size)]

            if len(df_subset) == 0:
                continue

            f1_mean = df_subset['test_f1'].mean()
            ap_mean = df_subset['test_ap'].mean()
            max_f1_mean = df_subset['test_max_f1'].mean()

            f1_drop = ((baseline_f1 - f1_mean) / baseline_f1) * 100
            ap_drop = ((baseline_ap - ap_mean) / baseline_ap) * 100 if not np.isnan(baseline_ap) else float('nan')
            max_f1_drop = ((baseline_max_f1 - max_f1_mean) / baseline_max_f1) * 100 if not np.isnan(baseline_max_f1) else float('nan')

            output.append(
                f"{base_method:<15} {batch_size:<8} {f1_drop:>10.1f}%   "
                f"{ap_drop:>10.1f}%   {max_f1_drop:>10.1f}%"
            )

    return "\n".join(output)


def analyze_by_dataset(df: pd.DataFrame) -> str:
    """Analyze batch size sensitivity by dataset."""
    output = []
    output.append("\n" + "=" * 80)
    output.append("BATCH SIZE SENSITIVITY BY DATASET")
    output.append("=" * 80)

    for dataset in sorted(df['dataset'].unique()):
        df_dataset = df[df['dataset'] == dataset]

        output.append(f"\n{dataset}")
        output.append("-" * 80)
        output.append(f"{'Method':<15} {'Batch':<8} {'F1':<10} {'AP':<10} {'Degradation':<12}")
        output.append("-" * 80)

        for base_method in ['vpu', 'vpu_mean']:
            # Baseline
            df_baseline = df_dataset[(df_dataset['base_method'] == base_method) &
                                    (df_dataset['batch_size'] == 256)]

            if len(df_baseline) == 0:
                continue

            baseline_f1 = df_baseline['test_f1'].mean()

            for batch_size in sorted(df_dataset['batch_size'].unique()):
                df_subset = df_dataset[(df_dataset['base_method'] == base_method) &
                                      (df_dataset['batch_size'] == batch_size)]

                if len(df_subset) == 0:
                    continue

                f1_mean = df_subset['test_f1'].mean()
                ap_mean = df_subset['test_ap'].mean()

                degradation = ((baseline_f1 - f1_mean) / baseline_f1) * 100

                output.append(
                    f"{base_method:<15} {batch_size:<8} {f1_mean:>8.4f}   "
                    f"{ap_mean:>8.4f}   {degradation:>10.1f}%"
                )

    return "\n".join(output)


def statistical_comparison(df: pd.DataFrame) -> str:
    """Statistical test: is VPU more sensitive than VPU-Mean?"""
    output = []
    output.append("\n" + "=" * 80)
    output.append("STATISTICAL TEST: VPU vs VPU-MEAN SENSITIVITY")
    output.append("=" * 80)
    output.append("")

    # Compute degradation for each method at batch 16 vs 256
    vpu_degradations = []
    vpu_mean_degradations = []

    for dataset in df['dataset'].unique():
        for seed in df['seed'].unique():
            # VPU
            df_vpu_256 = df[(df['base_method'] == 'vpu') &
                           (df['batch_size'] == 256) &
                           (df['dataset'] == dataset) &
                           (df['seed'] == seed)]

            df_vpu_16 = df[(df['base_method'] == 'vpu') &
                          (df['batch_size'] == 16) &
                          (df['dataset'] == dataset) &
                          (df['seed'] == seed)]

            if len(df_vpu_256) > 0 and len(df_vpu_16) > 0:
                f1_256 = df_vpu_256['test_f1'].iloc[0]
                f1_16 = df_vpu_16['test_f1'].iloc[0]
                degradation = ((f1_256 - f1_16) / f1_256) * 100
                vpu_degradations.append(degradation)

            # VPU-Mean
            df_mean_256 = df[(df['base_method'] == 'vpu_mean') &
                            (df['batch_size'] == 256) &
                            (df['dataset'] == dataset) &
                            (df['seed'] == seed)]

            df_mean_16 = df[(df['base_method'] == 'vpu_mean') &
                           (df['batch_size'] == 16) &
                           (df['dataset'] == dataset) &
                           (df['seed'] == seed)]

            if len(df_mean_256) > 0 and len(df_mean_16) > 0:
                f1_256 = df_mean_256['test_f1'].iloc[0]
                f1_16 = df_mean_16['test_f1'].iloc[0]
                degradation = ((f1_256 - f1_16) / f1_256) * 100
                vpu_mean_degradations.append(degradation)

    if len(vpu_degradations) == 0 or len(vpu_mean_degradations) == 0:
        output.append("Insufficient data for statistical test")
        return "\n".join(output)

    # Independent t-test (comparing degradations)
    t_stat, p_value = stats.ttest_ind(vpu_degradations, vpu_mean_degradations)

    vpu_mean_deg = np.mean(vpu_degradations)
    vpu_mean_mean_deg = np.mean(vpu_mean_degradations)

    output.append(f"Degradation at Batch 16 vs 256 (F1):")
    output.append(f"  VPU:       {vpu_mean_deg:>6.2f}% ± {np.std(vpu_degradations):>5.2f}%")
    output.append(f"  VPU-Mean:  {vpu_mean_mean_deg:>6.2f}% ± {np.std(vpu_mean_degradations):>5.2f}%")
    output.append(f"  Difference: {vpu_mean_deg - vpu_mean_mean_deg:+.2f}%")
    output.append(f"  t-statistic: {t_stat:.3f}")
    output.append(f"  p-value: {p_value:.4f}")

    if p_value < 0.05:
        if vpu_mean_deg > vpu_mean_mean_deg:
            output.append("\n✓ HYPOTHESIS CONFIRMED:")
            output.append("  VPU is significantly more sensitive to batch size (p<0.05)")
        else:
            output.append("\n✗ HYPOTHESIS REJECTED:")
            output.append("  VPU-Mean is more sensitive (unexpected!)")
    else:
        output.append("\n? INCONCLUSIVE:")
        output.append("  No significant difference detected (p≥0.05)")
        output.append("  May need more data or full batch size sweep")

    return "\n".join(output)


def generate_recommendations(df: pd.DataFrame) -> str:
    """Generate recommendations based on findings."""
    output = []
    output.append("\n" + "=" * 80)
    output.append("RECOMMENDATIONS")
    output.append("=" * 80)
    output.append("")

    # Compute key metrics
    vpu_16_f1 = df[(df['base_method'] == 'vpu') & (df['batch_size'] == 16)]['test_f1'].mean()
    vpu_256_f1 = df[(df['base_method'] == 'vpu') & (df['batch_size'] == 256)]['test_f1'].mean()
    vpu_mean_16_f1 = df[(df['base_method'] == 'vpu_mean') & (df['batch_size'] == 16)]['test_f1'].mean()
    vpu_mean_256_f1 = df[(df['base_method'] == 'vpu_mean') & (df['batch_size'] == 256)]['test_f1'].mean()

    vpu_deg = ((vpu_256_f1 - vpu_16_f1) / vpu_256_f1) * 100 if vpu_256_f1 > 0 else 0
    vpu_mean_deg = ((vpu_mean_256_f1 - vpu_mean_16_f1) / vpu_mean_256_f1) * 100 if vpu_mean_256_f1 > 0 else 0

    if vpu_deg > 5.0 and vpu_deg > 2 * vpu_mean_deg:
        output.append("✓ HYPOTHESIS VALIDATED:")
        output.append(f"  VPU degrades {vpu_deg:.1f}% at batch 16 (vs {vpu_mean_deg:.1f}% for VPU-Mean)")
        output.append("")
        output.append("NEXT STEPS:")
        output.append("1. Run full batch size sweep (Experiment 1)")
        output.append("2. Test very small batches (Experiment 2)")
        output.append("3. Test batch size × label frequency interaction (Experiment 3)")
        output.append("")
        output.append("PRACTICAL IMPLICATIONS:")
        output.append("- Memory-constrained environments: Use VPU-Mean")
        output.append("- Batch size < 64: VPU-Mean more stable")
        output.append("- Low label frequency: VPU-Mean more robust")
    else:
        output.append("? HYPOTHESIS NOT STRONGLY VALIDATED:")
        output.append(f"  VPU degradation: {vpu_deg:.1f}%")
        output.append(f"  VPU-Mean degradation: {vpu_mean_deg:.1f}%")
        output.append("")
        output.append("POSSIBLE REASONS:")
        output.append("- Batch sizes tested (16, 64, 256) may not be extreme enough")
        output.append("- Need more seeds for statistical power")
        output.append("- Specific datasets may not show the effect")
        output.append("")
        output.append("NEXT STEPS:")
        output.append("1. Try smaller batch sizes (8, 4)")
        output.append("2. Test on more datasets")
        output.append("3. Test at lower label frequency (c < 0.1)")

    return "\n".join(output)


def main():
    """Run batch size sensitivity analysis."""
    print("Loading batch size experiment results...")
    df = load_batch_size_results()

    if len(df) == 0:
        print("\nNo batch size results found!")
        print("Run: ./scripts/run_batch_size_validation.sh")
        return

    print("\nGenerating analysis...")

    sections = [
        analyze_overall_sensitivity(df),
        analyze_by_dataset(df),
        statistical_comparison(df),
        generate_recommendations(df),
    ]

    full_report = "\n\n".join(sections)

    output_file = RESULTS_DIR / 'BATCH_SIZE_SENSITIVITY_ANALYSIS.md'
    with open(output_file, 'w') as f:
        f.write(full_report)

    print(f"\n✓ Analysis complete!")
    print(f"  Report saved to: {output_file}")
    print(f"\n{generate_recommendations(df)}")


if __name__ == '__main__':
    main()
