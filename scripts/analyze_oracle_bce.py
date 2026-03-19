#!/usr/bin/env python3
"""
Analyze Oracle BCE performance as upper bound baseline.

Oracle BCE is a fully supervised method with access to both positive
and negative labels, serving as the performance ceiling for PU methods.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
SEEDS = [42, 123, 456, 789, 2024]
DATASETS = ['MNIST', 'FashionMNIST', 'IMDB', '20News', 'Spambase', 'Mushrooms']
RESULTS_DIR = Path('results')

METRICS = ['test_f1', 'test_recall', 'test_precision', 'test_auc',
           'test_accuracy', 'test_anice', 'test_snice', 'test_ece',
           'test_mce', 'test_brier']


def load_oracle_results() -> pd.DataFrame:
    """Load all Oracle BCE results."""
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

            # Extract experiment details
            parts = experiment.split('_')
            dataset = parts[0] if parts else 'unknown'

            # Skip non-main datasets
            if dataset not in DATASETS:
                continue

            # Extract c and prior
            c_value = None
            prior_value = None
            for part in parts:
                if part.startswith('c') and not part.startswith('case'):
                    try:
                        c_value = float(part[1:])
                    except:
                        pass
                if part.startswith('prior'):
                    try:
                        prior_value = float(part[5:])
                    except:
                        pass

            # Get oracle_bce metrics
            oracle_data = data.get('runs', {}).get('oracle_bce')
            if not oracle_data:
                continue

            if 'best' not in oracle_data or 'metrics' not in oracle_data['best']:
                continue

            metrics = oracle_data['best']['metrics']

            # Get convergence info
            best_epoch = oracle_data['best'].get('epoch', None)
            global_epochs = oracle_data.get('global_epochs', None)

            record = {
                'experiment': experiment,
                'dataset': dataset,
                'seed': seed,
                'c': c_value,
                'prior': prior_value,
                'best_epoch': best_epoch,
                'global_epochs': global_epochs,
            }

            for metric in METRICS:
                record[metric] = metrics.get(metric)

            records.append(record)

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} Oracle BCE results")
    print(f"Datasets: {sorted(df['dataset'].unique())}")
    print(f"Seeds: {sorted(df['seed'].unique())}")

    return df


def analyze_overall(df: pd.DataFrame) -> str:
    """Overall Oracle BCE performance."""
    output = []
    output.append("=" * 80)
    output.append("ORACLE BCE: FULLY SUPERVISED UPPER BOUND")
    output.append("=" * 80)
    output.append("")
    output.append("Oracle BCE has access to BOTH positive and negative labels.")
    output.append("It represents the performance ceiling for PU learning methods.")
    output.append("")

    output.append("Overall Performance (mean ± std)")
    output.append("-" * 80)

    for metric in METRICS:
        mean = df[metric].mean()
        std = df[metric].std()
        metric_name = metric.replace('test_', '').upper()
        output.append(f"{metric_name:<15} {mean:>7.4f} ± {std:>6.4f}")

    output.append("")
    output.append(f"Based on {len(df)} experiments across {len(df['dataset'].unique())} datasets")

    return "\n".join(output)


def analyze_by_dataset(df: pd.DataFrame) -> str:
    """Performance by dataset."""
    output = []
    output.append("\n" + "=" * 80)
    output.append("ORACLE BCE BY DATASET")
    output.append("=" * 80)

    for dataset in DATASETS:
        df_dataset = df[df['dataset'] == dataset]

        if len(df_dataset) == 0:
            continue

        output.append(f"\n{dataset}")
        output.append("-" * 80)

        # Key metrics only
        f1 = df_dataset['test_f1'].mean()
        auc = df_dataset['test_auc'].mean()
        prec = df_dataset['test_precision'].mean()
        rec = df_dataset['test_recall'].mean()
        acc = df_dataset['test_accuracy'].mean()

        # Calibration
        anice = df_dataset['test_anice'].mean()
        ece = df_dataset['test_ece'].mean()

        output.append(f"F1:         {f1:.4f}")
        output.append(f"AUC:        {auc:.4f}")
        output.append(f"Precision:  {prec:.4f}")
        output.append(f"Recall:     {rec:.4f}")
        output.append(f"Accuracy:   {acc:.4f}")
        output.append(f"A-NICE:     {anice:.4f}")
        output.append(f"ECE:        {ece:.4f}")
        output.append(f"Experiments: {len(df_dataset)}")

    return "\n".join(output)


def analyze_convergence(df: pd.DataFrame) -> str:
    """Analyze Oracle BCE convergence speed."""
    output = []
    output.append("\n" + "=" * 80)
    output.append("ORACLE BCE CONVERGENCE SPEED")
    output.append("=" * 80)
    output.append("")

    # Overall convergence
    epochs_mean = df['best_epoch'].mean()
    epochs_std = df['best_epoch'].std()

    output.append(f"Average epochs to best: {epochs_mean:.1f} ± {epochs_std:.1f}")
    output.append("")

    # By dataset
    output.append("Convergence by Dataset")
    output.append("-" * 80)
    output.append(f"{'Dataset':<15} {'Avg Epochs':<12} {'Std':<10} {'Min':<8} {'Max':<8}")
    output.append("-" * 80)

    for dataset in DATASETS:
        df_dataset = df[df['dataset'] == dataset]
        if len(df_dataset) == 0:
            continue

        epochs_mean = df_dataset['best_epoch'].mean()
        epochs_std = df_dataset['best_epoch'].std()
        epochs_min = df_dataset['best_epoch'].min()
        epochs_max = df_dataset['best_epoch'].max()

        output.append(f"{dataset:<15} {epochs_mean:>8.1f}     {epochs_std:>6.1f}   {epochs_min:>5.0f}    {epochs_max:>5.0f}")

    return "\n".join(output)


def compare_to_best_pu(df_oracle: pd.DataFrame) -> str:
    """Compare Oracle to best PU methods."""
    output = []
    output.append("\n" + "=" * 80)
    output.append("GAP BETWEEN ORACLE BCE AND BEST PU METHODS")
    output.append("=" * 80)
    output.append("")

    # Load VPU-Mean and VPU-NoMixUp-Mean results
    all_records = []

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

            # Get VPU-Mean and VPU-NoMixUp-Mean
            for method_name in ['vpu_mean', 'vpu_nomixup_mean']:
                method_data = data.get('runs', {}).get(method_name)
                if not method_data:
                    continue

                if 'best' not in method_data or 'metrics' not in method_data['best']:
                    continue

                metrics = method_data['best']['metrics']

                record = {
                    'experiment': experiment,
                    'dataset': dataset,
                    'seed': seed,
                    'method': method_name,
                    'test_f1': metrics.get('test_f1'),
                    'test_auc': metrics.get('test_auc'),
                }

                all_records.append(record)

    df_pu = pd.DataFrame(all_records)

    if len(df_pu) == 0:
        output.append("No PU method results found for comparison")
        return "\n".join(output)

    # Merge oracle and PU on experiment
    df_oracle_subset = df_oracle[['experiment', 'dataset', 'test_f1', 'test_auc']].copy()
    df_oracle_subset = df_oracle_subset.rename(columns={'test_f1': 'oracle_f1', 'test_auc': 'oracle_auc'})

    df_merged = df_pu.merge(df_oracle_subset, on='experiment', how='inner', suffixes=('', '_oracle'))

    # Calculate gaps
    df_merged['f1_gap'] = df_merged['oracle_f1'] - df_merged['test_f1']
    df_merged['auc_gap'] = df_merged['oracle_auc'] - df_merged['test_auc']

    output.append("Average F1 Gap (Oracle - PU Method)")
    output.append("-" * 80)
    output.append(f"{'Method':<25} {'Gap':<10} {'Oracle F1':<12} {'PU F1':<12}")
    output.append("-" * 80)

    for method in ['vpu_mean', 'vpu_nomixup_mean']:
        df_method = df_merged[df_merged['method'] == method]
        if len(df_method) == 0:
            continue

        gap = df_method['f1_gap'].mean()
        oracle_f1 = df_method['oracle_f1'].mean()
        pu_f1 = df_method['test_f1'].mean()

        output.append(f"{method:<25} {gap:>6.4f}     {oracle_f1:>8.4f}     {pu_f1:>8.4f}")

    output.append("")
    output.append("Average AUC Gap (Oracle - PU Method)")
    output.append("-" * 80)
    output.append(f"{'Method':<25} {'Gap':<10} {'Oracle AUC':<12} {'PU AUC':<12}")
    output.append("-" * 80)

    for method in ['vpu_mean', 'vpu_nomixup_mean']:
        df_method = df_merged[df_merged['method'] == method]
        if len(df_method) == 0:
            continue

        gap = df_method['auc_gap'].mean()
        oracle_auc = df_method['oracle_auc'].mean()
        pu_auc = df_method['test_auc'].mean()

        output.append(f"{method:<25} {gap:>6.4f}     {oracle_auc:>8.4f}     {pu_auc:>8.4f}")

    output.append("")
    output.append("Gap by Dataset (VPU-Mean)")
    output.append("-" * 80)
    output.append(f"{'Dataset':<15} {'F1 Gap':<10} {'Oracle F1':<12} {'VPU-Mean F1':<12}")
    output.append("-" * 80)

    for dataset in DATASETS:
        df_dataset = df_merged[(df_merged['method'] == 'vpu_mean') &
                               (df_merged['dataset'] == dataset)]
        if len(df_dataset) == 0:
            continue

        gap = df_dataset['f1_gap'].mean()
        oracle_f1 = df_dataset['oracle_f1'].mean()
        pu_f1 = df_dataset['test_f1'].mean()

        output.append(f"{dataset:<15} {gap:>6.4f}     {oracle_f1:>8.4f}     {pu_f1:>8.4f}")

    # Convergence speed comparison
    output.append("")
    output.append("")
    output.append("Convergence Speed: Oracle vs VPU Methods")
    output.append("-" * 80)

    # Load VPU convergence data
    vpu_records = []
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

            for method_name in ['vpu', 'vpu_mean', 'vpu_nomixup', 'vpu_nomixup_mean']:
                method_data = data.get('runs', {}).get(method_name)
                if not method_data or 'best' not in method_data:
                    continue

                best_epoch = method_data['best'].get('epoch', None)

                vpu_records.append({
                    'experiment': experiment,
                    'dataset': dataset,
                    'method': method_name,
                    'best_epoch': best_epoch,
                })

    if len(vpu_records) > 0:
        df_vpu_conv = pd.DataFrame(vpu_records)

        # Merge oracle convergence data
        df_oracle_conv = df_oracle[['experiment', 'best_epoch']].copy()
        df_oracle_conv = df_oracle_conv.rename(columns={'best_epoch': 'oracle_epochs'})

        df_conv_merged = df_vpu_conv.merge(df_oracle_conv, on='experiment', how='inner')

        output.append(f"{'Method':<25} {'Avg Epochs':<12} {'Oracle Epochs':<15} {'Difference':<12}")
        output.append("-" * 80)

        oracle_avg = df_oracle['best_epoch'].mean()

        for method in ['vpu', 'vpu_mean', 'vpu_nomixup', 'vpu_nomixup_mean']:
            df_method = df_conv_merged[df_conv_merged['method'] == method]
            if len(df_method) == 0:
                continue

            method_avg = df_method['best_epoch'].mean()
            diff = method_avg - oracle_avg

            output.append(f"{method:<25} {method_avg:>8.1f}     {oracle_avg:>11.1f}     {diff:+8.1f}")

        output.append("")
        output.append(f"Oracle converges in {oracle_avg:.1f} epochs on average")

    return "\n".join(output)


def main():
    """Run Oracle BCE analysis."""
    print("Loading Oracle BCE results...")
    df = load_oracle_results()

    if len(df) == 0:
        print("No Oracle BCE results found!")
        return

    print("\nGenerating analysis...")

    sections = [
        analyze_overall(df),
        analyze_by_dataset(df),
        analyze_convergence(df),
        compare_to_best_pu(df),
    ]

    full_report = "\n\n".join(sections)

    output_file = RESULTS_DIR / 'ORACLE_BCE_ANALYSIS.md'
    with open(output_file, 'w') as f:
        f.write(full_report)

    print(f"\n✓ Analysis complete!")
    print(f"  Report saved to: {output_file}")
    print(f"\n{analyze_overall(df)}")


if __name__ == '__main__':
    main()
