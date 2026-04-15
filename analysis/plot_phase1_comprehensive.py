#!/usr/bin/env python3
"""
Comprehensive Phase 1 Analysis Plots

Creates detailed comparison plots across all Phase 1 datasets:
- vpu_nomixup
- vpu_nomixup_mean_prior (auto)
- vpu_nomixup_mean_prior (0.5)
- distpu and nnpu (baselines)
- mixup counterparts

Metrics analyzed:
- Calibration: ECE, MCE, Brier, ANICE, SNICE
- Threshold-invariant: AUC, AP, max_f1, oracle_ce
- Other: accuracy, f1, precision, recall
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import pandas as pd

# Configuration
RESULTS_DIR = Path("results_phase1_extended")
OUTPUT_DIR = Path("analysis/plots/phase1_extended")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Phase 1 datasets only
PHASE1_DATASETS = ["20News", "Connect4", "FashionMNIST", "IMDB", "MNIST", "Mushrooms", "Spambase"]

# Method groups
NOMIXUP_METHODS = ["vpu_nomixup", "vpu_nomixup_mean_prior", "nnpu", "distpu"]
MIXUP_METHODS = ["vpu", "vpu_mean_prior", "nnpu", "distpu"]

# Metric groups
CALIBRATION_METRICS = {
    "ece": "ECE ↓",
    "mce": "MCE ↓",
    "brier": "Brier ↓",
    "anice": "ANICE ↓",
    "snice": "SNICE ↓"
}

THRESHOLD_INVARIANT_METRICS = {
    "auc": "AUC ↑",
    "ap": "AP ↑",
    "max_f1": "Max F1 ↑",
    "oracle_ce": "Oracle CE ↓"
}

OTHER_METRICS = {
    "accuracy": "Accuracy ↑",
    "f1": "F1 ↑",
    "precision": "Precision ↑",
    "recall": "Recall ↑"
}

# Color scheme
COLORS = {
    "vpu_nomixup": "#1f77b4",
    "vpu_nomixup_mean_prior_auto": "#ff7f0e",
    "vpu_nomixup_mean_prior_0.5": "#2ca02c",
    "vpu": "#9467bd",
    "vpu_mean_prior_auto": "#8c564b",
    "vpu_mean_prior_0.5": "#e377c2",
    "nnpu": "#7f7f7f",
    "distpu": "#bcbd22"
}

LABELS = {
    "vpu_nomixup": "VPU (no mixup)",
    "vpu_nomixup_mean_prior_auto": "VPU-Mean-Prior (auto, no mixup)",
    "vpu_nomixup_mean_prior_0.5": "VPU-Mean-Prior (0.5, no mixup)",
    "vpu": "VPU (mixup)",
    "vpu_mean_prior_auto": "VPU-Mean-Prior (auto, mixup)",
    "vpu_mean_prior_0.5": "VPU-Mean-Prior (0.5, mixup)",
    "nnpu": "nnPU",
    "distpu": "Dist-PU"
}


def load_phase1_results():
    """Load all Phase 1 experiment results."""
    results = defaultdict(lambda: defaultdict(dict))

    for json_file in RESULTS_DIR.glob("seed_*/*.json"):
        # Skip Phase 2 datasets
        if any(ds in json_file.name for ds in ["CIFAR10", "AlzheimerMRI"]):
            continue

        # Extract dataset name
        dataset = None
        for ds in PHASE1_DATASETS:
            if json_file.name.startswith(ds):
                dataset = ds
                break

        if dataset is None:
            continue

        # Load JSON
        with open(json_file) as f:
            data = json.load(f)

        experiment_name = data["experiment"]

        # Process each method
        for method_key, method_data in data["runs"].items():
            if "best" not in method_data or "metrics" not in method_data["best"]:
                continue

            metrics = method_data["best"]["metrics"]

            # Determine method identifier
            # For vpu_nomixup_mean_prior and vpu_mean_prior, need to check method_prior
            if method_key in ["vpu_nomixup_mean_prior", "vpu_mean_prior"]:
                # Check if this is auto (method_prior not in filename) or 0.5/1.0
                if "methodprior0.5" in json_file.name:
                    method_id = f"{method_key}_0.5"
                elif "methodprior1" in json_file.name:
                    method_id = f"{method_key}_1.0"
                else:
                    method_id = f"{method_key}_auto"
            else:
                method_id = method_key

            # Store test metrics
            result_key = f"{dataset}_{experiment_name}"
            results[result_key][method_id] = {
                k.replace("test_", ""): v
                for k, v in metrics.items()
                if k.startswith("test_")
            }
            results[result_key][method_id]["dataset"] = dataset
            results[result_key][method_id]["experiment"] = experiment_name

    return results


def aggregate_by_dataset_and_prior(results):
    """Aggregate results by dataset and true prior."""
    aggregated = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for exp_key, methods in results.items():
        # Extract dataset and true prior from experiment name
        for method_id, metrics in methods.items():
            dataset = metrics["dataset"]
            exp_name = metrics["experiment"]

            # Extract c value (labeled ratio) as the grouping variable
            # Experiment names are like: MNIST_case-control_random_c0.5_seed42
            import re
            match = re.search(r'_c([\d.]+)_', exp_name)
            if match:
                true_prior = float(match.group(1))
            else:
                continue

            # Store entire metrics dict (excluding metadata)
            metrics_copy = {k: v for k, v in metrics.items() if k not in ["dataset", "experiment"]}
            aggregated[dataset][true_prior][method_id].append(metrics_copy)

    # Compute mean and std
    summary = {}
    for dataset in aggregated:
        summary[dataset] = {}
        for prior in aggregated[dataset]:
            summary[dataset][prior] = {}
            for method in aggregated[dataset][prior]:
                values = aggregated[dataset][prior][method]
                if not values:
                    continue
                # Get all metric names from first entry
                metric_names = values[0].keys()
                summary[dataset][prior][method] = {
                    "mean": {k: np.mean([v[k] for v in values if k in v]) for k in metric_names},
                    "std": {k: np.std([v[k] for v in values if k in v]) for k in metric_names},
                    "count": len(values)
                }

    return summary


def plot_metric_comparison(summary, metric_name, metric_label, methods_to_plot, output_name):
    """Create bar plot comparing methods across datasets for a single metric."""
    datasets = sorted(summary.keys())

    # Filter methods_to_plot to only those that actually exist
    available_methods = set()
    for dataset in summary:
        for prior in summary[dataset]:
            available_methods.update(summary[dataset][prior].keys())
    methods_to_plot = [m for m in methods_to_plot if m in available_methods]

    if not methods_to_plot:
        print(f"WARNING: No methods available for {output_name}. Skipping.")
        return

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]

        # Aggregate across all priors for this dataset
        method_means = defaultdict(list)

        for prior in summary[dataset]:
            for method in methods_to_plot:
                if method in summary[dataset][prior]:
                    method_means[method].append(
                        summary[dataset][prior][method]["mean"].get(metric_name, np.nan)
                    )

        # Compute overall mean across priors
        plot_data = []
        plot_labels = []
        plot_colors = []

        for method in methods_to_plot:
            if method in method_means and method_means[method]:
                plot_data.append(np.nanmean(method_means[method]))
                plot_labels.append(LABELS.get(method, method))
                plot_colors.append(COLORS.get(method, "#cccccc"))

        if plot_data:
            bars = ax.bar(range(len(plot_data)), plot_data, color=plot_colors, alpha=0.7, edgecolor='black')
            ax.set_xticks(range(len(plot_labels)))
            ax.set_xticklabels(plot_labels, rotation=45, ha='right', fontsize=8)
            ax.set_title(dataset, fontsize=10, fontweight='bold')
            ax.set_ylabel(metric_label, fontsize=9)
            ax.grid(axis='y', alpha=0.3)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=7)

    # Remove extra subplot
    if len(datasets) < len(axes):
        fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / output_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_name}")


def plot_all_metrics_heatmap(summary, methods_to_plot, metric_dict, output_name, title):
    """Create heatmap showing all metrics across datasets for selected methods."""
    datasets = sorted(summary.keys())
    metrics = list(metric_dict.keys())

    # Filter methods_to_plot to only those that actually exist in summary
    # Check which methods have data across any dataset/prior
    available_methods = set()
    for dataset in summary:
        for prior in summary[dataset]:
            available_methods.update(summary[dataset][prior].keys())

    methods_to_plot_filtered = [m for m in methods_to_plot if m in available_methods]

    if not methods_to_plot_filtered:
        print(f"WARNING: No methods from {methods_to_plot} found in results. Skipping {output_name}")
        return

    print(f"Plotting {len(methods_to_plot_filtered)}/{len(methods_to_plot)} methods for {output_name}")

    # Create data matrix: rows = datasets, cols = methods x metrics
    method_metric_combos = [(m, met) for m in methods_to_plot_filtered for met in metrics]
    data_matrix = np.zeros((len(datasets), len(method_metric_combos)))

    for i, dataset in enumerate(datasets):
        # Aggregate across all priors
        method_metric_means = defaultdict(list)

        for prior in summary[dataset]:
            for method in methods_to_plot_filtered:
                if method in summary[dataset][prior]:
                    for metric in metrics:
                        val = summary[dataset][prior][method]["mean"].get(metric, np.nan)
                        method_metric_means[(method, metric)].append(val)

        for j, (method, metric) in enumerate(method_metric_combos):
            if (method, metric) in method_metric_means:
                data_matrix[i, j] = np.nanmean(method_metric_means[(method, metric)])
            else:
                data_matrix[i, j] = np.nan

    # Create heatmap
    fig, ax = plt.subplots(figsize=(len(method_metric_combos) * 0.8, len(datasets) * 0.6))

    # Normalize each metric column separately for better visualization
    normalized_data = data_matrix.copy()
    for j in range(len(method_metric_combos)):
        col = data_matrix[:, j]
        if not np.all(np.isnan(col)):
            col_min, col_max = np.nanmin(col), np.nanmax(col)
            if col_max > col_min:
                normalized_data[:, j] = (col - col_min) / (col_max - col_min)

    sns.heatmap(normalized_data, annot=data_matrix, fmt='.3f', cmap='RdYlGn',
                xticklabels=[f"{LABELS.get(m, m)[:15]}\n{metric_dict[met]}"
                            for m, met in method_metric_combos],
                yticklabels=datasets, ax=ax, cbar_kws={'label': 'Normalized Value'},
                vmin=0, vmax=1)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / output_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_name}")


def plot_mixup_vs_nomixup_comparison(summary, metric_name, metric_label, output_name):
    """Compare mixup vs no-mixup for vpu and vpu_mean_prior methods."""
    datasets = sorted(summary.keys())

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    comparisons = [
        ("vpu_nomixup", "vpu", "VPU"),
        ("vpu_nomixup_mean_prior_auto", "vpu_mean_prior_auto", "VPU-Mean-Prior (auto)"),
        ("vpu_nomixup_mean_prior_0.5", "vpu_mean_prior_0.5", "VPU-Mean-Prior (0.5)")
    ]

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]

        # Aggregate across all priors
        comparison_data = []
        comparison_labels = []
        comparison_colors = []

        for nomixup_method, mixup_method, label in comparisons:
            nomixup_vals = []
            mixup_vals = []

            for prior in summary[dataset]:
                if nomixup_method in summary[dataset][prior]:
                    nomixup_vals.append(
                        summary[dataset][prior][nomixup_method]["mean"].get(metric_name, np.nan)
                    )
                if mixup_method in summary[dataset][prior]:
                    mixup_vals.append(
                        summary[dataset][prior][mixup_method]["mean"].get(metric_name, np.nan)
                    )

            if nomixup_vals and mixup_vals:
                nomixup_mean = np.nanmean(nomixup_vals)
                mixup_mean = np.nanmean(mixup_vals)

                comparison_data.extend([nomixup_mean, mixup_mean])
                comparison_labels.extend([f"{label}\n(no mixup)", f"{label}\n(mixup)"])
                comparison_colors.extend([COLORS.get(nomixup_method, "#aaa"),
                                        COLORS.get(mixup_method, "#888")])

        if comparison_data:
            bars = ax.bar(range(len(comparison_data)), comparison_data,
                         color=comparison_colors, alpha=0.7, edgecolor='black')
            ax.set_xticks(range(len(comparison_labels)))
            ax.set_xticklabels(comparison_labels, rotation=45, ha='right', fontsize=7)
            ax.set_title(dataset, fontsize=10, fontweight='bold')
            ax.set_ylabel(metric_label, fontsize=9)
            ax.grid(axis='y', alpha=0.3)

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=6)

    # Remove extra subplot
    if len(datasets) < len(axes):
        fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / output_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_name}")


def main():
    print("Loading Phase 1 results...")
    results = load_phase1_results()
    print(f"Loaded {len(results)} experiments")

    print("Aggregating by dataset and prior...")
    summary = aggregate_by_dataset_and_prior(results)

    # Debug: Show what methods are actually in the results
    all_methods = set()
    for dataset in summary:
        for prior in summary[dataset]:
            all_methods.update(summary[dataset][prior].keys())
    print(f"\nMethods found in results ({len(all_methods)}):")
    for method in sorted(all_methods):
        print(f"  - {method}")
    print()

    print("Creating plots...")

    # Define method groups for different comparisons
    nomixup_primary = ["vpu_nomixup", "vpu_nomixup_mean_prior_auto",
                       "vpu_nomixup_mean_prior_0.5", "nnpu", "distpu"]

    mixup_primary = ["vpu", "vpu_mean_prior_auto",
                     "vpu_mean_prior_0.5", "nnpu", "distpu"]

    # 1. Calibration metrics - no mixup
    print("\n1. Calibration metrics (no mixup)...")
    for metric, label in CALIBRATION_METRICS.items():
        plot_metric_comparison(summary, metric, label, nomixup_primary,
                             f"calibration_{metric}_nomixup.png")

    plot_all_metrics_heatmap(summary, nomixup_primary, CALIBRATION_METRICS,
                            "calibration_heatmap_nomixup.png",
                            "Calibration Metrics - No Mixup Methods")

    # 2. Threshold-invariant metrics - no mixup
    print("\n2. Threshold-invariant metrics (no mixup)...")
    for metric, label in THRESHOLD_INVARIANT_METRICS.items():
        plot_metric_comparison(summary, metric, label, nomixup_primary,
                             f"threshold_invariant_{metric}_nomixup.png")

    plot_all_metrics_heatmap(summary, nomixup_primary, THRESHOLD_INVARIANT_METRICS,
                            "threshold_invariant_heatmap_nomixup.png",
                            "Threshold-Invariant Metrics - No Mixup Methods")

    # 3. Other metrics - no mixup
    print("\n3. Other metrics (no mixup)...")
    for metric, label in OTHER_METRICS.items():
        plot_metric_comparison(summary, metric, label, nomixup_primary,
                             f"other_{metric}_nomixup.png")

    plot_all_metrics_heatmap(summary, nomixup_primary, OTHER_METRICS,
                            "other_heatmap_nomixup.png",
                            "Classification Metrics - No Mixup Methods")

    # 4. Mixup vs no mixup comparison
    print("\n4. Mixup vs no-mixup comparisons...")
    all_metrics = {**CALIBRATION_METRICS, **THRESHOLD_INVARIANT_METRICS, **OTHER_METRICS}

    for metric, label in all_metrics.items():
        plot_mixup_vs_nomixup_comparison(summary, metric, label,
                                        f"mixup_vs_nomixup_{metric}.png")

    # 5. Calibration metrics - with mixup
    print("\n5. Calibration metrics (with mixup)...")
    plot_all_metrics_heatmap(summary, mixup_primary, CALIBRATION_METRICS,
                            "calibration_heatmap_mixup.png",
                            "Calibration Metrics - Mixup Methods")

    # 6. Threshold-invariant metrics - with mixup
    print("\n6. Threshold-invariant metrics (with mixup)...")
    plot_all_metrics_heatmap(summary, mixup_primary, THRESHOLD_INVARIANT_METRICS,
                            "threshold_invariant_heatmap_mixup.png",
                            "Threshold-Invariant Metrics - Mixup Methods")

    # 7. Other metrics - with mixup
    print("\n7. Other metrics (with mixup)...")
    plot_all_metrics_heatmap(summary, mixup_primary, OTHER_METRICS,
                            "other_heatmap_mixup.png",
                            "Classification Metrics - Mixup Methods")

    print(f"\nAll plots saved to {OUTPUT_DIR}/")
    print("\nPlot categories created:")
    print("  - Calibration metrics (ECE, MCE, Brier, ANICE, SNICE)")
    print("  - Threshold-invariant metrics (AUC, AP, max_f1, oracle_ce)")
    print("  - Other metrics (accuracy, f1, precision, recall)")
    print("  - Mixup vs no-mixup comparisons")
    print("\nMethod comparisons:")
    print("  - No mixup: vpu_nomixup, vpu_nomixup_mean_prior (auto/0.5), nnpu, distpu")
    print("  - With mixup: vpu, vpu_mean_prior (auto/0.5), nnpu, distpu")


if __name__ == "__main__":
    main()
