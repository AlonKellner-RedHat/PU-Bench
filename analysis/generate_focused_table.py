#!/usr/bin/env python3
"""
Generate Focused Stratified Table

Focus on core method comparison:
- distpu, nnpu (baselines)
- vpu, vpu-nomixup (standard VPU)
- vpu-nomixup-mean-prior(0.5) (best variant)
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import re

# Configuration
RESULTS_DIR = Path("results_comprehensive")
OUTPUT_FILE = Path("analysis/PHASE1_FOCUSED_TABLE.md")

# Phase 1 datasets grouped by modality
DATASET_MODALITIES = {
    "Text": ["20News", "IMDB"],
    "Tabular": ["Mushrooms", "Spambase", "Connect4"],
    "Vision": ["MNIST", "FashionMNIST"]
}

# Key metrics
PAPER_METRICS = {
    "auc": {"label": "AUC", "higher_better": True, "format": ".3f"},
    "ap": {"label": "AP", "higher_better": True, "format": ".3f"},
    "max_f1": {"label": "Max F1", "higher_better": True, "format": ".3f"},
    "ece": {"label": "ECE", "higher_better": False, "format": ".3f"},
    "brier": {"label": "Brier", "higher_better": False, "format": ".3f"},
    "anice": {"label": "ANICE", "higher_better": False, "format": ".3f"},
    "oracle_ce": {"label": "Oracle CE", "higher_better": False, "format": ".3f"},
    "convergence": {"label": "Epochs", "higher_better": False, "format": ".1f"},
}

# Focused method set (in display order)
FOCUSED_METHODS = [
    ("nnpu", "nnPU"),
    ("distpu", "Dist-PU"),
    ("vpu", "VPU"),
    ("vpu_nomixup", "VPU-nomix"),
    ("vpu_nomixup_mean_prior_0.5", "VPU-nomix-MP(0.5)"),
]


def load_results_by_dataset():
    """Load results organized by dataset and method."""
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for json_file in RESULTS_DIR.glob("seed_*/*.json"):
        # Skip Phase 2 datasets
        if any(ds in json_file.name for ds in ["CIFAR10", "AlzheimerMRI"]):
            continue

        # Extract dataset name
        dataset = None
        for modality, datasets in DATASET_MODALITIES.items():
            for ds in datasets:
                if json_file.name.startswith(ds):
                    dataset = ds
                    break
            if dataset:
                break

        if dataset is None:
            continue

        with open(json_file) as f:
            data = json.load(f)

        # Process each method
        for method_key, method_data in data["runs"].items():
            if "best" not in method_data or "metrics" not in method_data["best"]:
                continue

            metrics = method_data["best"]["metrics"]

            # Determine method identifier
            if method_key in ["vpu_nomixup_mean_prior", "vpu_mean_prior"]:
                if "methodprior0.5" in json_file.name:
                    method_id = f"{method_key}_0.5"
                elif "methodprior1" in json_file.name:
                    continue
                else:
                    method_id = f"{method_key}_auto"
            else:
                method_id = method_key

            # Skip methods not in our focused set
            focused_ids = [m[0] for m in FOCUSED_METHODS]
            if method_id not in focused_ids:
                continue

            # Store test metrics for this dataset
            for metric in PAPER_METRICS.keys():
                test_metric = f"test_{metric}"
                if test_metric in metrics:
                    results[dataset][method_id][metric].append(metrics[test_metric])

            # Store convergence speed (best epoch)
            if "epoch" in method_data["best"]:
                if "convergence" not in results[dataset][method_id]:
                    results[dataset][method_id]["convergence"] = []
                results[dataset][method_id]["convergence"].append(method_data["best"]["epoch"])

    return results


def compute_dataset_statistics(results):
    """Compute mean and std per dataset (across seeds only)."""
    stats = {}

    for dataset in results:
        stats[dataset] = {}
        for method in results[dataset]:
            stats[dataset][method] = {}
            for metric in results[dataset][method]:
                values = np.array(results[dataset][method][metric])
                stats[dataset][method][metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values, ddof=1),  # Sample std across seeds
                    "n": len(values)
                }

    return stats


def compute_modality_statistics(results):
    """Compute mean and std per modality."""
    modality_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for modality, datasets in DATASET_MODALITIES.items():
        for dataset in datasets:
            if dataset in results:
                for method in results[dataset]:
                    for metric in results[dataset][method]:
                        modality_results[modality][method][metric].extend(
                            results[dataset][method][metric]
                        )

    # Compute statistics
    stats = {}
    for modality in modality_results:
        stats[modality] = {}
        for method in modality_results[modality]:
            stats[modality][method] = {}
            for metric in modality_results[modality][method]:
                values = np.array(modality_results[modality][method][metric])
                stats[modality][method][metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values, ddof=1),
                    "n": len(values)
                }

    return stats


def rank_within_dataset(dataset_stats, metric, higher_better):
    """Rank methods within a single dataset for a metric."""
    rankings = []
    for method in dataset_stats:
        if metric in dataset_stats[method]:
            rankings.append((method, dataset_stats[method][metric]["mean"]))

    rankings.sort(key=lambda x: x[1], reverse=higher_better)

    rank_dict = {}
    for rank, (method, _) in enumerate(rankings, 1):
        rank_dict[method] = rank

    return rank_dict


def generate_per_dataset_table(dataset_stats):
    """Generate table with one row per dataset."""
    lines = []

    lines.append("**Table 1: Phase 1 Results by Dataset**")
    lines.append("")
    lines.append("*Mean ± Std across 15 runs per dataset (3 label frequencies × 5 seeds). "
                 "**Bold** = best, *italic* = second-best per dataset+metric. "
                 "Epochs = convergence speed (fewer is faster).*")
    lines.append("")

    # Build header
    header = "| Dataset | Method |"
    separator = "|---------|--------|"

    for metric_key, metric_info in PAPER_METRICS.items():
        direction = "↑" if metric_info["higher_better"] else "↓"
        header += f" {metric_info['label']} {direction} |"
        separator += "------:|"

    lines.append(header)
    lines.append(separator)

    # Process by modality, then dataset
    for modality, datasets in DATASET_MODALITIES.items():
        lines.append(f"| **{modality}** | | | | | |")

        for dataset in sorted(datasets):
            if dataset not in dataset_stats:
                continue

            # Compute rankings for this dataset
            rankings = {}
            for metric_key, metric_info in PAPER_METRICS.items():
                rankings[metric_key] = rank_within_dataset(
                    dataset_stats[dataset], metric_key, metric_info["higher_better"]
                )

            # Create row for each method
            first_method = True
            for method_id, method_name in FOCUSED_METHODS:
                if method_id not in dataset_stats[dataset]:
                    continue

                row = f"| {dataset if first_method else ''} | {method_name} |"
                first_method = False

                for metric_key, metric_info in PAPER_METRICS.items():
                    if metric_key not in dataset_stats[dataset][method_id]:
                        row += " — |"
                        continue

                    mean = dataset_stats[dataset][method_id][metric_key]["mean"]
                    std = dataset_stats[dataset][method_id][metric_key]["std"]

                    rank = rankings[metric_key].get(method_id, 999)
                    is_best = (rank == 1)
                    is_second = (rank == 2)

                    formatted = f"{mean:{metric_info['format']}}"
                    std_formatted = f"{std:{metric_info['format']}}"

                    if is_best:
                        cell = f"**{formatted} ± {std_formatted}**"
                    elif is_second:
                        cell = f"*{formatted} ± {std_formatted}*"
                    else:
                        cell = f"{formatted} ± {std_formatted}"

                    row += f" {cell} |"

                lines.append(row)

            lines.append("|")  # Separator between datasets

    return "\n".join(lines)


def generate_modality_summary_table(modality_stats):
    """Generate summary table aggregated by modality."""
    lines = []

    lines.append("**Table 2: Results by Modality**")
    lines.append("")
    lines.append("*Aggregates across all datasets within each modality. "
                 "Error bars include both seed and dataset variance. "
                 "Epochs = average convergence speed.*")
    lines.append("")

    # Build header
    header = "| Modality | Method |"
    separator = "|----------|--------|"

    for metric_key, metric_info in PAPER_METRICS.items():
        direction = "↑" if metric_info["higher_better"] else "↓"
        header += f" {metric_info['label']} {direction} |"
        separator += "------:|"

    lines.append(header)
    lines.append(separator)

    for modality in ["Text", "Tabular", "Vision"]:
        if modality not in modality_stats:
            continue

        # Compute rankings for this modality
        rankings = {}
        for metric_key, metric_info in PAPER_METRICS.items():
            rankings[metric_key] = rank_within_dataset(
                modality_stats[modality], metric_key, metric_info["higher_better"]
            )

        first_method = True
        for method_id, method_name in FOCUSED_METHODS:
            if method_id not in modality_stats[modality]:
                continue

            row = f"| {modality if first_method else ''} | {method_name} |"
            first_method = False

            for metric_key, metric_info in PAPER_METRICS.items():
                if metric_key not in modality_stats[modality][method_id]:
                    row += " — |"
                    continue

                mean = modality_stats[modality][method_id][metric_key]["mean"]
                std = modality_stats[modality][method_id][metric_key]["std"]

                rank = rankings[metric_key].get(method_id, 999)
                is_best = (rank == 1)
                is_second = (rank == 2)

                formatted = f"{mean:{metric_info['format']}}"
                std_formatted = f"{std:{metric_info['format']}}"

                if is_best:
                    cell = f"**{formatted} ± {std_formatted}**"
                elif is_second:
                    cell = f"*{formatted} ± {std_formatted}*"
                else:
                    cell = f"{formatted} ± {std_formatted}"

                row += f" {cell} |"

            lines.append(row)

        lines.append("|")

    return "\n".join(lines)


def generate_variance_analysis(results, dataset_stats):
    """Detailed variance analysis showing seed vs dataset variance."""
    lines = []

    lines.append("## Variance Decomposition")
    lines.append("")
    lines.append("**Demonstrates tight error margins when stratified by dataset**")
    lines.append("")

    # Focus on best method for clear example
    method_id = "vpu_nomixup_mean_prior_0.5"
    method_name = "VPU-nomix-MP(0.5)"
    metric = "auc"

    lines.append(f"**Example: {method_name}, {PAPER_METRICS[metric]['label']}**")
    lines.append("")
    lines.append("| Dataset | Mean | Std (seed variance) | n |")
    lines.append("|---------|------|---------------------|---|")

    dataset_means = []
    dataset_stds = []

    for dataset in sorted(results.keys()):
        if method_id in dataset_stats[dataset] and metric in dataset_stats[dataset][method_id]:
            mean = dataset_stats[dataset][method_id][metric]["mean"]
            std = dataset_stats[dataset][method_id][metric]["std"]
            n = dataset_stats[dataset][method_id][metric]["n"]

            dataset_means.append(mean)
            dataset_stds.append(std)

            lines.append(f"| {dataset} | {mean:.4f} | **{std:.4f}** | {n} |")

    lines.append("")

    # Compute cross-dataset variance
    overall_mean = np.mean(dataset_means)
    overall_std = np.std(dataset_means, ddof=1)
    avg_within_std = np.mean(dataset_stds)

    lines.append("**Summary:**")
    lines.append(f"- Grand mean (across all datasets): {overall_mean:.4f}")
    lines.append(f"- Average within-dataset std (seed variance): **{avg_within_std:.4f}**")
    lines.append(f"- Cross-dataset std (dataset heterogeneity): {overall_std:.4f}")
    lines.append("")
    lines.append(f"**Variance ratio**: Dataset heterogeneity is **{overall_std / avg_within_std:.1f}× larger** than seed variance")
    lines.append("")
    lines.append("**Key insight**: By stratifying, we get **{:.1f}× tighter error margins** "
                 "(0.{:03d} vs 0.{:03d})".format(
                     overall_std / avg_within_std,
                     int(avg_within_std * 1000),
                     int(overall_std * 1000)
                 ))
    lines.append("")

    return "\n".join(lines)


def generate_method_comparison_summary(dataset_stats, modality_stats):
    """High-level comparison of methods."""
    lines = []

    lines.append("## Method Comparison Summary")
    lines.append("")

    # Count wins per method across all datasets
    method_wins = defaultdict(int)
    method_ranks = defaultdict(list)

    for dataset in dataset_stats:
        for metric_key, metric_info in PAPER_METRICS.items():
            rankings = rank_within_dataset(
                dataset_stats[dataset], metric_key, metric_info["higher_better"]
            )
            for method, rank in rankings.items():
                method_ranks[method].append(rank)
                if rank == 1:
                    method_wins[method] += 1

    num_metrics = len(PAPER_METRICS)
    total_comparisons = 7 * num_metrics

    lines.append(f"### Overall Performance (across all 7 datasets × {num_metrics} metrics)")
    lines.append("")
    lines.append("| Method | Total Wins | Avg Rank |")
    lines.append("|--------|-----------|----------|")

    for method_id, method_name in FOCUSED_METHODS:
        if method_id in method_wins:
            wins = method_wins[method_id]
            avg_rank = np.mean(method_ranks[method_id])
            lines.append(f"| {method_name} | {wins}/{total_comparisons} | {avg_rank:.2f} |")

    lines.append("")

    # Modality-specific winners
    lines.append("### Best Method by Modality")
    lines.append("")

    num_metrics = len(PAPER_METRICS)

    for modality in ["Text", "Tabular", "Vision"]:
        if modality not in modality_stats:
            continue

        modality_wins = defaultdict(int)
        for metric_key, metric_info in PAPER_METRICS.items():
            rankings = rank_within_dataset(
                modality_stats[modality], metric_key, metric_info["higher_better"]
            )
            for method, rank in rankings.items():
                if rank == 1:
                    modality_wins[method] += 1

        best_method = max(modality_wins.items(), key=lambda x: x[1])
        best_method_name = dict(FOCUSED_METHODS).get(best_method[0], best_method[0])

        lines.append(f"**{modality}**: {best_method_name} ({best_method[1]}/{num_metrics} metrics)")

    lines.append("")

    return "\n".join(lines)


def main():
    print("Loading results by dataset...")
    results = load_results_by_dataset()
    print(f"Loaded datasets: {list(results.keys())}")

    print("Computing dataset-level statistics...")
    dataset_stats = compute_dataset_statistics(results)

    print("Computing modality-level statistics...")
    modality_stats = compute_modality_statistics(results)

    print("Generating focused tables...")

    output = []
    output.append("# Phase 1 Results - Focused Method Comparison")
    output.append("")
    output.append("**Methods Compared:**")
    output.append("- **Baselines**: nnPU, Dist-PU")
    output.append("- **VPU variants**: VPU (mixup), VPU-nomix (no mixup)")
    output.append("- **Best variant**: VPU-nomix-MP(0.5) (no mixup + mean-prior with 0.5)")
    output.append("")
    output.append("---")
    output.append("")

    # Per-dataset table
    output.append(generate_per_dataset_table(dataset_stats))
    output.append("")
    output.append("---")
    output.append("")

    # Modality summary table
    output.append(generate_modality_summary_table(modality_stats))
    output.append("")
    output.append("---")
    output.append("")

    # Method comparison summary
    output.append(generate_method_comparison_summary(dataset_stats, modality_stats))
    output.append("")
    output.append("---")
    output.append("")

    # Variance analysis
    output.append(generate_variance_analysis(results, dataset_stats))
    output.append("")

    # Write to file
    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(output))

    print(f"\nFocused table saved to {OUTPUT_FILE}")
    print("\nKey features:")
    print("- 5 focused methods (baselines + VPU variants)")
    print("- ± notation for error margins")
    print("- Stratified by dataset (tight margins)")
    print("- Modality-level summary")
    print("- Variance decomposition analysis")


if __name__ == "__main__":
    main()
