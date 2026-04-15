#!/usr/bin/env python3
"""
Generate comprehensive Phase 1 Extended results table with all methods and baselines.

Similar style to PHASE1_FOCUSED_TABLE.md but with all 19 methods and timing metrics.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import re

RESULTS_DIR = Path("results_phase1_extended")
PHASE1_DATASETS = ["20News", "Connect4", "FashionMNIST", "IMDB", "MNIST", "Mushrooms", "Spambase"]

# Method display names
METHOD_LABELS = {
    "nnpu": "nnPU",
    "nnpusb": "nnPU-SB",
    "bbepu": "BBE-PU",
    "lbe": "LBE",
    "puet": "PUET",
    "distpu": "Dist-PU",
    "selfpu": "Self-PU",
    "p3mixe": "P3Mix-E",
    "p3mixc": "P3Mix-C",
    "robustpu": "Robust-PU",
    "pn_naive": "PN-Naive",
    "oracle_bce": "Oracle-PN",
    "vaepu": "VAE-PU",
    "vpu": "VPU",
    "vpu_nomixup": "VPU-nomix",
    "vpu_mean_prior_auto": "VPU-MP(auto)",
    "vpu_mean_prior_0.5": "VPU-MP(0.5)",
    "vpu_nomixup_mean_prior_auto": "VPU-nomix-MP(auto)",
    "vpu_nomixup_mean_prior_0.5": "VPU-nomix-MP(0.5)",
}

# Method ordering (by category)
# Excluding PUET (missing most metrics) and VAE-PU (incomplete runs)
METHOD_ORDER = [
    # Core PU baselines
    "nnpu", "nnpusb", "bbepu", "lbe", "distpu",
    # Recent PU methods
    "selfpu", "p3mixe", "p3mixc", "robustpu",
    # VPU variants (no mixup)
    "vpu_nomixup", "vpu_nomixup_mean_prior_auto", "vpu_nomixup_mean_prior_0.5",
    # VPU variants (with mixup)
    "vpu", "vpu_mean_prior_auto", "vpu_mean_prior_0.5",
    # Oracles
    "pn_naive", "oracle_bce",
]

# Methods to exclude from analysis (will be updated dynamically)
EXCLUDED_METHODS = {}


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
            # Skip excluded methods
            if method_key in EXCLUDED_METHODS:
                continue

            if "best" not in method_data or "metrics" not in method_data["best"]:
                continue

            metrics = method_data["best"]["metrics"]
            timing = method_data.get("timing", {})
            best_epoch = method_data["best"].get("epoch", None)

            # Determine method identifier
            if method_key in ["vpu_nomixup_mean_prior", "vpu_mean_prior"]:
                if "methodprior0.5" in json_file.name:
                    method_id = f"{method_key}_0.5"
                elif "methodprior1" in json_file.name:
                    method_id = f"{method_key}_1.0"
                else:
                    method_id = f"{method_key}_auto"
            else:
                method_id = method_key

            # Store test metrics + timing
            result_key = f"{dataset}_{experiment_name}"
            results[result_key][method_id] = {
                k.replace("test_", ""): v
                for k, v in metrics.items()
                if k.startswith("test_")
            }

            # Add timing and epoch info
            global_epochs = method_data.get("global_epochs")
            total_duration = timing.get("duration_seconds")

            results[result_key][method_id]["best_epoch"] = best_epoch
            results[result_key][method_id]["global_epochs"] = global_epochs
            results[result_key][method_id]["total_duration"] = total_duration

            # Calculate derived metrics
            if global_epochs and total_duration and global_epochs > 0:
                time_per_epoch = total_duration / global_epochs
                results[result_key][method_id]["time_per_epoch"] = time_per_epoch

                if best_epoch is not None:
                    time_to_best = time_per_epoch * best_epoch
                    results[result_key][method_id]["time_to_best"] = time_to_best
                else:
                    results[result_key][method_id]["time_to_best"] = None
            else:
                results[result_key][method_id]["time_per_epoch"] = None
                results[result_key][method_id]["time_to_best"] = None

            results[result_key][method_id]["dataset"] = dataset
            results[result_key][method_id]["experiment"] = experiment_name

    return results


def validate_methods(results):
    """Validate that methods have complete runs and required metrics."""
    # Expected runs: 7 datasets × 10 seeds × 3 c values = 210 runs per method
    # Each mean_prior variant (auto/0.5) is counted separately, so also 210 runs each

    EXPECTED_RUNS = 210

    # Required metrics for comparison
    REQUIRED_METRICS = {
        "auc", "ap", "max_f1", "ece", "brier", "anice", "oracle_ce"
    }

    method_counts = defaultdict(int)
    method_metrics = defaultdict(set)

    for exp_key, methods in results.items():
        for method_id, method_data in methods.items():
            method_counts[method_id] += 1

            # Track which metrics this method has
            for metric_key in method_data.keys():
                if metric_key not in ["dataset", "experiment", "best_epoch", "global_epochs",
                                     "total_duration", "time_per_epoch", "time_to_best"]:
                    method_metrics[method_id].add(metric_key)

    complete_methods = set()
    incomplete_methods = {}

    for method, count in method_counts.items():
        # Check run count
        if count != EXPECTED_RUNS:
            incomplete_methods[method] = f"Incomplete runs ({count}/{EXPECTED_RUNS})"
            continue

        # Check required metrics
        missing_metrics = REQUIRED_METRICS - method_metrics[method]
        if missing_metrics:
            incomplete_methods[method] = f"Missing metrics: {', '.join(sorted(missing_metrics))}"
            continue

        complete_methods.add(method)

    return complete_methods, incomplete_methods


def aggregate_by_dataset(results):
    """Aggregate results by dataset (across all c values and seeds)."""
    aggregated = defaultdict(lambda: defaultdict(list))

    for exp_key, methods in results.items():
        for method_id, metrics in methods.items():
            dataset = metrics["dataset"]

            # Store entire metrics dict (excluding metadata)
            metrics_copy = {k: v for k, v in metrics.items() if k not in ["dataset", "experiment"]}
            aggregated[dataset][method_id].append(metrics_copy)

    # Compute mean and std
    summary = {}
    for dataset in aggregated:
        summary[dataset] = {}
        for method in aggregated[dataset]:
            values = aggregated[dataset][method]
            if not values:
                continue

            # Get all metric names from first entry
            metric_names = values[0].keys()
            summary[dataset][method] = {
                "mean": {k: np.nanmean([v.get(k) for v in values if v.get(k) is not None]) for k in metric_names},
                "std": {k: np.nanstd([v.get(k) for v in values if v.get(k) is not None]) for k in metric_names},
                "count": len(values)
            }

    return summary


def format_metric(mean, std, lower_is_better=False):
    """Format metric as mean ± std."""
    if np.isnan(mean) or np.isnan(std):
        return "—"
    return f"{mean:.3f} ± {std:.3f}"


def find_best_methods(summary, datasets, methods, metric, lower_is_better=False):
    """Find best and second-best methods for each dataset."""
    best = {}
    second_best = {}

    for dataset in datasets:
        values = []
        for method in methods:
            if method in summary[dataset]:
                mean_val = summary[dataset][method]["mean"].get(metric, np.nan)
                if not np.isnan(mean_val):
                    values.append((mean_val, method))

        if len(values) >= 2:
            values.sort(reverse=not lower_is_better)
            best[dataset] = values[0][1]
            second_best[dataset] = values[1][1]
        elif len(values) == 1:
            best[dataset] = values[0][1]

    return best, second_best


def generate_table_markdown(summary):
    """Generate markdown table with all methods and datasets."""

    datasets = sorted([ds for ds in PHASE1_DATASETS if ds in summary])
    methods = [m for m in METHOD_ORDER if any(m in summary[ds] for ds in datasets)]

    # Metric definitions with categories
    METRIC_CATEGORIES = {
        "threshold_invariant": [
            ("auc", "AUC ↑", False),
            ("ap", "AP ↑", False),
            ("max_f1", "Max F1 ↑", False),
        ],
        "calibration": [
            ("ece", "ECE ↓", True),
            ("brier", "Brier ↓", True),
            ("anice", "ANICE ↓", True),
        ],
        "cross_entropy": [
            ("oracle_ce", "Oracle CE ↓", True),
        ],
        "speed": [
            ("best_epoch", "Epochs ↓", True),
            ("time_to_best", "Time to Best (s) ↓", True),
            ("time_per_epoch", "Time/Epoch (s) ↓", True),
        ],
    }

    # Flatten for table generation
    metrics = []
    for category_metrics in METRIC_CATEGORIES.values():
        metrics.extend(category_metrics)

    # Count actual methods that will be included
    num_methods = len(methods)

    lines = []
    lines.append("# Phase 1 Extended Results - Comprehensive Method Comparison")
    lines.append("")
    lines.append(f"**Methods Compared ({num_methods} total):**")
    lines.append("- **Core PU baselines** (5): nnPU, nnPU-SB, BBE-PU, LBE, Dist-PU")
    lines.append("- **Recent PU methods** (4): Self-PU, P3Mix-E, P3Mix-C, Robust-PU")
    lines.append("- **VPU variants (no mixup)** (3): VPU-nomix, VPU-nomix-MP(auto), VPU-nomix-MP(0.5)")
    lines.append("- **VPU variants (with mixup)** (3): VPU, VPU-MP(auto), VPU-MP(0.5)")
    lines.append("- **Oracle baselines** (2): PN-Naive, Oracle-PN")
    lines.append("")
    lines.append("**Excluded from comparison:**")
    for method, reason in sorted(EXCLUDED_METHODS.items()):
        lines.append(f"- **{METHOD_LABELS.get(method, method)}**: {reason}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Generate table for each dataset
    for dataset in datasets:
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append("*Mean ± Std across 30 runs (3 label frequencies × 10 seeds). **Bold** = best, *italic* = second-best per metric.*")
        lines.append("")

        # Header
        header = "| Method |"
        separator = "|--------|"
        for _, label, _ in metrics:
            header += f" {label} |"
            separator += "------:|"
        lines.append(header)
        lines.append(separator)

        # Find best/second-best for each metric
        best_methods = {}
        second_best_methods = {}
        for metric_key, _, lower_is_better in metrics:
            best, second = find_best_methods(summary, [dataset], methods, metric_key, lower_is_better)
            best_methods[metric_key] = best.get(dataset)
            second_best_methods[metric_key] = second.get(dataset)

        # Data rows
        for method in methods:
            if method not in summary[dataset]:
                continue

            row = f"| {METHOD_LABELS[method]} |"

            for metric_key, _, lower_is_better in metrics:
                mean = summary[dataset][method]["mean"].get(metric_key, np.nan)
                std = summary[dataset][method]["std"].get(metric_key, np.nan)

                if np.isnan(mean):
                    row += " — |"
                    continue

                # Format value
                if metric_key in ["best_epoch", "time_to_best", "time_per_epoch"]:
                    val_str = f"{mean:.1f} ± {std:.1f}"
                else:
                    val_str = f"{mean:.3f} ± {std:.3f}"

                # Mark best/second-best
                if method == best_methods.get(metric_key):
                    val_str = f"**{val_str}**"
                elif method == second_best_methods.get(metric_key):
                    val_str = f"*{val_str}*"

                row += f" {val_str} |"

            lines.append(row)

        lines.append("")

    # Summary statistics table
    num_metrics = len(metrics)
    total_comparisons = len(datasets) * num_metrics

    lines.append("---")
    lines.append("")
    lines.append("## Summary: Wins and Average Ranks")
    lines.append("")

    # Overall wins and ranks
    lines.append("### Overall Performance")
    lines.append("")
    lines.append(f"*Aggregated across {len(datasets)} datasets and {num_metrics} metrics ({total_comparisons} total comparisons)*")
    lines.append("")

    # Count wins and compute average ranks
    method_wins_overall = defaultdict(int)
    method_ranks_overall = defaultdict(list)

    for dataset in datasets:
        for metric_key, _, lower_is_better in metrics:
            # Get all valid values
            values = []
            for method in methods:
                if method in summary[dataset]:
                    mean_val = summary[dataset][method]["mean"].get(metric_key, np.nan)
                    if not np.isnan(mean_val):
                        values.append((mean_val, method))

            # Sort and assign ranks
            if values:
                values.sort(reverse=not lower_is_better)
                for rank, (_, method) in enumerate(values, 1):
                    method_ranks_overall[method].append(rank)
                    if rank == 1:
                        method_wins_overall[method] += 1

    lines.append("| Method | Total Wins | Avg Rank | Count |")
    lines.append("|--------|-----------|----------|-------|")

    # Sort by average rank
    method_stats = []
    for method in methods:
        if method in method_ranks_overall:
            avg_rank = np.mean(method_ranks_overall[method])
            wins = method_wins_overall[method]
            count = len(method_ranks_overall[method])
            method_stats.append((avg_rank, wins, method, count))

    method_stats.sort()  # Sort by average rank (ascending)

    for avg_rank, wins, method, count in method_stats:
        lines.append(f"| {METHOD_LABELS[method]} | {wins}/{total_comparisons} | {avg_rank:.2f} | {count}/{total_comparisons} |")

    lines.append("")

    # Per-category wins and ranks
    lines.append("### Performance by Metric Category")
    lines.append("")

    category_labels = {
        "threshold_invariant": "Threshold-Invariant Metrics",
        "calibration": "Calibration Metrics",
        "cross_entropy": "Cross-Entropy",
        "speed": "Speed Metrics",
    }

    for category_name, category_metrics in METRIC_CATEGORIES.items():
        category_label = category_labels[category_name]
        category_metric_keys = [m[0] for m in category_metrics]
        num_category_metrics = len(category_metric_keys)
        category_comparisons = len(datasets) * num_category_metrics

        lines.append(f"#### {category_label}")
        lines.append("")
        lines.append(f"*{len(datasets)} datasets × {num_category_metrics} metrics = {category_comparisons} comparisons*")
        lines.append("")

        # Count wins and ranks for this category
        method_wins_cat = defaultdict(int)
        method_ranks_cat = defaultdict(list)

        for dataset in datasets:
            for metric_key, _, lower_is_better in category_metrics:
                # Get all valid values
                values = []
                for method in methods:
                    if method in summary[dataset]:
                        mean_val = summary[dataset][method]["mean"].get(metric_key, np.nan)
                        if not np.isnan(mean_val):
                            values.append((mean_val, method))

                # Sort and assign ranks
                if values:
                    values.sort(reverse=not lower_is_better)
                    for rank, (_, method) in enumerate(values, 1):
                        method_ranks_cat[method].append(rank)
                        if rank == 1:
                            method_wins_cat[method] += 1

        lines.append("| Method | Wins | Avg Rank |")
        lines.append("|--------|------|----------|")

        # Sort by average rank
        method_stats_cat = []
        for method in methods:
            if method in method_ranks_cat:
                avg_rank = np.mean(method_ranks_cat[method])
                wins = method_wins_cat[method]
                method_stats_cat.append((avg_rank, wins, method))

        method_stats_cat.sort()  # Sort by average rank (ascending)

        # Show all methods for each category
        for avg_rank, wins, method in method_stats_cat:
            lines.append(f"| {METHOD_LABELS[method]} | {wins}/{category_comparisons} | {avg_rank:.2f} |")

        lines.append("")

    # Timing statistics table
    lines.append("---")
    lines.append("")
    lines.append("## Timing Statistics")
    lines.append("")
    lines.append("*Average across all datasets and runs*")
    lines.append("")
    lines.append("| Method | Avg Best Epoch | Avg Time to Best (s) | Avg Time/Epoch (s) |")
    lines.append("|--------|---------------|---------------------|-------------------|")

    for method in methods:
        epochs_all = []
        time_to_best_all = []
        time_per_epoch_all = []

        for dataset in datasets:
            if method in summary[dataset]:
                epochs = summary[dataset][method]["mean"].get("best_epoch", np.nan)
                ttb = summary[dataset][method]["mean"].get("time_to_best", np.nan)
                tpe = summary[dataset][method]["mean"].get("time_per_epoch", np.nan)

                if not np.isnan(epochs):
                    epochs_all.append(epochs)
                if not np.isnan(ttb):
                    time_to_best_all.append(ttb)
                if not np.isnan(tpe):
                    time_per_epoch_all.append(tpe)

        if epochs_all:
            avg_epochs = np.mean(epochs_all)
            avg_ttb = np.mean(time_to_best_all) if time_to_best_all else np.nan
            avg_tpe = np.mean(time_per_epoch_all) if time_per_epoch_all else np.nan

            epochs_str = f"{avg_epochs:.1f}"
            ttb_str = f"{avg_ttb:.1f}" if not np.isnan(avg_ttb) else "—"
            tpe_str = f"{avg_tpe:.1f}" if not np.isnan(avg_tpe) else "—"

            lines.append(f"| {METHOD_LABELS[method]} | {epochs_str} | {ttb_str} | {tpe_str} |")

    lines.append("")

    return "\n".join(lines)


def main():
    print("Loading Phase 1 Extended results...")
    results = load_phase1_results()
    print(f"Loaded {len(results)} experiment runs")

    print("Validating method completeness...")
    complete_methods, incomplete_methods = validate_methods(results)
    print(f"Complete methods: {len(complete_methods)}")
    if incomplete_methods:
        print(f"Incomplete methods: {len(incomplete_methods)}")
        for method, reason in incomplete_methods.items():
            print(f"  - {method}: {reason}")

    # Update excluded methods with incomplete ones
    global EXCLUDED_METHODS
    EXCLUDED_METHODS.update(incomplete_methods)

    print("Aggregating by dataset...")
    summary = aggregate_by_dataset(results)

    print("Generating markdown table...")
    markdown = generate_table_markdown(summary)

    output_path = Path("analysis/PHASE1_EXTENDED_COMPREHENSIVE_TABLE.md")
    with open(output_path, "w") as f:
        f.write(markdown)

    print(f"\n✅ Table saved to: {output_path}")
    print(f"\nDatasets: {len(summary)}")
    print(f"Methods found: {len({m for ds in summary.values() for m in ds.keys()})}")


if __name__ == "__main__":
    main()
