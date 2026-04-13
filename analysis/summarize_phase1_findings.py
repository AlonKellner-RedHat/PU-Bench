#!/usr/bin/env python3
"""
Phase 1 Results Summary

Generates a comprehensive summary of Phase 1 experimental findings,
including statistical comparisons and rankings across all metrics.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import re

# Configuration
RESULTS_DIR = Path("results_comprehensive")
OUTPUT_FILE = Path("analysis/PHASE1_SUMMARY.md")

# Phase 1 datasets
PHASE1_DATASETS = ["20News", "Connect4", "FashionMNIST", "IMDB", "MNIST", "Mushrooms", "Spambase"]

# Key metrics to analyze
KEY_METRICS = {
    "Calibration": ["ece", "brier", "anice"],
    "Ranking": ["auc", "ap", "max_f1"],
    "Classification": ["accuracy", "f1", "precision", "recall"]
}


def load_phase1_results():
    """Load all Phase 1 results."""
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

        with open(json_file) as f:
            data = json.load(f)

        experiment_name = data["experiment"]

        # Extract true prior
        match = re.search(r'trueprior([\d.]+)', experiment_name)
        if not match:
            continue
        true_prior = float(match.group(1))

        # Extract label frequency (c)
        match_c = re.search(r'_c([\d.]+)_', experiment_name)
        c_value = float(match_c.group(1)) if match_c else None

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
                    method_id = f"{method_key}_1.0"
                else:
                    method_id = f"{method_key}_auto"
            else:
                method_id = method_key

            # Store result
            result_key = f"{dataset}_{experiment_name}"
            results[result_key][method_id] = {
                "dataset": dataset,
                "true_prior": true_prior,
                "c_value": c_value,
                "test_metrics": {k.replace("test_", ""): v for k, v in metrics.items() if k.startswith("test_")}
            }

    return results


def aggregate_results(results):
    """Aggregate results by method across all experiments."""
    method_stats = defaultdict(lambda: defaultdict(list))

    for exp_key, methods in results.items():
        for method_id, data in methods.items():
            for metric, value in data["test_metrics"].items():
                method_stats[method_id][metric].append(value)

    # Compute statistics
    summary = {}
    for method in method_stats:
        summary[method] = {}
        for metric in method_stats[method]:
            values = method_stats[method][metric]
            summary[method][metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "median": np.median(values),
                "min": np.min(values),
                "max": np.max(values),
                "count": len(values)
            }

    return summary


def rank_methods(summary, metric, higher_is_better=True):
    """Rank methods by a specific metric."""
    rankings = []
    for method in summary:
        if metric in summary[method]:
            rankings.append((method, summary[method][metric]["mean"]))

    rankings.sort(key=lambda x: x[1], reverse=higher_is_better)
    return rankings


def generate_summary_report(results, summary):
    """Generate comprehensive summary report."""
    lines = []

    lines.append("# Phase 1 Experimental Results Summary")
    lines.append("")
    lines.append(f"**Generated:** 2026-04-13")
    lines.append("")
    lines.append(f"**Total Experiments:** {len(results)}")
    lines.append(f"**Datasets:** {', '.join(PHASE1_DATASETS)}")
    lines.append(f"**Methods Evaluated:** {len(summary)}")
    lines.append("")

    # List all methods
    lines.append("## Methods Analyzed")
    lines.append("")
    for method in sorted(summary.keys()):
        count = summary[method][list(summary[method].keys())[0]]["count"]
        lines.append(f"- **{method}**: {count} experiments")
    lines.append("")

    # Overall rankings by metric category
    lines.append("## Overall Method Rankings")
    lines.append("")

    for category, metrics in KEY_METRICS.items():
        lines.append(f"### {category} Metrics")
        lines.append("")

        for metric in metrics:
            # Determine if higher is better
            higher_is_better = metric not in ["ece", "brier", "anice", "snice", "oracle_ce", "error"]

            rankings = rank_methods(summary, metric, higher_is_better)

            if not rankings:
                continue

            direction = "↑" if higher_is_better else "↓"
            lines.append(f"#### {metric.upper()} {direction}")
            lines.append("")
            lines.append("| Rank | Method | Mean | Std | Median |")
            lines.append("|------|--------|------|-----|--------|")

            for rank, (method, mean_val) in enumerate(rankings[:10], 1):
                std_val = summary[method][metric]["std"]
                median_val = summary[method][metric]["median"]
                lines.append(f"| {rank} | {method} | {mean_val:.4f} | {std_val:.4f} | {median_val:.4f} |")

            lines.append("")

    # Method comparisons
    lines.append("## Key Method Comparisons")
    lines.append("")

    # VPU vs VPU-Mean-Prior (auto) vs VPU-Mean-Prior (0.5) - No Mixup
    lines.append("### No-Mixup: VPU vs VPU-Mean-Prior (auto) vs VPU-Mean-Prior (0.5)")
    lines.append("")

    comparison_methods = ["vpu_nomixup", "vpu_nomixup_mean_prior_auto", "vpu_nomixup_mean_prior_0.5"]
    lines.append("| Metric | VPU | VPU-MP(auto) | VPU-MP(0.5) | Best |")
    lines.append("|--------|-----|--------------|-------------|------|")

    for metric in ["auc", "ap", "ece", "brier"]:
        if metric not in summary.get("vpu_nomixup", {}):
            continue

        values = {m: summary[m][metric]["mean"] for m in comparison_methods if m in summary and metric in summary[m]}

        if not values:
            continue

        higher_is_better = metric not in ["ece", "brier"]
        best_method = max(values.items(), key=lambda x: x[1] if higher_is_better else -x[1])[0]

        row = f"| {metric} "
        for m in comparison_methods:
            if m in values:
                marker = "**" if m == best_method else ""
                row += f"| {marker}{values[m]:.4f}{marker} "
            else:
                row += "| N/A "
        row += f"| {best_method.split('_')[-1]} |"
        lines.append(row)

    lines.append("")

    # Mixup vs No-Mixup for VPU
    lines.append("### Mixup Impact: VPU")
    lines.append("")
    lines.append("| Metric | No Mixup | With Mixup | Difference | Winner |")
    lines.append("|--------|----------|------------|------------|--------|")

    for metric in ["auc", "ap", "ece", "brier"]:
        if metric in summary.get("vpu_nomixup", {}) and metric in summary.get("vpu", {}):
            nomixup = summary["vpu_nomixup"][metric]["mean"]
            mixup = summary["vpu"][metric]["mean"]
            diff = mixup - nomixup

            higher_is_better = metric not in ["ece", "brier"]
            winner = "Mixup" if (diff > 0 and higher_is_better) or (diff < 0 and not higher_is_better) else "No Mixup"

            lines.append(f"| {metric} | {nomixup:.4f} | {mixup:.4f} | {diff:+.4f} | {winner} |")

    lines.append("")

    # Mixup vs No-Mixup for VPU-Mean-Prior (auto)
    lines.append("### Mixup Impact: VPU-Mean-Prior (auto)")
    lines.append("")
    lines.append("| Metric | No Mixup | With Mixup | Difference | Winner |")
    lines.append("|--------|----------|------------|------------|--------|")

    for metric in ["auc", "ap", "ece", "brier"]:
        if metric in summary.get("vpu_nomixup_mean_prior_auto", {}) and metric in summary.get("vpu_mean_prior_auto", {}):
            nomixup = summary["vpu_nomixup_mean_prior_auto"][metric]["mean"]
            mixup = summary["vpu_mean_prior_auto"][metric]["mean"]
            diff = mixup - nomixup

            higher_is_better = metric not in ["ece", "brier"]
            winner = "Mixup" if (diff > 0 and higher_is_better) or (diff < 0 and not higher_is_better) else "No Mixup"

            lines.append(f"| {metric} | {nomixup:.4f} | {mixup:.4f} | {diff:+.4f} | {winner} |")

    lines.append("")

    # Baseline comparisons
    lines.append("### Comparison with Baselines (nnPU, Dist-PU)")
    lines.append("")
    lines.append("Best VPU variant vs baselines (averaged across all experiments):")
    lines.append("")

    baseline_methods = ["nnpu", "distpu"]
    vpu_methods = [m for m in summary.keys() if "vpu" in m]

    lines.append("| Metric | Best VPU | nnPU | Dist-PU |")
    lines.append("|--------|----------|------|---------|")

    for metric in ["auc", "ap", "ece", "brier"]:
        higher_is_better = metric not in ["ece", "brier"]

        # Find best VPU variant for this metric
        vpu_values = {m: summary[m][metric]["mean"] for m in vpu_methods if m in summary and metric in summary[m]}
        if not vpu_values:
            continue

        best_vpu = max(vpu_values.items(), key=lambda x: x[1] if higher_is_better else -x[1])
        nnpu_val = summary["nnpu"][metric]["mean"] if "nnpu" in summary and metric in summary["nnpu"] else np.nan
        distpu_val = summary["distpu"][metric]["mean"] if "distpu" in summary and metric in summary["distpu"] else np.nan

        lines.append(f"| {metric} | **{best_vpu[1]:.4f}** ({best_vpu[0]}) | {nnpu_val:.4f} | {distpu_val:.4f} |")

    lines.append("")

    # Key findings
    lines.append("## Key Findings")
    lines.append("")

    # Auto vs 0.5 analysis
    if "vpu_nomixup_mean_prior_auto" in summary and "vpu_nomixup_mean_prior_0.5" in summary:
        auto_wins = 0
        half_wins = 0
        for metric in ["auc", "ap", "ece", "brier"]:
            if metric in summary["vpu_nomixup_mean_prior_auto"] and metric in summary["vpu_nomixup_mean_prior_0.5"]:
                auto_val = summary["vpu_nomixup_mean_prior_auto"][metric]["mean"]
                half_val = summary["vpu_nomixup_mean_prior_0.5"][metric]["mean"]
                higher_is_better = metric not in ["ece", "brier"]

                if (auto_val > half_val and higher_is_better) or (auto_val < half_val and not higher_is_better):
                    auto_wins += 1
                else:
                    half_wins += 1

        lines.append(f"### Auto vs 0.5 Prior (VPU-Mean-Prior, No Mixup)")
        lines.append(f"- Auto wins: {auto_wins}/4 key metrics")
        lines.append(f"- 0.5 wins: {half_wins}/4 key metrics")
        lines.append("")

    # Mixup impact
    mixup_helps_count = 0
    mixup_hurts_count = 0

    for method_base in ["vpu", "vpu_mean_prior_auto", "vpu_mean_prior_0.5"]:
        nomixup = method_base.replace("vpu_mean_prior", "vpu_nomixup_mean_prior").replace("vpu", "vpu_nomixup")
        if nomixup in summary and method_base in summary:
            for metric in ["auc", "ap"]:
                if metric in summary[nomixup] and metric in summary[method_base]:
                    nomixup_val = summary[nomixup][metric]["mean"]
                    mixup_val = summary[method_base][metric]["mean"]
                    if mixup_val > nomixup_val:
                        mixup_helps_count += 1
                    else:
                        mixup_hurts_count += 1

    lines.append(f"### Mixup Impact")
    lines.append(f"- Mixup improves performance: {mixup_helps_count} cases")
    lines.append(f"- Mixup hurts performance: {mixup_hurts_count} cases")
    lines.append("")

    lines.append("## Recommendations")
    lines.append("")
    lines.append("Based on Phase 1 results:")
    lines.append("")

    # Find overall best method for key metrics
    best_auc = rank_methods(summary, "auc", higher_is_better=True)[0]
    best_ece = rank_methods(summary, "ece", higher_is_better=False)[0]

    lines.append(f"1. **Best Overall Ranking (AUC):** {best_auc[0]} (AUC={best_auc[1]:.4f})")
    lines.append(f"2. **Best Calibration (ECE):** {best_ece[0]} (ECE={best_ece[1]:.4f})")
    lines.append("")
    lines.append("For detailed visualizations, see plots in `analysis/plots/phase1_comprehensive/`")
    lines.append("")

    return "\n".join(lines)


def main():
    print("Loading Phase 1 results...")
    results = load_phase1_results()
    print(f"Loaded {len(results)} experiments")

    print("Aggregating statistics...")
    summary = aggregate_results(results)
    print(f"Methods found: {list(summary.keys())}")

    print("Generating summary report...")
    report = generate_summary_report(results, summary)

    print(f"Writing to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        f.write(report)

    print(f"Summary report saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
