#!/usr/bin/env python3
"""
Generate merged Phase 1 + Phase 3 results report.

A streamlined, publication-ready summary with key methods and metrics.
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# Results directories
PHASE1_DIR = Path("results_phase1_extended")
PHASE3_DIR = Path("results_phase3")

# Phase configuration
PHASE1_DATASETS = ["20News", "Connect4", "FashionMNIST", "IMDB", "MNIST", "Mushrooms", "Spambase"]
PHASE3_DATASETS = PHASE1_DATASETS
PHASE3_SEEDS = [42, 456, 789, 1024, 2048]

# Selected methods for streamlined report
SELECTED_METHODS = [
    # Phase 1 baselines
    "nnpu", "nnpusb", "bbepu", "lbe", "distpu",
    "selfpu", "p3mixe", "p3mixc", "robustpu",
    # VPU variants
    "vpu_nomixup", "vpu_nomixup_mean_prior_0.69", "vpu_nomixup_mean_prior_auto",
    "vpu", "vpu_mean_prior_0.69", "vpu_mean_prior_auto",
    # Oracles
    "pn_naive", "oracle_bce",
]

METHOD_ORDER = [
    # Baselines
    "nnpu", "nnpusb", "bbepu", "lbe", "distpu",
    "selfpu", "p3mixe", "p3mixc", "robustpu",
    # VPU (no mixup)
    "vpu_nomixup", "vpu_nomixup_mean_prior_auto", "vpu_nomixup_mean_prior_0.69",
    # VPU (with mixup)
    "vpu", "vpu_mean_prior_auto", "vpu_mean_prior_0.69",
    # Oracles
    "pn_naive", "oracle_bce",
]

METHOD_LABELS = {
    "nnpu": "nnPU",
    "nnpusb": "nnPU-SB",
    "bbepu": "BBE-PU",
    "lbe": "LBE",
    "distpu": "Dist-PU",
    "selfpu": "Self-PU",
    "p3mixe": "P3Mix-E",
    "p3mixc": "P3Mix-C",
    "robustpu": "Robust-PU",
    "vpu": "VPU",
    "vpu_nomixup": "VPU-nomix",
    "vpu_mean_prior_auto": "VPU-MP(auto)",
    "vpu_mean_prior_0.69": "VPU-MP(0.69)",
    "vpu_nomixup_mean_prior_auto": "VPU-nomix-MP(auto)",
    "vpu_nomixup_mean_prior_0.69": "VPU-nomix-MP(0.69)",
    "oracle_bce": "Oracle-PN",
    "pn_naive": "PN-Naive",
}


def parse_phase3_filename(filename):
    """Extract dataset, c, and π from Phase 3 filename."""
    parts = filename.stem.split("_")
    dataset = parts[0]

    c_str = [p for p in parts if p.startswith("c") and len(p) > 1 and p[1].isdigit()][0]
    c = float(c_str[1:])

    pi_str = [p for p in parts if p.startswith("trueprior")][0]
    pi = float(pi_str[9:])

    return dataset, c, pi


def load_phase1_results():
    """Load Phase 1 results (7 datasets, 10 seeds, 3 c values)."""
    results = defaultdict(lambda: defaultdict(list))

    for json_file in PHASE1_DIR.glob("seed_*/*.json"):
        # Skip Phase 2 datasets
        if any(ds in json_file.name for ds in ["CIFAR10", "AlzheimerMRI"]):
            continue

        # Extract dataset
        dataset = None
        for ds in PHASE1_DATASETS:
            if json_file.name.startswith(ds):
                dataset = ds
                break

        if dataset is None:
            continue

        with open(json_file) as f:
            data = json.load(f)

        for method_key, method_data in data["runs"].items():
            if "best" not in method_data or "metrics" not in method_data["best"]:
                continue

            # Determine method identifier
            if method_key in ["vpu_nomixup_mean_prior", "vpu_mean_prior"]:
                if "methodprior0.69" in json_file.name:
                    method_id = f"{method_key}_0.69"
                elif "methodprior1" in json_file.name:
                    method_id = f"{method_key}_1.0"
                else:
                    method_id = f"{method_key}_auto"
            else:
                method_id = method_key

            # Only keep selected methods
            if method_id not in SELECTED_METHODS:
                continue

            metrics = method_data["best"]["metrics"]

            # Store test metrics (exclude timing metrics)
            result = {
                k.replace("test_", ""): v
                for k, v in metrics.items()
                if k.startswith("test_") and k not in ["test_duration", "test_time_to_best", "test_time_per_epoch"]
            }

            # Add best_epoch if available
            if "best" in method_data and "epoch" in method_data["best"]:
                result["best_epoch"] = method_data["best"]["epoch"]

            results[method_id]["all"].append(result)

    return results


def load_phase3_results():
    """Load Phase 3 results (7 datasets, 5 seeds, 7×7 c×π grid)."""
    results = defaultdict(lambda: defaultdict(list))

    for seed in PHASE3_SEEDS:
        seed_dir = PHASE3_DIR / f"seed_{seed}"
        if not seed_dir.exists():
            continue

        for json_file in seed_dir.glob("*.json"):
            dataset, c, pi = parse_phase3_filename(json_file)

            with open(json_file) as f:
                data = json.load(f)

            for method_key, method_data in data.get("runs", {}).items():
                if "best" not in method_data or "metrics" not in method_data["best"]:
                    continue

                metrics = method_data["best"]["metrics"]

                # Determine method identifier
                if method_key in ["vpu_mean_prior", "vpu_nomixup_mean_prior"]:
                    if "methodprior0.69" in json_file.name:
                        method_id = f"{method_key}_0.69"
                    elif "methodprior1" in json_file.name:
                        method_id = f"{method_key}_1.0"
                    else:
                        method_id = f"{method_key}_auto"
                else:
                    method_id = method_key

                # Only keep selected methods
                if method_id not in SELECTED_METHODS:
                    continue

                # Store test metrics
                result = {
                    k.replace("test_", ""): v
                    for k, v in metrics.items()
                    if k.startswith("test_")
                }
                result["dataset"] = dataset
                result["c"] = c
                result["pi"] = pi
                result["seed"] = seed

                results[method_id]["all"].append(result)

    return results


def compute_summary_stats(results, filter_fn=None):
    """Compute mean ± std for each method."""
    summary = {}

    for method, data in results.items():
        values = data["all"]
        if filter_fn:
            values = [v for v in values if filter_fn(v)]

        if not values:
            continue

        # Aggregate metrics
        metrics = {}
        for key in ["auc", "ap", "max_f1", "accuracy", "f1", "precision", "recall",
                   "ece", "brier", "anice", "oracle_ce", "best_epoch"]:
            vals = [v[key] for v in values if key in v and v.get(key) is not None and not np.isnan(v[key])]
            if vals:
                metrics[key] = {
                    "mean": np.mean(vals),
                    "std": np.std(vals),
                    "count": len(vals)
                }

        summary[method] = metrics

    return summary


def format_metric(mean, std):
    """Format metric as mean ± std."""
    if np.isnan(mean) or np.isnan(std):
        return "—"
    return f"{mean:.3f} ± {std:.3f}"


def find_best_methods(summary, metric, lower_is_better=False):
    """Find best and second-best methods."""
    values = []
    for method, metrics in summary.items():
        if metric in metrics:
            mean_val = metrics[metric]["mean"]
            if not np.isnan(mean_val):
                values.append((mean_val, method))

    if len(values) < 2:
        return None, None

    values.sort(reverse=not lower_is_better)
    return values[0][1], values[1][1]


def generate_table(summary, methods, title, description, metrics_display):
    """Generate a markdown table."""
    lines = []
    lines.append(f"### {title}")
    lines.append("")
    lines.append(description)
    lines.append("")

    # Find best/second-best
    best_methods = {}
    second_best_methods = {}
    for metric, _, lower_is_better in metrics_display:
        best, second = find_best_methods(summary, metric, lower_is_better)
        best_methods[metric] = best
        second_best_methods[metric] = second

    # Header
    header_cols = ["Method"] + [label for _, label, _ in metrics_display]
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("|" + "|".join(["--------" for _ in header_cols]) + "|")

    # Rows
    for method in methods:
        if method not in summary:
            continue

        label = METHOD_LABELS.get(method, method)
        row = [label]

        for metric, _, lower_is_better in metrics_display:
            if metric in summary[method]:
                mean = summary[method][metric]["mean"]
                std = summary[method][metric]["std"]
                formatted = format_metric(mean, std)

                # Add bold/italic markers
                if method == best_methods[metric]:
                    formatted = f"**{formatted}**"
                elif method == second_best_methods[metric]:
                    formatted = f"*{formatted}*"

                row.append(formatted)
            else:
                row.append("—")

        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    return "\n".join(lines)


def generate_ranking_table(summary, methods, metrics_list, title):
    """Generate ranking table showing wins and average rank."""
    lines = []
    lines.append(f"#### {title}")
    lines.append("")

    method_wins = defaultdict(int)
    method_ranks = defaultdict(list)

    for metric, lower_is_better in metrics_list:
        values = []
        for method in methods:
            if method in summary and metric in summary[method]:
                mean_val = summary[method][metric]["mean"]
                if not np.isnan(mean_val):
                    values.append((mean_val, method))

        if not values:
            continue

        values.sort(reverse=not lower_is_better)
        ranks = {method: rank+1 for rank, (_, method) in enumerate(values)}

        for method in methods:
            if method in ranks:
                rank = ranks[method]
                method_ranks[method].append(rank)
                if rank == 1:
                    method_wins[method] += 1

    # Calculate average ranks
    method_avg_ranks = {
        method: np.mean(ranks) if ranks else float('inf')
        for method, ranks in method_ranks.items()
    }

    # Sort by average rank
    sorted_methods = sorted(method_avg_ranks.items(), key=lambda x: x[1])

    total_comparisons = len(metrics_list)

    lines.append(f"*{total_comparisons} metrics*")
    lines.append("")
    lines.append("| Method | Wins | Avg Rank |")
    lines.append("|--------|------|----------|")

    for method, avg_rank in sorted_methods:
        label = METHOD_LABELS.get(method, method)
        wins = method_wins[method]
        lines.append(f"| {label} | {wins}/{total_comparisons} | {avg_rank:.2f} |")

    lines.append("")
    return "\n".join(lines)


def main():
    print("Loading Phase 1 results...", file=sys.stderr)
    phase1_results = load_phase1_results()
    phase1_total = sum(len(v['all']) for v in phase1_results.values())
    print(f"  Loaded {phase1_total} runs", file=sys.stderr)

    print("Loading Phase 3 results...", file=sys.stderr)
    phase3_results = load_phase3_results()
    phase3_total = sum(len(v['all']) for v in phase3_results.values())
    print(f"  Loaded {phase3_total} runs", file=sys.stderr)

    # Metrics to display
    metrics_display = [
        ("auc", "AUC ↑", False),
        ("ap", "AP ↑", False),
        ("max_f1", "Max F1 ↑", False),
        ("accuracy", "Accuracy ↑", False),
        ("f1", "F1 ↑", False),
        ("ece", "ECE ↓", True),
        ("brier", "Brier ↓", True),
    ]

    # All metrics for rankings
    all_metrics = [
        ("auc", False), ("ap", False), ("max_f1", False),
        ("accuracy", False), ("f1", False), ("precision", False), ("recall", False),
        ("ece", True), ("brier", True), ("anice", True), ("oracle_ce", True)
    ]

    # Metric categories
    metric_categories = {
        "Threshold-Invariant": [("auc", False), ("ap", False), ("max_f1", False)],
        "Threshold-Dependent": [("accuracy", False), ("f1", False), ("precision", False), ("recall", False)],
        "Calibration": [("ece", True), ("brier", True), ("anice", True)],
        "Cross-Entropy": [("oracle_ce", True)],
    }

    output = []

    # Header
    output.append("# Positive-Unlabeled Learning: Comprehensive Experimental Results")
    output.append("")
    output.append("**A streamlined comparison of 17 PU learning methods across two experimental phases.**")
    output.append("")
    output.append("---")
    output.append("")

    # Experimental setup
    output.append("## Experimental Design")
    output.append("")
    output.append("### Phase 1: Fixed Prior, Variable Label Frequency")
    output.append("")
    output.append("- **Datasets**: 7 (20News, Connect4, FashionMNIST, IMDB, MNIST, Mushrooms, Spambase)")
    output.append("- **Random seeds**: 10 [42, 456, 789, 1024, 2048, 3000, 4096, 5555, 6789, 8192]")
    output.append("- **Label frequency (c)**: 3 values [0.1, 0.3, 0.5]")
    output.append("- **True prior (π)**: Dataset natural prior (fixed per dataset)")
    output.append("- **Configurations**: 7 datasets × 10 seeds × 3 c = 210 per method")
    output.append("")

    output.append("### Phase 3: Full Hyperparameter Grid")
    output.append("")
    output.append("- **Datasets**: 7 (same as Phase 1)")
    output.append("- **Random seeds**: 5 [42, 456, 789, 1024, 2048]")
    output.append("- **Label frequency (c)**: 7 values [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]")
    output.append("- **True prior (π)**: 7 values [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]")
    output.append("- **Configurations**: 7 datasets × 5 seeds × 7 c × 7 π = 1,715 per method")
    output.append("")

    output.append("### Methods Evaluated")
    output.append("")
    output.append("**Baseline PU Methods (9):**")
    output.append("- nnPU, nnPU-SB, BBE-PU, LBE, Dist-PU")
    output.append("- Self-PU, P3Mix-E, P3Mix-C, Robust-PU")
    output.append("")
    output.append("**VPU Variants (6):**")
    output.append("- VPU, VPU-nomix (base methods)")
    output.append("- VPU-MP(auto), VPU-MP(0.69) (with mixup, mean-prior regularization)")
    output.append("- VPU-nomix-MP(auto), VPU-nomix-MP(0.69) (without mixup, mean-prior regularization)")
    output.append("")
    output.append("**Oracle Baselines (2):**")
    output.append("- PN-Naive (treats unlabeled as negative)")
    output.append("- Oracle-PN (trained with true labels)")
    output.append("")
    output.append("---")
    output.append("")

    # Phase 1 results
    print("Computing Phase 1 statistics...", file=sys.stderr)
    summary_p1 = compute_summary_stats(phase1_results)

    output.append("## Phase 1 Results: Performance with Fixed Priors")
    output.append("")
    output.append(generate_table(
        summary_p1, METHOD_ORDER,
        "Overall Performance",
        "*Mean ± Std across 210 runs per method (7 datasets × 10 seeds × 3 label frequencies). **Bold** = best, *italic* = second-best per metric.*",
        metrics_display
    ))

    # Phase 3 results
    print("Computing Phase 3 statistics...", file=sys.stderr)
    summary_p3_all = compute_summary_stats(phase3_results)

    output.append("## Phase 3 Results: Performance Across Full Hyperparameter Grid")
    output.append("")
    output.append(generate_table(
        summary_p3_all, METHOD_ORDER,
        "Overall Performance",
        "*Mean ± Std across 1,715 runs per method (7 datasets × 5 seeds × 7 c × 7 π). **Bold** = best, *italic* = second-best per metric.*",
        metrics_display
    ))

    # Prior regime analysis
    output.append("### Performance by Prior Regime")
    output.append("")

    summary_p3_pi_low = compute_summary_stats(phase3_results, lambda v: v.get("pi") is not None and v["pi"] < 0.5)
    output.append(generate_table(
        summary_p3_pi_low, METHOD_ORDER,
        "Low Priors (π < 0.5)",
        "*Mean ± Std across configurations with π ∈ {0.01, 0.1, 0.3}. **Bold** = best, *italic* = second-best.*",
        metrics_display
    ))

    summary_p3_pi_high = compute_summary_stats(phase3_results, lambda v: v.get("pi") is not None and v["pi"] >= 0.5)
    output.append(generate_table(
        summary_p3_pi_high, METHOD_ORDER,
        "High Priors (π ≥ 0.5)",
        "*Mean ± Std across configurations with π ∈ {0.5, 0.7, 0.9, 0.99}. **Bold** = best, *italic* = second-best.*",
        metrics_display
    ))

    # Label frequency regime analysis
    output.append("### Performance by Label Frequency Regime")
    output.append("")

    summary_p3_c_low = compute_summary_stats(phase3_results, lambda v: v.get("c") is not None and v["c"] < 0.5)
    output.append(generate_table(
        summary_p3_c_low, METHOD_ORDER,
        "Low Label Frequency (c < 0.5)",
        "*Mean ± Std across configurations with c ∈ {0.01, 0.1, 0.3}. **Bold** = best, *italic* = second-best.*",
        metrics_display
    ))

    summary_p3_c_high = compute_summary_stats(phase3_results, lambda v: v.get("c") is not None and v["c"] >= 0.5)
    output.append(generate_table(
        summary_p3_c_high, METHOD_ORDER,
        "High Label Frequency (c ≥ 0.5)",
        "*Mean ± Std across configurations with c ∈ {0.5, 0.7, 0.9, 0.99}. **Bold** = best, *italic* = second-best.*",
        metrics_display
    ))

    # Rankings
    output.append("---")
    output.append("")
    output.append("## Method Rankings")
    output.append("")

    # Phase 1 rankings
    output.append("### Phase 1: Fixed Prior Experiments")
    output.append("")
    output.append(generate_ranking_table(summary_p1, METHOD_ORDER, all_metrics,
                                        "Overall Performance"))

    # Category breakdowns for Phase 1
    for category_name, category_metrics in metric_categories.items():
        output.append(generate_ranking_table(summary_p1, METHOD_ORDER, category_metrics,
                                            f"{category_name} Metrics"))

    # Phase 3 rankings
    output.append("### Phase 3: Full Grid Experiments")
    output.append("")
    output.append(generate_ranking_table(summary_p3_all, METHOD_ORDER, all_metrics,
                                        "Overall Performance"))

    # Category breakdowns for Phase 3
    for category_name, category_metrics in metric_categories.items():
        output.append(generate_ranking_table(summary_p3_all, METHOD_ORDER, category_metrics,
                                            f"{category_name} Metrics"))

    # Rankings by regime (Phase 3 only)
    output.append("### Phase 3: By Hyperparameter Regime")
    output.append("")

    output.append(generate_ranking_table(summary_p3_pi_low, METHOD_ORDER, all_metrics,
                                        "Low Priors (π < 0.5)"))
    output.append(generate_ranking_table(summary_p3_pi_high, METHOD_ORDER, all_metrics,
                                        "High Priors (π ≥ 0.5)"))
    output.append(generate_ranking_table(summary_p3_c_low, METHOD_ORDER, all_metrics,
                                        "Low Label Frequency (c < 0.5)"))
    output.append(generate_ranking_table(summary_p3_c_high, METHOD_ORDER, all_metrics,
                                        "High Label Frequency (c ≥ 0.5)"))

    # Key findings
    output.append("---")
    output.append("")
    output.append("## Key Findings")
    output.append("")

    output.append("### Overall Performance")
    output.append("")

    # Phase 1 top performers
    p1_auc_ranks = []
    for method in METHOD_ORDER:
        if method in summary_p1 and "auc" in summary_p1[method]:
            p1_auc_ranks.append((summary_p1[method]["auc"]["mean"], method))
    p1_auc_ranks.sort(reverse=True)

    output.append("**Phase 1 (Fixed Prior):**")
    for i, (auc, method) in enumerate(p1_auc_ranks[:3], 1):
        label = METHOD_LABELS[method]
        std = summary_p1[method]["auc"]["std"]
        output.append(f"  {i}. **{label}**: AUC = {auc:.3f} ± {std:.3f}")
    output.append("")

    # Phase 3 top performers
    p3_auc_ranks = []
    for method in METHOD_ORDER:
        if method in summary_p3_all and "auc" in summary_p3_all[method]:
            p3_auc_ranks.append((summary_p3_all[method]["auc"]["mean"], method))
    p3_auc_ranks.sort(reverse=True)

    output.append("**Phase 3 (Full Grid):**")
    for i, (auc, method) in enumerate(p3_auc_ranks[:3], 1):
        label = METHOD_LABELS[method]
        std = summary_p3_all[method]["auc"]["std"]
        output.append(f"  {i}. **{label}**: AUC = {auc:.3f} ± {std:.3f}")
    output.append("")

    # VPU variants comparison
    output.append("### VPU Method Variants")
    output.append("")

    vpu_variants = ["vpu", "vpu_nomixup", "vpu_mean_prior_auto", "vpu_mean_prior_0.69",
                    "vpu_nomixup_mean_prior_auto", "vpu_nomixup_mean_prior_0.69"]

    output.append("**Phase 3 AUC Performance:**")
    vpu_auc = [(summary_p3_all[m]["auc"]["mean"], m) for m in vpu_variants if m in summary_p3_all and "auc" in summary_p3_all[m]]
    vpu_auc.sort(reverse=True)
    for i, (auc, method) in enumerate(vpu_auc, 1):
        label = METHOD_LABELS[method]
        std = summary_p3_all[method]["auc"]["std"]
        output.append(f"  {i}. **{label}**: {auc:.3f} ± {std:.3f}")
    output.append("")

    # Stability
    output.append("### Method Stability")
    output.append("")
    output.append("Standard deviation of AUC across Phase 3 configurations (lower = more stable):")
    output.append("")

    stability = []
    for method in METHOD_ORDER:
        if method in summary_p3_all and "auc" in summary_p3_all[method]:
            std = summary_p3_all[method]["auc"]["std"]
            mean = summary_p3_all[method]["auc"]["mean"]
            stability.append((std, method, mean))

    stability.sort()
    for i, (std, method, mean) in enumerate(stability[:5], 1):
        label = METHOD_LABELS[method]
        output.append(f"  {i}. **{label}**: σ = {std:.4f} (mean AUC = {mean:.3f})")
    output.append("")

    # Write output
    output_file = Path("analysis/MERGED_RESULTS.md")
    with open(output_file, "w") as f:
        f.write("\n".join(output))

    print(f"\n✅ Merged results saved to: {output_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
