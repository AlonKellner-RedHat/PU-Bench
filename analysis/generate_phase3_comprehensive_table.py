#!/usr/bin/env python3
"""
Generate comprehensive Phase 3 results table with stability analysis.

Focuses on:
1. Performance across all configurations (7×7×7 grid)
2. Stability comparison: all priors vs priors < 0.5
3. Performance at extreme cases (low/high c, low/high π)
4. Method ranking by prior regime
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy import stats

RESULTS_DIR = Path("results_phase3/seed_42")

# Configuration
DATASETS = ["20News", "Connect4", "FashionMNIST", "IMDB", "MNIST", "Mushrooms", "Spambase"]
C_VALUES = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
PI_VALUES = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]

METHOD_ORDER = [
    # Base methods
    "vpu", "vpu_nomixup",
    # VPU-nomix with mean-prior
    "vpu_nomixup_mean_prior_auto",
    "vpu_nomixup_mean_prior_0.353",
    "vpu_nomixup_mean_prior_0.5",
    "vpu_nomixup_mean_prior_0.69",
    "vpu_nomixup_mean_prior_1.0",
    # VPU with mean-prior
    "vpu_mean_prior_auto",
    "vpu_mean_prior_0.353",
    "vpu_mean_prior_0.5",
    "vpu_mean_prior_0.69",
    "vpu_mean_prior_1.0",
    # Oracles
    "pn_naive", "oracle_bce",
]

METHOD_LABELS = {
    "vpu": "VPU",
    "vpu_nomixup": "VPU-nomix",
    "vpu_mean_prior_auto": "VPU-MP(auto)",
    "vpu_mean_prior_0.353": "VPU-MP(0.353)",
    "vpu_mean_prior_0.5": "VPU-MP(0.5)",
    "vpu_mean_prior_0.69": "VPU-MP(0.69)",
    "vpu_mean_prior_1.0": "VPU-MP(1.0)",
    "vpu_nomixup_mean_prior_auto": "VPU-nomix-MP(auto)",
    "vpu_nomixup_mean_prior_0.353": "VPU-nomix-MP(0.353)",
    "vpu_nomixup_mean_prior_0.5": "VPU-nomix-MP(0.5)",
    "vpu_nomixup_mean_prior_0.69": "VPU-nomix-MP(0.69)",
    "vpu_nomixup_mean_prior_1.0": "VPU-nomix-MP(1.0)",
    "oracle_bce": "Oracle-PN",
    "pn_naive": "PN-Naive",
}


def parse_filename(filename):
    """Extract dataset, c, π, and method_prior from filename."""
    parts = filename.stem.split("_")
    dataset = parts[0]

    c_str = [p for p in parts if p.startswith("c") and p[1:2].isdigit()][0]
    c = float(c_str[1:])

    pi_str = [p for p in parts if p.startswith("trueprior")][0]
    pi = float(pi_str[9:])

    if "methodprior" in filename.stem:
        if "methodprior_auto" in filename.stem:
            method_prior = "auto"
        else:
            mp_str = [p for p in parts if p.startswith("methodprior")][0]
            method_prior = mp_str[11:]
    else:
        method_prior = None

    return dataset, c, pi, method_prior


def load_phase3_results():
    """Load all Phase 3 results."""
    results = defaultdict(lambda: defaultdict(list))

    for json_file in RESULTS_DIR.glob("*.json"):
        dataset, c, pi, method_prior = parse_filename(json_file)

        with open(json_file) as f:
            data = json.load(f)

        for method_key, method_data in data.get("runs", {}).items():
            if "best" not in method_data or "metrics" not in method_data["best"]:
                continue

            metrics = method_data["best"]["metrics"]

            # Determine full method identifier
            if method_key in ["vpu_mean_prior", "vpu_nomixup_mean_prior"]:
                method_id = f"{method_key}_{method_prior}"
            else:
                method_id = method_key

            # Store test metrics
            result = {
                k.replace("test_", ""): v
                for k, v in metrics.items()
                if k.startswith("test_")
            }
            result["dataset"] = dataset
            result["c"] = c
            result["pi"] = pi

            results[method_id]["all"].append(result)

    return results


def compute_summary_stats(results, filter_fn=None):
    """Compute mean ± std for each method, optionally filtering results."""
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
                   "ece", "brier", "anice", "oracle_ce"]:
            vals = [v[key] for v in values if key in v and not np.isnan(v[key])]
            if vals:
                metrics[key] = {
                    "mean": np.mean(vals),
                    "std": np.std(vals),
                    "count": len(vals)
                }

        summary[method] = metrics

    return summary


def rank_methods(summary, metric, lower_is_better=False):
    """Rank methods by a metric."""
    scores = []
    for method, metrics in summary.items():
        if metric in metrics:
            scores.append((method, metrics[metric]["mean"]))

    scores.sort(key=lambda x: x[1], reverse=not lower_is_better)
    ranks = {method: rank+1 for rank, (method, _) in enumerate(scores)}
    return ranks


def format_metric(mean, std, lower_is_better=False):
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


def generate_markdown_table(summary, methods, title, description):
    """Generate a markdown table for the given summary."""
    lines = []
    lines.append(f"## {title}")
    lines.append("")
    lines.append(description)
    lines.append("")

    # Metrics to display
    metrics_display = [
        ("auc", "AUC ↑", False),
        ("ap", "AP ↑", False),
        ("max_f1", "Max F1 ↑", False),
        ("accuracy", "Accuracy ↑", False),
        ("f1", "F1 ↑", False),
        ("precision", "Precision ↑", False),
        ("recall", "Recall ↑", False),
        ("ece", "ECE ↓", True),
        ("brier", "Brier ↓", True),
        ("anice", "ANICE ↓", True),
        ("oracle_ce", "Oracle CE ↓", True),
    ]

    # Find best/second-best for each metric
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
                formatted = format_metric(mean, std, lower_is_better)

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
    """Generate a ranking table showing wins and average rank."""
    lines = []
    lines.append(f"### {title}")
    lines.append("")

    method_wins = defaultdict(int)
    method_ranks = defaultdict(list)

    for metric, lower_is_better in metrics_list:
        ranks = rank_methods(summary, metric, lower_is_better)
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


def analyze_extreme_cases(results):
    """Analyze performance at extreme c and π values."""
    lines = []
    lines.append("## Extreme Case Analysis")
    lines.append("")
    lines.append("Performance at corner cases of the (c, π) grid.")
    lines.append("")

    cases = [
        ("Low c, Low π (c=0.01, π=0.01)", lambda v: v["c"] == 0.01 and v["pi"] == 0.01),
        ("Low c, High π (c=0.01, π=0.99)", lambda v: v["c"] == 0.01 and v["pi"] == 0.99),
        ("High c, Low π (c=0.99, π=0.01)", lambda v: v["c"] == 0.99 and v["pi"] == 0.01),
        ("High c, High π (c=0.99, π=0.99)", lambda v: v["c"] == 0.99 and v["pi"] == 0.99),
        ("Balanced (c=0.5, π=0.5)", lambda v: v["c"] == 0.5 and v["pi"] == 0.5),
    ]

    for case_name, filter_fn in cases:
        summary = compute_summary_stats(results, filter_fn)

        # Show top 3 methods by AUC
        auc_ranks = rank_methods(summary, "auc", lower_is_better=False)
        sorted_by_auc = sorted(auc_ranks.items(), key=lambda x: x[1])[:3]

        lines.append(f"### {case_name}")
        lines.append("")
        lines.append("Top 3 by AUC:")
        for method, rank in sorted_by_auc:
            label = METHOD_LABELS.get(method, method)
            auc_mean = summary[method]["auc"]["mean"]
            auc_std = summary[method]["auc"]["std"]
            lines.append(f"  {rank}. **{label}**: {auc_mean:.3f} ± {auc_std:.3f}")
        lines.append("")

    return "\n".join(lines)


def main():
    print("Loading Phase 3 results...")
    results = load_phase3_results()

    print(f"Loaded {sum(len(v['all']) for v in results.values())} total measurements")

    # Overall summary (all configurations)
    print("\nComputing overall statistics...")
    summary_all = compute_summary_stats(results)

    # Prior-specific summaries
    print("Computing prior-specific statistics...")
    summary_pi_low = compute_summary_stats(results, lambda v: v["pi"] < 0.5)
    summary_pi_high = compute_summary_stats(results, lambda v: v["pi"] >= 0.5)

    # Generate markdown output
    output = []

    output.append("# Phase 3 Results - Comprehensive Analysis (Seed 42)")
    output.append("")
    output.append("**Configuration:**")
    output.append("- **Datasets**: 7 (20News, Connect4, FashionMNIST, IMDB, MNIST, Mushrooms, Spambase)")
    output.append("- **Label frequency (c)**: 7 values [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]")
    output.append("- **True prior (π)**: 7 values [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]")
    output.append("- **Total configurations**: 7 × 7 × 7 = 343 per method")
    output.append("- **Methods**: 14 (2 base + 10 mean-prior variants + 2 oracles)")
    output.append("")
    output.append("---")
    output.append("")

    # Overall performance table
    output.append(generate_markdown_table(
        summary_all,
        METHOD_ORDER,
        "Overall Performance Across All Configurations",
        "*Mean ± Std across 343 runs (7 datasets × 7 c values × 7 π values). **Bold** = best, *italic* = second-best per metric.*"
    ))

    # Prior regime tables
    output.append(generate_markdown_table(
        summary_pi_low,
        METHOD_ORDER,
        "Performance on Low Priors (π < 0.5)",
        "*Mean ± Std across configurations with π ∈ {0.01, 0.1, 0.3}. **Bold** = best, *italic* = second-best per metric.*"
    ))

    output.append(generate_markdown_table(
        summary_pi_high,
        METHOD_ORDER,
        "Performance on High Priors (π ≥ 0.5)",
        "*Mean ± Std across configurations with π ∈ {0.5, 0.7, 0.9, 0.99}. **Bold** = best, *italic* = second-best per metric.*"
    ))

    # Rankings
    output.append("---")
    output.append("")
    output.append("## Method Rankings")
    output.append("")

    all_metrics = [
        ("auc", False), ("ap", False), ("max_f1", False),
        ("accuracy", False), ("f1", False), ("precision", False), ("recall", False),
        ("ece", True), ("brier", True), ("anice", True), ("oracle_ce", True)
    ]

    output.append(generate_ranking_table(summary_all, METHOD_ORDER, all_metrics,
                                        "Overall Performance (All Configurations)"))
    output.append(generate_ranking_table(summary_pi_low, METHOD_ORDER, all_metrics,
                                        "Performance on Low Priors (π < 0.5)"))
    output.append(generate_ranking_table(summary_pi_high, METHOD_ORDER, all_metrics,
                                        "Performance on High Priors (π ≥ 0.5)"))

    # Extreme cases
    output.append("---")
    output.append("")
    output.append(analyze_extreme_cases(results))

    # Key insights
    output.append("---")
    output.append("")
    output.append("## Key Insights")
    output.append("")

    # Find best constant priors for different regimes
    prior_variants = [m for m in METHOD_ORDER if "mean_prior" in m and m.endswith(("0.353", "0.5", "0.69", "1.0"))]

    # Rank constant priors for all π
    output.append("### Best Constant method_prior Values")
    output.append("")
    output.append("**For all priors (π ∈ [0.01, 0.99]):**")
    auc_ranks_all = rank_methods(summary_all, "auc", False)
    const_prior_ranks_all = [(m, auc_ranks_all[m]) for m in prior_variants if m in auc_ranks_all]
    const_prior_ranks_all.sort(key=lambda x: x[1])
    for i, (method, rank) in enumerate(const_prior_ranks_all[:3], 1):
        label = METHOD_LABELS[method]
        auc = summary_all[method]["auc"]["mean"]
        std = summary_all[method]["auc"]["std"]
        output.append(f"  {i}. **{label}**: Rank {rank}, AUC = {auc:.3f} ± {std:.3f}")
    output.append("")

    output.append("**For low priors (π < 0.5):**")
    auc_ranks_low = rank_methods(summary_pi_low, "auc", False)
    const_prior_ranks_low = [(m, auc_ranks_low[m]) for m in prior_variants if m in auc_ranks_low]
    const_prior_ranks_low.sort(key=lambda x: x[1])
    for i, (method, rank) in enumerate(const_prior_ranks_low[:3], 1):
        label = METHOD_LABELS[method]
        auc = summary_pi_low[method]["auc"]["mean"]
        std = summary_pi_low[method]["auc"]["std"]
        output.append(f"  {i}. **{label}**: Rank {rank}, AUC = {auc:.3f} ± {std:.3f}")
    output.append("")

    # Stability analysis
    output.append("### Stability Analysis")
    output.append("")
    output.append("Standard deviation of AUC across all 343 configurations (lower = more stable):")
    output.append("")

    stability_scores = []
    for method in METHOD_ORDER:
        if method in summary_all and "auc" in summary_all[method]:
            std = summary_all[method]["auc"]["std"]
            stability_scores.append((method, std))

    stability_scores.sort(key=lambda x: x[1])
    for i, (method, std) in enumerate(stability_scores[:5], 1):
        label = METHOD_LABELS[method]
        mean = summary_all[method]["auc"]["mean"]
        output.append(f"  {i}. **{label}**: {std:.4f} (mean AUC = {mean:.3f})")
    output.append("")

    # Write output
    output_file = Path("analysis/PHASE3_SEED42_COMPREHENSIVE_TABLE.md")
    with open(output_file, "w") as f:
        f.write("\n".join(output))

    print(f"\n✅ Comprehensive table saved to: {output_file}")


if __name__ == "__main__":
    main()
