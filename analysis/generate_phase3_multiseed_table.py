#!/usr/bin/env python3
"""
Generate comprehensive Phase 3 multi-seed results table.

Analyzes performance across:
- All 5 seeds (42, 456, 789, 1024, 2048)
- 7 datasets
- 7 label frequencies (c)
- 7 true priors (π)
- 16 method variants (14 actual + 2 adaptive virtual methods)

Adaptive methods:
- VPU-nomix-MP(0.353;0.69): Uses 0.353 when π < 0.5, else 0.69
- VPU-MP(0.353;0.69): Uses 0.353 when π < 0.5, else 0.69

Organizes results by dataset and hyperparameter regions.
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("results_phase3")
DATASETS = ["20News", "Connect4", "FashionMNIST", "IMDB", "MNIST", "Mushrooms", "Spambase"]
SEEDS = [42, 456, 789, 1024, 2048]

METHOD_ORDER = [
    # Base VPU methods
    "vpu", "vpu_nomixup",
    # VPU-nomix with mean-prior
    "vpu_nomixup_mean_prior_auto",
    "vpu_nomixup_mean_prior_0.353",
    "vpu_nomixup_mean_prior_0.5",
    "vpu_nomixup_mean_prior_0.69",
    "vpu_nomixup_mean_prior_1",
    "vpu_nomixup_mean_prior_0.353;0.69",  # Adaptive: 0.353 if π<0.5, else 0.69
    # VPU with mean-prior
    "vpu_mean_prior_auto",
    "vpu_mean_prior_0.353",
    "vpu_mean_prior_0.5",
    "vpu_mean_prior_0.69",
    "vpu_mean_prior_1",
    "vpu_mean_prior_0.353;0.69",  # Adaptive: 0.353 if π<0.5, else 0.69
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
    "vpu_mean_prior_1": "VPU-MP(1.0)",
    "vpu_mean_prior_0.353;0.69": "VPU-MP(0.353;0.69)",
    "vpu_nomixup_mean_prior_auto": "VPU-nomix-MP(auto)",
    "vpu_nomixup_mean_prior_0.353": "VPU-nomix-MP(0.353)",
    "vpu_nomixup_mean_prior_0.5": "VPU-nomix-MP(0.5)",
    "vpu_nomixup_mean_prior_0.69": "VPU-nomix-MP(0.69)",
    "vpu_nomixup_mean_prior_1": "VPU-nomix-MP(1.0)",
    "vpu_nomixup_mean_prior_0.353;0.69": "VPU-nomix-MP(0.353;0.69)",
    "oracle_bce": "Oracle-PN",
    "pn_naive": "PN-Naive",
}


def parse_filename(filename):
    """Extract dataset, c, and π from filename."""
    parts = filename.stem.split("_")
    dataset = parts[0]

    # Extract c value
    c_str = [p for p in parts if p.startswith("c") and len(p) > 1 and p[1].isdigit()][0]
    c = float(c_str[1:])

    # Extract π (true prior) value
    pi_str = [p for p in parts if p.startswith("trueprior")][0]
    pi = float(pi_str[9:])

    return dataset, c, pi


def load_phase3_results():
    """Load all Phase 3 results across seeds."""
    results = defaultdict(lambda: defaultdict(list))

    for seed in SEEDS:
        seed_dir = RESULTS_DIR / f"seed_{seed}"
        if not seed_dir.exists():
            continue

        for json_file in seed_dir.glob("*.json"):
            # Parse metadata from filename
            dataset, c, pi = parse_filename(json_file)

            with open(json_file) as f:
                data = json.load(f)

            for method_key, method_data in data.get("runs", {}).items():
                if "best" not in method_data or "metrics" not in method_data["best"]:
                    continue

                metrics = method_data["best"]["metrics"]

                # Determine full method identifier
                if method_key in ["vpu_mean_prior", "vpu_nomixup_mean_prior"]:
                    if "methodprior0.353" in json_file.name:
                        method_id = f"{method_key}_0.353"
                    elif "methodprior0.5" in json_file.name:
                        method_id = f"{method_key}_0.5"
                    elif "methodprior0.69" in json_file.name:
                        method_id = f"{method_key}_0.69"
                    elif "methodprior1" in json_file.name:
                        method_id = f"{method_key}_1"
                    else:
                        method_id = f"{method_key}_auto"
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
                result["seed"] = seed

                results[method_id]["all"].append(result)

    return results


def create_adaptive_methods(results):
    """Create virtual adaptive methods that switch based on prior regime.

    VPU-nomix-MP(0.353;0.69): Uses 0.353 when π < 0.5, else 0.69
    VPU-MP(0.353;0.69): Uses 0.353 when π < 0.5, else 0.69
    """
    # VPU-nomix-MP(0.353;0.69)
    adaptive_nomix = []
    for result in results["vpu_nomixup_mean_prior_0.353"]["all"]:
        if result.get("pi") is not None and result["pi"] < 0.5:
            adaptive_nomix.append(result)
    for result in results["vpu_nomixup_mean_prior_0.69"]["all"]:
        if result.get("pi") is not None and result["pi"] >= 0.5:
            adaptive_nomix.append(result)

    results["vpu_nomixup_mean_prior_0.353;0.69"] = {"all": adaptive_nomix}

    # VPU-MP(0.353;0.69)
    adaptive_mixup = []
    for result in results["vpu_mean_prior_0.353"]["all"]:
        if result.get("pi") is not None and result["pi"] < 0.5:
            adaptive_mixup.append(result)
    for result in results["vpu_mean_prior_0.69"]["all"]:
        if result.get("pi") is not None and result["pi"] >= 0.5:
            adaptive_mixup.append(result)

    results["vpu_mean_prior_0.353;0.69"] = {"all": adaptive_mixup}

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
    """Generate a single comparison table."""
    lines = []
    lines.append(f"### {title}")
    lines.append("")
    lines.append(description)
    lines.append("")

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
    """Generate a ranking table showing wins and average rank."""
    lines = []
    lines.append(f"#### {title}")
    lines.append("")

    method_wins = defaultdict(int)
    method_ranks = defaultdict(list)

    for metric, lower_is_better in metrics_list:
        # Rank methods
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


def generate_category_rankings(summary, methods, title_prefix=""):
    """Generate ranking tables by metric category."""
    lines = []

    # Define metric categories
    metric_categories = {
        "Threshold-Invariant Metrics": [
            ("auc", False), ("ap", False), ("max_f1", False)
        ],
        "Threshold-Dependent Metrics": [
            ("accuracy", False), ("f1", False), ("precision", False), ("recall", False)
        ],
        "Calibration Metrics": [
            ("ece", True), ("brier", True), ("anice", True)
        ],
        "Cross-Entropy": [
            ("oracle_ce", True)
        ],
    }

    for category_name, category_metrics in metric_categories.items():
        category_title = f"{title_prefix}{category_name}" if title_prefix else category_name
        lines.append(generate_ranking_table(summary, methods, category_metrics, category_title))

    return "\n".join(lines)


def main():
    print("Loading Phase 3 multi-seed results...", file=sys.stderr)
    results = load_phase3_results()

    print("Creating adaptive methods...", file=sys.stderr)
    results = create_adaptive_methods(results)

    total_runs = sum(len(v['all']) for v in results.values())
    print(f"Loaded {total_runs} total method runs", file=sys.stderr)

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

    all_metrics = [
        ("auc", False), ("ap", False), ("max_f1", False),
        ("accuracy", False), ("f1", False), ("precision", False), ("recall", False),
        ("ece", True), ("brier", True), ("anice", True), ("oracle_ce", True)
    ]

    output = []

    # Header
    output.append("# Phase 3 Multi-Seed Results - Comprehensive Analysis")
    output.append("")
    output.append("**Configuration:**")
    output.append("- **Datasets**: 7 (20News, Connect4, FashionMNIST, IMDB, MNIST, Mushrooms, Spambase)")
    output.append("- **Seeds**: 5 [42, 456, 789, 1024, 2048]")
    output.append("- **Label frequency (c)**: 7 values [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]")
    output.append("- **True prior (π)**: 7 values [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]")
    output.append("- **Total configurations per seed**: 7 × 7 × 7 = 343")
    output.append("- **Methods**: 16 (2 base + 10 mean-prior variants + 2 adaptive variants + 2 oracles)")
    output.append("- **Total method runs**: 24,010 (343 configs × 14 actual methods × 5 seeds; 2 adaptive methods are virtual)")
    output.append("")
    output.append("---")
    output.append("")

    # Overall performance
    print("Computing overall statistics...", file=sys.stderr)
    summary_all = compute_summary_stats(results)

    output.append("## Overall Performance (All Configurations)")
    output.append("")
    output.append("*Mean ± Std across all configurations (7 datasets × 7 c × 7 π × 5 seeds). Actual methods: 8,575 runs/method. Adaptive methods: virtual selection from constituent methods. **Bold** = best, *italic* = second-best per metric.*")
    output.append("")
    output.append(generate_table(summary_all, METHOD_ORDER, "", "", metrics_display))

    # Performance by prior regime
    print("Computing prior-specific statistics...", file=sys.stderr)

    output.append("## Performance by Prior Regime")
    output.append("")

    summary_pi_low = compute_summary_stats(results, lambda v: v.get("pi") is not None and v["pi"] < 0.5)
    output.append(generate_table(
        summary_pi_low, METHOD_ORDER,
        "Low Priors (π < 0.5)",
        "*Mean ± Std across configurations with π ∈ {0.01, 0.1, 0.3}. **Bold** = best, *italic* = second-best per metric.*",
        metrics_display
    ))

    summary_pi_high = compute_summary_stats(results, lambda v: v.get("pi") is not None and v["pi"] >= 0.5)
    output.append(generate_table(
        summary_pi_high, METHOD_ORDER,
        "High Priors (π ≥ 0.5)",
        "*Mean ± Std across configurations with π ∈ {0.5, 0.7, 0.9, 0.99}. **Bold** = best, *italic* = second-best per metric.*",
        metrics_display
    ))

    # Performance by label frequency regime
    output.append("## Performance by Label Frequency Regime")
    output.append("")

    summary_c_low = compute_summary_stats(results, lambda v: v.get("c") is not None and v["c"] < 0.5)
    output.append(generate_table(
        summary_c_low, METHOD_ORDER,
        "Low Label Frequency (c < 0.5)",
        "*Mean ± Std across configurations with c ∈ {0.01, 0.1, 0.3}. **Bold** = best, *italic* = second-best per metric.*",
        metrics_display
    ))

    summary_c_high = compute_summary_stats(results, lambda v: v.get("c") is not None and v["c"] >= 0.5)
    output.append(generate_table(
        summary_c_high, METHOD_ORDER,
        "High Label Frequency (c ≥ 0.5)",
        "*Mean ± Std across configurations with c ∈ {0.5, 0.7, 0.9, 0.99}. **Bold** = best, *italic* = second-best per metric.*",
        metrics_display
    ))

    # Performance at extreme priors
    output.append("## Performance at Extreme Priors")
    output.append("")

    summary_pi_extreme = compute_summary_stats(results, lambda v: v.get("pi") is not None and v["pi"] in [0.01, 0.99])
    output.append(generate_table(
        summary_pi_extreme, METHOD_ORDER,
        "Extreme Priors (π ∈ {0.01, 0.99})",
        "*Mean ± Std across configurations with extremely low or high priors. **Bold** = best, *italic* = second-best per metric.*",
        metrics_display
    ))

    # Performance at extreme label frequencies
    output.append("## Performance at Extreme Label Frequencies")
    output.append("")

    summary_c_extreme = compute_summary_stats(results, lambda v: v.get("c") is not None and v["c"] in [0.01, 0.99])
    output.append(generate_table(
        summary_c_extreme, METHOD_ORDER,
        "Extreme Label Frequencies (c ∈ {0.01, 0.99})",
        "*Mean ± Std across configurations with extremely low or high label frequencies. **Bold** = best, *italic* = second-best per metric.*",
        metrics_display
    ))

    # Extreme corner cases
    output.append("## Extreme Corner Cases")
    output.append("")

    corners = [
        ("Low c, Low π (c ≤ 0.1, π ≤ 0.1)", lambda v: v.get("c") is not None and v.get("pi") is not None and v["c"] <= 0.1 and v["pi"] <= 0.1),
        ("Low c, High π (c ≤ 0.1, π ≥ 0.9)", lambda v: v.get("c") is not None and v.get("pi") is not None and v["c"] <= 0.1 and v["pi"] >= 0.9),
        ("High c, Low π (c ≥ 0.9, π ≤ 0.1)", lambda v: v.get("c") is not None and v.get("pi") is not None and v["c"] >= 0.9 and v["pi"] <= 0.1),
        ("High c, High π (c ≥ 0.9, π ≥ 0.9)", lambda v: v.get("c") is not None and v.get("pi") is not None and v["c"] >= 0.9 and v["pi"] >= 0.9),
        ("Balanced (c = 0.5, π = 0.5)", lambda v: v.get("c") is not None and v.get("pi") is not None and v["c"] == 0.5 and v["pi"] == 0.5),
    ]

    for corner_name, filter_fn in corners:
        summary = compute_summary_stats(results, filter_fn)
        output.append(generate_table(
            summary, METHOD_ORDER,
            corner_name,
            f"*Performance in extreme regime. **Bold** = best, *italic* = second-best per metric.*",
            metrics_display
        ))

    # Per-dataset breakdowns
    output.append("---")
    output.append("")
    output.append("## Per-Dataset Analysis")
    output.append("")

    for dataset in DATASETS:
        print(f"Processing dataset: {dataset}", file=sys.stderr)

        output.append(f"### {dataset}")
        output.append("")

        # Overall for this dataset
        summary_ds = compute_summary_stats(results, lambda v: v["dataset"] == dataset)
        output.append(generate_table(
            summary_ds, METHOD_ORDER,
            "All Configurations",
            f"*Mean ± Std across 1,225 runs per method (7 c × 7 π × 5 seeds). **Bold** = best, *italic* = second-best per metric.*",
            metrics_display
        ))

        # By prior for this dataset
        summary_ds_pi_low = compute_summary_stats(results, lambda v: v["dataset"] == dataset and v.get("pi") is not None and v["pi"] < 0.5)
        output.append(generate_table(
            summary_ds_pi_low, METHOD_ORDER,
            "Low Priors (π < 0.5)",
            "*Low prior regime for this dataset. **Bold** = best, *italic* = second-best per metric.*",
            metrics_display
        ))

        summary_ds_pi_high = compute_summary_stats(results, lambda v: v["dataset"] == dataset and v.get("pi") is not None and v["pi"] >= 0.5)
        output.append(generate_table(
            summary_ds_pi_high, METHOD_ORDER,
            "High Priors (π ≥ 0.5)",
            "*High prior regime for this dataset. **Bold** = best, *italic* = second-best per metric.*",
            metrics_display
        ))

    # Rankings
    output.append("---")
    output.append("")
    output.append("## Method Rankings")
    output.append("")

    # Overall rankings
    output.append("### Overall Performance")
    output.append("")
    output.append(generate_ranking_table(summary_all, METHOD_ORDER, all_metrics,
                                        "All Metrics Combined"))

    # Category-specific rankings for overall
    output.append("### By Metric Category - Overall")
    output.append("")
    output.append(generate_category_rankings(summary_all, METHOD_ORDER))

    # Prior regime rankings
    output.append("### By Prior Regime")
    output.append("")
    output.append(generate_ranking_table(summary_pi_low, METHOD_ORDER, all_metrics,
                                        "Low Priors (π < 0.5) - All Metrics"))
    output.append(generate_ranking_table(summary_pi_high, METHOD_ORDER, all_metrics,
                                        "High Priors (π ≥ 0.5) - All Metrics"))

    # Label frequency regime rankings
    output.append("### By Label Frequency Regime")
    output.append("")
    output.append(generate_ranking_table(summary_c_low, METHOD_ORDER, all_metrics,
                                        "Low Label Frequency (c < 0.5) - All Metrics"))
    output.append(generate_ranking_table(summary_c_high, METHOD_ORDER, all_metrics,
                                        "High Label Frequency (c ≥ 0.5) - All Metrics"))

    # Extreme regime rankings
    output.append("### By Extreme Regimes")
    output.append("")
    output.append(generate_ranking_table(summary_pi_extreme, METHOD_ORDER, all_metrics,
                                        "Extreme Priors (π ∈ {0.01, 0.99}) - All Metrics"))
    output.append(generate_ranking_table(summary_c_extreme, METHOD_ORDER, all_metrics,
                                        "Extreme Label Frequencies (c ∈ {0.01, 0.99}) - All Metrics"))

    # Key insights
    output.append("---")
    output.append("")
    output.append("## Key Insights")
    output.append("")

    # Best constant priors
    output.append("### Best Constant method_prior Values")
    output.append("")

    prior_variants = [m for m in METHOD_ORDER if "mean_prior" in m and
                     m.endswith(("0.353", "0.5", "0.69", "1"))]

    output.append("**Overall (all c, π):**")
    auc_scores = [(summary_all[m]["auc"]["mean"], m) for m in prior_variants if m in summary_all and "auc" in summary_all[m]]
    auc_scores.sort(reverse=True)
    for i, (auc, method) in enumerate(auc_scores[:3], 1):
        label = METHOD_LABELS[method]
        std = summary_all[method]["auc"]["std"]
        output.append(f"  {i}. **{label}**: AUC = {auc:.3f} ± {std:.3f}")
    output.append("")

    output.append("**Low priors (π < 0.5):**")
    auc_scores_low = [(summary_pi_low[m]["auc"]["mean"], m) for m in prior_variants if m in summary_pi_low and "auc" in summary_pi_low[m]]
    auc_scores_low.sort(reverse=True)
    for i, (auc, method) in enumerate(auc_scores_low[:3], 1):
        label = METHOD_LABELS[method]
        std = summary_pi_low[method]["auc"]["std"]
        output.append(f"  {i}. **{label}**: AUC = {auc:.3f} ± {std:.3f}")
    output.append("")

    output.append("**High priors (π ≥ 0.5):**")
    auc_scores_high = [(summary_pi_high[m]["auc"]["mean"], m) for m in prior_variants if m in summary_pi_high and "auc" in summary_pi_high[m]]
    auc_scores_high.sort(reverse=True)
    for i, (auc, method) in enumerate(auc_scores_high[:3], 1):
        label = METHOD_LABELS[method]
        std = summary_pi_high[method]["auc"]["std"]
        output.append(f"  {i}. **{label}**: AUC = {auc:.3f} ± {std:.3f}")
    output.append("")

    # Stability analysis
    output.append("### Stability Analysis")
    output.append("")
    output.append("Standard deviation of AUC across all configurations (lower = more stable):")
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
        output.append(f"  {i}. **{label}**: σ = {std:.4f} (mean AUC = {mean:.3f})")
    output.append("")

    # Write output
    output_file = Path("analysis/PHASE3_MULTISEED_COMPREHENSIVE_TABLE.md")
    with open(output_file, "w") as f:
        f.write("\n".join(output))

    print(f"\n✅ Comprehensive table saved to: {output_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
