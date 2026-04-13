#!/usr/bin/env python3
"""
Analyze Timing and Efficiency Metrics

Computes timing-based metrics from experiment results:
- Time to best epoch
- Average time per epoch
- Efficiency ratio (how early the model converged)
- Total convergence time

Usage:
    python analysis/analyze_timing_metrics.py \
        --results-dir results_phase1_extended \
        --output analysis/TIMING_ANALYSIS.md
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse
import statistics


def compute_timing_metrics(method_data):
    """Compute timing-based metrics from method results."""
    timing_metrics = {}

    # Extract timing info
    duration_sec = method_data.get("timing", {}).get("duration_seconds", None)
    global_epochs = method_data.get("global_epochs", None)
    best_epoch = method_data.get("best", {}).get("epoch", None)

    if duration_sec is not None and global_epochs is not None and best_epoch is not None:
        # Total convergence time (wall clock)
        timing_metrics["total_time"] = duration_sec

        # Average time per epoch
        timing_metrics["avg_time_per_epoch"] = duration_sec / global_epochs if global_epochs > 0 else None

        # Time to best epoch (when best validation performance was achieved)
        timing_metrics["time_to_best"] = (best_epoch / global_epochs) * duration_sec if global_epochs > 0 else None

        # Efficiency ratio (lower = converged earlier)
        timing_metrics["efficiency_ratio"] = best_epoch / global_epochs if global_epochs > 0 else None

        # Convergence speed (epochs)
        timing_metrics["epochs_to_best"] = best_epoch
        timing_metrics["total_epochs"] = global_epochs

    return timing_metrics


def load_results_with_timing(results_dir: Path):
    """Load results and compute timing metrics for each experiment."""
    results = defaultdict(lambda: defaultdict(list))

    for json_file in results_dir.glob("seed_*/*.json"):
        with open(json_file) as f:
            data = json.load(f)

        # Extract dataset name from filename
        filename = json_file.name
        dataset = filename.split("_case-control")[0]

        # Process each method
        for method_key, method_data in data.get("runs", {}).items():
            if "best" not in method_data:
                continue

            # Compute timing metrics
            timing = compute_timing_metrics(method_data)

            if timing:
                # Store all timing metrics
                results[method_key][dataset].append(timing)

    return results


def aggregate_timing_stats(results):
    """Aggregate timing statistics across experiments."""
    aggregated = {}

    for method, datasets in results.items():
        method_stats = {}

        # Aggregate across all datasets for this method
        all_times = []
        all_epochs = []
        all_efficiency = []
        all_time_per_epoch = []
        all_time_to_best = []

        for dataset, experiments in datasets.items():
            for exp in experiments:
                if exp.get("total_time") is not None:
                    all_times.append(exp["total_time"])
                if exp.get("epochs_to_best") is not None:
                    all_epochs.append(exp["epochs_to_best"])
                if exp.get("efficiency_ratio") is not None:
                    all_efficiency.append(exp["efficiency_ratio"])
                if exp.get("avg_time_per_epoch") is not None:
                    all_time_per_epoch.append(exp["avg_time_per_epoch"])
                if exp.get("time_to_best") is not None:
                    all_time_to_best.append(exp["time_to_best"])

        if all_times:
            method_stats = {
                "total_time_mean": np.mean(all_times),
                "total_time_std": np.std(all_times),
                "epochs_to_best_mean": np.mean(all_epochs) if all_epochs else None,
                "epochs_to_best_std": np.std(all_epochs) if all_epochs else None,
                "efficiency_ratio_mean": np.mean(all_efficiency) if all_efficiency else None,
                "efficiency_ratio_std": np.std(all_efficiency) if all_efficiency else None,
                "time_per_epoch_mean": np.mean(all_time_per_epoch) if all_time_per_epoch else None,
                "time_per_epoch_std": np.std(all_time_per_epoch) if all_time_per_epoch else None,
                "time_to_best_mean": np.mean(all_time_to_best) if all_time_to_best else None,
                "time_to_best_std": np.std(all_time_to_best) if all_time_to_best else None,
                "num_experiments": len(all_times),
            }
            aggregated[method] = method_stats

    return aggregated


def generate_timing_report(aggregated, output_file: Path):
    """Generate markdown report with timing analysis."""

    # Sort methods by total time
    sorted_methods = sorted(
        aggregated.items(),
        key=lambda x: x[1]["total_time_mean"]
    )

    with open(output_file, "w") as f:
        f.write("# Timing and Efficiency Analysis\n\n")
        f.write("Analysis of computational efficiency metrics across all methods.\n\n")

        # Total convergence time
        f.write("## 1. Total Convergence Time (Wall Clock)\n\n")
        f.write("Average time to complete training (including early stopping).\n\n")
        f.write("| Method | Mean Time (s) | Std Dev (s) | Mean Time (min) | Experiments |\n")
        f.write("|--------|---------------|-------------|-----------------|-------------|\n")

        for method, stats in sorted_methods:
            mean_sec = stats["total_time_mean"]
            std_sec = stats["total_time_std"]
            mean_min = mean_sec / 60
            num_exp = stats["num_experiments"]
            f.write(f"| {method:<30} | {mean_sec:>8.1f} | {std_sec:>8.1f} | {mean_min:>10.1f} | {num_exp:>11} |\n")

        f.write("\n")

        # Time per epoch
        f.write("## 2. Average Time per Epoch\n\n")
        f.write("Computational efficiency: average seconds per training epoch.\n\n")
        f.write("| Method | Mean (s/epoch) | Std Dev | Experiments |\n")
        f.write("|--------|----------------|---------|-------------|\n")

        sorted_by_epoch = sorted(
            [(m, s) for m, s in aggregated.items() if s.get("time_per_epoch_mean")],
            key=lambda x: x[1]["time_per_epoch_mean"]
        )

        for method, stats in sorted_by_epoch:
            mean_tpe = stats["time_per_epoch_mean"]
            std_tpe = stats["time_per_epoch_std"]
            num_exp = stats["num_experiments"]
            f.write(f"| {method:<30} | {mean_tpe:>12.2f} | {std_tpe:>7.2f} | {num_exp:>11} |\n")

        f.write("\n")

        # Convergence speed (epochs)
        f.write("## 3. Convergence Speed (Epochs to Best)\n\n")
        f.write("How many epochs needed to reach best validation performance.\n\n")
        f.write("| Method | Mean Epochs | Std Dev | Experiments |\n")
        f.write("|--------|-------------|---------|-------------|\n")

        sorted_by_conv = sorted(
            [(m, s) for m, s in aggregated.items() if s.get("epochs_to_best_mean")],
            key=lambda x: x[1]["epochs_to_best_mean"]
        )

        for method, stats in sorted_by_conv:
            mean_epochs = stats["epochs_to_best_mean"]
            std_epochs = stats["epochs_to_best_std"]
            num_exp = stats["num_experiments"]
            f.write(f"| {method:<30} | {mean_epochs:>11.1f} | {std_epochs:>7.1f} | {num_exp:>11} |\n")

        f.write("\n")

        # Efficiency ratio
        f.write("## 4. Efficiency Ratio\n\n")
        f.write("Ratio of best_epoch / total_epochs. Lower = converged earlier (more efficient early stopping).\n\n")
        f.write("| Method | Mean Ratio | Std Dev | Interpretation | Experiments |\n")
        f.write("|--------|------------|---------|----------------|-------------|\n")

        sorted_by_eff = sorted(
            [(m, s) for m, s in aggregated.items() if s.get("efficiency_ratio_mean")],
            key=lambda x: x[1]["efficiency_ratio_mean"]
        )

        for method, stats in sorted_by_eff:
            mean_eff = stats["efficiency_ratio_mean"]
            std_eff = stats["efficiency_ratio_std"]
            num_exp = stats["num_experiments"]

            # Interpretation
            if mean_eff < 0.3:
                interp = "Very Early"
            elif mean_eff < 0.5:
                interp = "Early"
            elif mean_eff < 0.7:
                interp = "Moderate"
            elif mean_eff < 0.9:
                interp = "Late"
            else:
                interp = "Very Late"

            f.write(f"| {method:<30} | {mean_eff:>10.3f} | {std_eff:>7.3f} | {interp:<14} | {num_exp:>11} |\n")

        f.write("\n")

        # Time to best epoch
        f.write("## 5. Time to Best Epoch\n\n")
        f.write("Wall clock time when best validation performance was achieved.\n\n")
        f.write("| Method | Mean Time (s) | Std Dev (s) | Mean Time (min) | Experiments |\n")
        f.write("|--------|---------------|-------------|-----------------|-------------|\n")

        sorted_by_ttb = sorted(
            [(m, s) for m, s in aggregated.items() if s.get("time_to_best_mean")],
            key=lambda x: x[1]["time_to_best_mean"]
        )

        for method, stats in sorted_by_ttb:
            mean_ttb = stats["time_to_best_mean"]
            std_ttb = stats["time_to_best_std"]
            mean_min = mean_ttb / 60
            num_exp = stats["num_experiments"]
            f.write(f"| {method:<30} | {mean_ttb:>8.1f} | {std_ttb:>8.1f} | {mean_min:>10.1f} | {num_exp:>11} |\n")

        f.write("\n")

        # Summary
        f.write("## Summary\n\n")

        # Fastest methods
        f.write("### Fastest Methods (by total convergence time)\n\n")
        for i, (method, stats) in enumerate(sorted_methods[:5]):
            mean_min = stats["total_time_mean"] / 60
            f.write(f"{i+1}. **{method}**: {mean_min:.1f} minutes\n")

        f.write("\n### Slowest Methods (by total convergence time)\n\n")
        for i, (method, stats) in enumerate(reversed(sorted_methods[-5:])):
            mean_min = stats["total_time_mean"] / 60
            f.write(f"{i+1}. **{method}**: {mean_min:.1f} minutes\n")

        f.write("\n### Most Efficient Convergers (earliest best epoch)\n\n")
        for i, (method, stats) in enumerate(sorted_by_eff[:5]):
            ratio = stats["efficiency_ratio_mean"]
            f.write(f"{i+1}. **{method}**: {ratio:.3f} (converges at {ratio*100:.1f}% of training)\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze timing and efficiency metrics from experiment results"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results_phase1_extended"),
        help="Directory containing experiment results"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/TIMING_ANALYSIS.md"),
        help="Output markdown file"
    )
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"ERROR: Results directory not found: {args.results_dir}")
        return 1

    print(f"Loading results from: {args.results_dir}")
    results = load_results_with_timing(args.results_dir)

    print(f"Aggregating timing statistics...")
    aggregated = aggregate_timing_stats(results)

    print(f"Generating report: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate_timing_report(aggregated, args.output)

    print(f"\nTiming analysis complete!")
    print(f"Methods analyzed: {len(aggregated)}")
    print(f"Report saved to: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
