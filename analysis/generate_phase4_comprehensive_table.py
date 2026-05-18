#!/usr/bin/env python3
"""Generate Phase 4 comprehensive table: method_prior (rows) × methods (columns)."""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class MetricStats:
    """Statistics for a single metric."""
    mean: float
    std: float
    count: int


def parse_experiment_name(exp_name: str) -> Optional[Dict]:
    """Parse experiment name to extract parameters.

    Example: MNIST_case-control_random_c0.01_seed42_trueprior0.5_methodprior0.6667
    """
    parts = exp_name.split('_')

    # Find dataset (everything before case-control)
    dataset_idx = parts.index('case-control') if 'case-control' in parts else -1
    if dataset_idx < 0:
        return None

    dataset = '_'.join(parts[:dataset_idx])

    # Extract parameters
    c = None
    seed = None
    true_prior = None
    method_prior = None

    for part in parts[dataset_idx:]:
        if part.startswith('c'):
            try:
                c = float(part[1:])
            except ValueError:
                pass
        elif part.startswith('seed'):
            try:
                seed = int(part[4:])
            except ValueError:
                pass
        elif part.startswith('trueprior'):
            try:
                true_prior = float(part[9:])
            except ValueError:
                pass
        elif part.startswith('methodprior'):
            val = part[11:]
            if val == 'auto':
                method_prior = 'auto'
            elif val == 'true':
                method_prior = 'true'
            elif val == 'eplinear':
                method_prior = 'ep_linear'
            else:
                try:
                    method_prior = float(val)
                except ValueError:
                    pass

    if c is None or seed is None or true_prior is None or method_prior is None:
        return None

    return {
        'dataset': dataset,
        'c': c,
        'seed': seed,
        'true_prior': true_prior,
        'method_prior': method_prior
    }


def load_phase4_results(results_dir: Path) -> Dict:
    """Load all Phase 4 results from JSON files.

    Returns:
        Dict mapping (method, method_prior) -> list of metric dicts
    """
    results = defaultdict(list)

    for json_file in results_dir.glob('*.json'):
        with open(json_file, 'r') as f:
            data = json.load(f)

        exp_name = data.get('experiment', '')
        params = parse_experiment_name(exp_name)

        if params is None:
            continue

        # Extract method_prior from experiment name
        method_prior = params['method_prior']

        # Process each method's results
        for method, method_data in data.get('runs', {}).items():
            if 'best' not in method_data:
                continue

            metrics = method_data['best'].get('metrics', {})
            if not metrics:
                continue

            # Add context to metrics
            result = {
                **metrics,
                'dataset': params['dataset'],
                'c': params['c'],
                'true_prior': params['true_prior'],
                'seed': params['seed']
            }

            results[(method, method_prior)].append(result)

    return results


def compute_stats(values: List[float]) -> MetricStats:
    """Compute mean, std, count from list of values (ignoring NaN)."""
    clean_values = [v for v in values if not np.isnan(v)]

    if not clean_values:
        return MetricStats(mean=float('nan'), std=float('nan'), count=0)

    return MetricStats(
        mean=float(np.mean(clean_values)),
        std=float(np.std(clean_values)),
        count=len(clean_values)
    )


def aggregate_results(results: Dict) -> Dict:
    """Aggregate results by (method, method_prior).

    Returns:
        Dict mapping (method, method_prior) -> Dict[metric_name -> MetricStats]
    """
    aggregated = {}

    for (method, method_prior), runs in results.items():
        metrics_dict = defaultdict(list)

        for run in runs:
            for metric_name, value in run.items():
                if metric_name in ['dataset', 'c', 'true_prior', 'seed']:
                    continue
                metrics_dict[metric_name].append(value)

        aggregated[(method, method_prior)] = {
            metric_name: compute_stats(values)
            for metric_name, values in metrics_dict.items()
        }

    return aggregated


def format_value(stats: MetricStats, precision: int = 3) -> str:
    """Format a metric value as 'mean ± std'."""
    if np.isnan(stats.mean):
        return "—"
    return f"{stats.mean:.{precision}f} ± {stats.std:.{precision}f}"


def generate_table(aggregated: Dict, metric: str, methods: List[str],
                   method_priors: List, higher_is_better: bool = True) -> str:
    """Generate markdown table with method_prior as rows, methods as columns.

    Args:
        aggregated: Dict mapping (method, method_prior) -> metrics
        metric: Metric name to display
        methods: List of method names (columns)
        method_priors: List of method_prior values (rows)
        higher_is_better: Whether higher values are better (for bolding)
    """
    # Build table data
    table_data = {}
    for method_prior in method_priors:
        table_data[method_prior] = {}
        for method in methods:
            stats = aggregated.get((method, method_prior), {}).get(metric)
            if stats:
                table_data[method_prior][method] = stats
            else:
                table_data[method_prior][method] = MetricStats(float('nan'), float('nan'), 0)

    # Find best method_prior for each method (column)
    best_by_method = {}
    for method in methods:
        best_mean = float('-inf') if higher_is_better else float('inf')
        best_prior = None

        for method_prior in method_priors:
            stats = table_data[method_prior][method]
            if np.isnan(stats.mean):
                continue

            is_better = (stats.mean > best_mean) if higher_is_better else (stats.mean < best_mean)
            if is_better:
                best_mean = stats.mean
                best_prior = method_prior

        best_by_method[method] = best_prior

    # Build markdown table
    lines = []

    # Header
    header = "| method_prior | " + " | ".join(methods) + " |"
    lines.append(header)

    # Separator
    sep = "|" + "|".join(["---"] * (len(methods) + 1)) + "|"
    lines.append(sep)

    # Rows
    for method_prior in method_priors:
        # Format method_prior for display
        if method_prior == 'auto':
            prior_str = "auto"
        elif method_prior == 'true':
            prior_str = "true"
        else:
            prior_str = f"{method_prior:.4f}" if isinstance(method_prior, float) else str(method_prior)

        row = f"| {prior_str} |"

        for method in methods:
            stats = table_data[method_prior][method]
            value_str = format_value(stats)

            # Bold if this is the best method_prior for this method
            if best_by_method[method] == method_prior and not np.isnan(stats.mean):
                value_str = f"**{value_str}**"

            row += f" {value_str} |"

        lines.append(row)

    return "\n".join(lines)


def load_phase4_results_multiseed(results_base: Path, methods_filter=None) -> Dict:
    """Load Phase 4 results from all seed directories.

    Args:
        results_base: Base directory (e.g., results_phase4/)
        methods_filter: Optional set of method names to include
    """
    results = defaultdict(list)

    for seed_dir in sorted(results_base.glob('seed_*')):
        for json_file in seed_dir.glob('*.json'):
            with open(json_file, 'r') as f:
                data = json.load(f)

            exp_name = data.get('experiment', '')
            params = parse_experiment_name(exp_name)
            if params is None:
                continue

            method_prior = params['method_prior']

            for method, method_data in data.get('runs', {}).items():
                if methods_filter and method not in methods_filter:
                    continue
                if 'best' not in method_data:
                    continue
                metrics = method_data['best'].get('metrics', {})
                if not metrics:
                    continue

                result = {
                    **metrics,
                    'dataset': params['dataset'],
                    'c': params['c'],
                    'true_prior': params['true_prior'],
                    'seed': params['seed']
                }
                results[(method, method_prior)].append(result)

    return results


def main():
    """Generate Phase 4 comprehensive table."""
    results_base = Path("results_phase4")
    seeds = sorted([d.name for d in results_base.glob('seed_*')])
    num_seeds = len(seeds)

    vpu_methods = {'vpu_mean_prior', 'vpu_nomixup_mean_prior'}

    print(f"Loading Phase 4 results from {num_seeds} seeds...")
    results = load_phase4_results_multiseed(results_base, methods_filter=vpu_methods)
    print(f"Loaded {len(results)} (method, method_prior) combinations")

    print("Aggregating results...")
    aggregated = aggregate_results(results)

    methods = ['vpu_mean_prior', 'vpu_nomixup_mean_prior']

    method_priors = [
        'auto', 'true', 'ep_linear',
        0.01, 0.1111, 0.2222, 0.3333, 0.4444,
        0.5, 0.5556, 0.6667, 0.7778, 0.8889, 0.99
    ]

    expected_per_combo = 7 * 3 * 7 * num_seeds

    print(f"Generating tables...")

    # Also generate seed-42-only for the full 10-method table
    print("Loading seed-42 results for 10-method table...")
    results_seed42 = load_phase4_results(Path("results_phase4/seed_42"))
    aggregated_seed42 = aggregate_results(results_seed42)

    all_methods = [
        'nnpu', 'nnpusb', 'lbe', 'distpu', 'selfpu',
        'p3mixe', 'p3mixc', 'robustpu',
        'vpu_mean_prior', 'vpu_nomixup_mean_prior'
    ]

    # --- Multi-seed VPU table ---
    output_file = Path("analysis/PHASE4_MULTISEED_COMPREHENSIVE_TABLE.md")
    lines = []
    lines.append("# Phase 4 Results - VPU Method Prior Robustness (Multi-Seed)")
    lines.append("")
    lines.append("**Configuration:**")
    lines.append("- **Datasets**: 7 (20News, Connect4, FashionMNIST, IMDB, MNIST, Mushrooms, Spambase)")
    lines.append(f"- **Seeds**: {num_seeds} {seeds}")
    lines.append("- **Label frequency (c)**: 3 values [0.01, 0.5, 0.99]")
    lines.append("- **True prior (π)**: 7 values [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]")
    lines.append("- **Method_prior values**: 13 values [auto, true, 0.01, 0.1111, 0.2222, 0.3333, 0.4444, 0.5, 0.5556, 0.6667, 0.7778, 0.8889, 0.99]")
    lines.append(f"- **Total configurations per (method, method_prior)**: {expected_per_combo}")
    lines.append("- **Methods**: 2 VPU variants (vpu_mean_prior, vpu_nomixup_mean_prior)")
    lines.append(f"- **Total experiments**: {expected_per_combo * 2 * 13:,}")
    lines.append("")
    lines.append("---")
    lines.append("")

    metrics_to_display = [
        ('test_auc', 'Test AUC ↑', True),
        ('test_ap', 'Test AP ↑', True),
        ('test_accuracy', 'Test Accuracy ↑', True),
        ('test_f1', 'Test F1 ↑', True),
        ('test_ece', 'Test ECE ↓', False),
        ('test_brier', 'Test Brier ↓', False),
    ]

    for metric_name, metric_label, higher_is_better in metrics_to_display:
        lines.append(f"## {metric_label}")
        lines.append("")
        lines.append(f"*Mean ± Std across {expected_per_combo} configurations (7 datasets × 3 c × 7 π × {num_seeds} seeds). **Bold** = best method_prior for each method.*")
        lines.append("")
        table = generate_table(aggregated, metric_name, methods, method_priors, higher_is_better)
        lines.append(table)
        lines.append("")
        lines.append("---")
        lines.append("")

    # Best method_prior per method
    def append_best_table(lines, agg, method_list, label):
        lines.append(f"## Best method_prior per Method ({label})")
        lines.append("")
        lines.append("*Based on Test AUC*")
        lines.append("")
        lines.append("| Method | Best method_prior | Mean AUC | Std AUC |")
        lines.append("|--------|-------------------|----------|---------|")
        for method in method_list:
            best_mean = float('-inf')
            best_prior = None
            best_stats = None
            for method_prior in method_priors:
                stats = agg.get((method, method_prior), {}).get('test_auc')
                if stats and not np.isnan(stats.mean):
                    if stats.mean > best_mean:
                        best_mean = stats.mean
                        best_prior = method_prior
                        best_stats = stats
            if best_prior is not None:
                prior_str = str(best_prior) if isinstance(best_prior, str) else f"{best_prior:.4f}"
                lines.append(f"| {method} | {prior_str} | {best_stats.mean:.3f} | {best_stats.std:.3f} |")
        lines.append("")
        lines.append("---")
        lines.append("")

    append_best_table(lines, aggregated, methods, f"VPU Multi-Seed, {num_seeds} seeds")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    print(f"✓ Generated {output_file}")

    # --- Seed-42 only, all 10 methods ---
    output_file_s42 = Path("analysis/PHASE4_SEED42_COMPREHENSIVE_TABLE.md")
    lines2 = []
    lines2.append("# Phase 4 Results - Method Prior Robustness Analysis (Seed 42)")
    lines2.append("")
    lines2.append("**Configuration:**")
    lines2.append("- **Datasets**: 7 (20News, Connect4, FashionMNIST, IMDB, MNIST, Mushrooms, Spambase)")
    lines2.append("- **Label frequency (c)**: 3 values [0.01, 0.5, 0.99]")
    lines2.append("- **True prior (π)**: 7 values [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]")
    lines2.append("- **Method_prior values**: 13 values [auto, true, 0.01, 0.1111, 0.2222, 0.3333, 0.4444, 0.5, 0.5556, 0.6667, 0.7778, 0.8889, 0.99]")
    lines2.append("- **Total configurations per (method, method_prior)**: 147")
    lines2.append("- **Methods**: 10 (8 baseline prior-based + 2 VPU variants)")
    lines2.append("- **Total experiments**: 19,110")
    lines2.append("")
    lines2.append("---")
    lines2.append("")

    for metric_name, metric_label, higher_is_better in metrics_to_display:
        lines2.append(f"## {metric_label}")
        lines2.append("")
        lines2.append(f"*Mean ± Std across 147 configurations (7 datasets × 3 c × 7 π). **Bold** = best method_prior for each method.*")
        lines2.append("")
        table = generate_table(aggregated_seed42, metric_name, all_methods, method_priors, higher_is_better)
        lines2.append(table)
        lines2.append("")
        lines2.append("---")
        lines2.append("")

    append_best_table(lines2, aggregated_seed42, all_methods, "Seed 42, All Methods")

    with open(output_file_s42, 'w') as f:
        f.write('\n'.join(lines2))
    print(f"✓ Generated {output_file_s42}")


if __name__ == "__main__":
    main()
