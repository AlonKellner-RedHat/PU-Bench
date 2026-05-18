#!/usr/bin/env python3
"""Generate Phase 4 analysis stratified by label frequency (c).

Separates results into three groups: c=0.01, c=0.5, c=0.99
to understand how method_prior performance depends on label availability.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class MetricStats:
    """Statistics for a single metric."""
    mean: float
    std: float
    count: int


def parse_experiment_name(exp_name: str) -> Optional[Dict]:
    """Parse experiment name to extract parameters."""
    parts = exp_name.split('_')

    dataset_idx = parts.index('case-control') if 'case-control' in parts else -1
    if dataset_idx < 0:
        return None

    dataset = '_'.join(parts[:dataset_idx])

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


def load_phase4_results(results_dir: Path, methods_filter=None) -> Dict:
    """Load Phase 4 results from a single seed directory."""
    results = defaultdict(list)

    for json_file in results_dir.glob('*.json'):
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

            config = {
                'dataset': params['dataset'],
                'c': params['c'],
                'true_prior': params['true_prior'],
                'seed': params['seed'],
                'method': method
            }

            results[method_prior].append((method, metrics, config))

    return results


def load_phase4_results_multiseed(results_base: Path, methods_filter=None) -> Dict:
    """Load Phase 4 results from all seed directories."""
    results = defaultdict(list)
    for seed_dir in sorted(results_base.glob('seed_*')):
        seed_results = load_phase4_results(seed_dir, methods_filter)
        for mp, runs in seed_results.items():
            results[mp].extend(runs)
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


def aggregate_by_method_prior(results: Dict) -> Dict:
    """Aggregate all results by method_prior (across all methods and configs).

    Returns:
        Dict mapping method_prior -> Dict[metric_name -> MetricStats]
    """
    aggregated = {}

    for method_prior, runs in results.items():
        metrics_dict = defaultdict(list)

        for method, metrics, config in runs:
            for metric_name, value in metrics.items():
                metrics_dict[metric_name].append(value)

        aggregated[method_prior] = {
            metric_name: compute_stats(values)
            for metric_name, values in metrics_dict.items()
        }

    return aggregated


def filter_by_c(results: Dict, c_value: float) -> Dict:
    """Filter results to only those with specific c value."""
    filtered = defaultdict(list)

    for method_prior, runs in results.items():
        for method, metrics, config in runs:
            if config['c'] == c_value:
                filtered[method_prior].append((method, metrics, config))

    return filtered


def aggregate_by_method_and_prior(results: Dict) -> Dict:
    """Aggregate results by (method, method_prior).

    Returns:
        Dict mapping (method, method_prior) -> Dict[metric_name -> MetricStats]
    """
    aggregated = {}

    # Group by (method, method_prior)
    grouped = defaultdict(list)
    for method_prior, runs in results.items():
        for method, metrics, config in runs:
            grouped[(method, method_prior)].append(metrics)

    # Compute stats for each group
    for (method, method_prior), metrics_list in grouped.items():
        metrics_dict = defaultdict(list)
        for metrics in metrics_list:
            for metric_name, value in metrics.items():
                metrics_dict[metric_name].append(value)

        aggregated[(method, method_prior)] = {
            metric_name: compute_stats(values)
            for metric_name, values in metrics_dict.items()
        }

    return aggregated


def find_best_prior_per_method(results: Dict, method_priors: List) -> Dict:
    """Find best method_prior for each method based on test_auc.

    Returns:
        Dict mapping method -> (best_prior, auc_mean, auc_std)
    """
    # Aggregate by (method, method_prior)
    aggregated = aggregate_by_method_and_prior(results)

    # Find all unique methods
    methods = set(method for method, _ in aggregated.keys())

    best_by_method = {}
    for method in methods:
        best_mean = float('-inf')
        best_prior = None
        best_stats = None

        for method_prior in method_priors:
            stats = aggregated.get((method, method_prior), {}).get('test_auc')
            if stats and not np.isnan(stats.mean):
                if stats.mean > best_mean:
                    best_mean = stats.mean
                    best_prior = method_prior
                    best_stats = stats

        if best_prior is not None:
            best_by_method[method] = (best_prior, best_stats.mean, best_stats.std)

    return best_by_method


def compute_auto_prior_stats(results: Dict, c_value: Optional[float] = None) -> Dict:
    """Compute statistics about what 'auto' prior values actually are.

    Returns dict with min, max, mean, std of auto priors (c × π)
    """
    auto_priors = []

    for method_prior, runs in results.items():
        if method_prior == 'auto':
            for method, metrics, config in runs:
                if c_value is None or config['c'] == c_value:
                    auto_prior = config['c'] * config['true_prior']
                    auto_priors.append(auto_prior)

    if not auto_priors:
        return {}

    return {
        'min': np.min(auto_priors),
        'max': np.max(auto_priors),
        'mean': np.mean(auto_priors),
        'std': np.std(auto_priors),
        'count': len(auto_priors)
    }


def format_value(stats: MetricStats, precision: int = 3) -> str:
    """Format a metric value as 'mean ± std'."""
    if np.isnan(stats.mean):
        return "—"
    return f"{stats.mean:.{precision}f} ± {stats.std:.{precision}f}"


def generate_table(aggregated: Dict, metrics: List[Tuple[str, str, bool]],
                   method_priors: List) -> str:
    """Generate markdown table with method_prior as rows, metrics as columns."""

    # Build markdown table
    lines = []

    # Header
    metric_headers = [label for _, label, _ in metrics]
    header = "| method_prior | " + " | ".join(metric_headers) + " |"
    lines.append(header)

    # Separator
    sep = "|" + "|".join(["---"] * (len(metrics) + 1)) + "|"
    lines.append(sep)

    # Find best for each metric
    best_by_metric = {}
    for metric_name, label, higher_is_better in metrics:
        best_value = float('-inf') if higher_is_better else float('inf')
        best_prior = None

        for method_prior in method_priors:
            stats = aggregated.get(method_prior, {}).get(metric_name)
            if stats and not np.isnan(stats.mean):
                is_better = (stats.mean > best_value) if higher_is_better else (stats.mean < best_value)
                if is_better:
                    best_value = stats.mean
                    best_prior = method_prior

        best_by_metric[metric_name] = best_prior

    # Rows
    for method_prior in method_priors:
        if method_prior == 'auto':
            prior_str = "auto"
        elif method_prior == 'true':
            prior_str = "true"
        else:
            prior_str = f"{method_prior:.4f}" if isinstance(method_prior, float) else str(method_prior)

        row = f"| {prior_str} |"

        for metric_name, label, higher_is_better in metrics:
            stats = aggregated.get(method_prior, {}).get(metric_name)
            if stats:
                value_str = format_value(stats)
                # Bold if best
                if best_by_metric[metric_name] == method_prior:
                    value_str = f"**{value_str}**"
            else:
                value_str = "—"

            row += f" {value_str} |"

        lines.append(row)

    return "\n".join(lines)


def compute_rankings(aggregated: Dict, metrics: List[Tuple[str, str, bool]],
                     method_priors: List) -> Dict:
    """Compute rankings for each method_prior across metrics."""

    rankings = {mp: [] for mp in method_priors}

    for metric_name, label, higher_is_better in metrics:
        # Get values for this metric
        values = []
        for mp in method_priors:
            stats = aggregated.get(mp, {}).get(metric_name)
            if stats and not np.isnan(stats.mean):
                values.append((mp, stats.mean))

        # Sort and assign ranks
        values.sort(key=lambda x: x[1], reverse=higher_is_better)
        for rank, (mp, val) in enumerate(values, 1):
            rankings[mp].append(rank)

    # Compute average rank and wins
    results = {}
    for mp in method_priors:
        ranks = rankings[mp]
        if ranks:
            results[mp] = {
                'avg_rank': np.mean(ranks),
                'wins': sum(1 for r in ranks if r == 1),
                'total_metrics': len(ranks)
            }

    return results


def main():
    """Generate Phase 4 analysis stratified by c."""
    results_base = Path("results_phase4")
    vpu_methods = {'vpu_mean_prior', 'vpu_nomixup_mean_prior'}
    seeds = sorted([d.name for d in results_base.glob('seed_*')])
    num_seeds = len(seeds)

    output_file = Path("analysis/PHASE4_MULTISEED_STRATIFIED_BY_C.md")

    print(f"Loading Phase 4 VPU results from {num_seeds} seeds...")
    results = load_phase4_results_multiseed(results_base, methods_filter=vpu_methods)
    print(f"Loaded results for {len(results)} method_prior values")

    # Define method_priors and metrics
    method_priors = [
        'auto', 'true', 'ep_linear',
        0.01, 0.1111, 0.2222, 0.3333, 0.4444,
        0.5, 0.5556, 0.6667, 0.7778, 0.8889, 0.99
    ]

    metrics = [
        ('test_auc', 'AUC ↑', True),
        ('test_ap', 'AP ↑', True),
        ('test_max_f1', 'Max F1 ↑', True),
        ('test_accuracy', 'Accuracy ↑', True),
        ('test_f1', 'F1 ↑', True),
        ('test_precision', 'Precision ↑', True),
        ('test_recall', 'Recall ↑', True),
        ('test_ece', 'ECE ↓', False),
        ('test_brier', 'Brier ↓', False),
    ]

    c_values = [0.01, 0.5, 0.99]

    # Aggregate for each c value
    print("Aggregating by c values...")
    aggregated_by_c = {}
    auto_stats_by_c = {}
    rankings_by_c = {}

    for c in c_values:
        print(f"  Processing c={c}...")
        filtered = filter_by_c(results, c)
        aggregated_by_c[c] = aggregate_by_method_prior(filtered)
        auto_stats_by_c[c] = compute_auto_prior_stats(results, c)
        rankings_by_c[c] = compute_rankings(aggregated_by_c[c], metrics, method_priors)

    # Also compute overall
    print("Computing overall aggregation...")
    overall_aggregated = aggregate_by_method_prior(results)
    overall_auto_stats = compute_auto_prior_stats(results)
    overall_rankings = compute_rankings(overall_aggregated, metrics, method_priors)

    # Find best prior per method
    print("Computing best prior per method...")
    best_prior_per_method = find_best_prior_per_method(results, method_priors)

    # Find best prior per method for each c value
    best_prior_per_method_by_c = {}
    for c in c_values:
        filtered = filter_by_c(results, c)
        best_prior_per_method_by_c[c] = find_best_prior_per_method(filtered, method_priors)

    # Generate markdown document
    lines = []
    lines.append("# Phase 4 Analysis - Stratified by Label Frequency (Multi-Seed)")
    lines.append("")
    lines.append("**Goal:** Separate results by label frequency (c) to understand how method_prior")
    lines.append("performance depends on the amount of labeled data available.")
    lines.append("")
    lines.append("**Configuration:**")
    lines.append("- **Datasets**: 7 (20News, Connect4, FashionMNIST, IMDB, MNIST, Mushrooms, Spambase)")
    lines.append("- **Methods**: 2 VPU variants (vpu_mean_prior, vpu_nomixup_mean_prior)")
    lines.append("- **Label frequencies (c)**: 3 values [0.01, 0.5, 0.99] - analyzed separately")
    lines.append("- **True priors (π)**: 7 values [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]")
    lines.append(f"- **Seeds**: {num_seeds} {seeds}")
    lines.append("")
    lines.append("**Key insight:** The 'auto' prior is c × π, so:")
    lines.append("- At c=0.01: auto ranges from 0.0001 to 0.0099 (very low)")
    lines.append("- At c=0.5: auto ranges from 0.005 to 0.495 (moderate)")
    lines.append("- At c=0.99: auto ranges from 0.0099 to 0.9801 (very high)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Section for each c value
    for c in c_values:
        lines.append(f"## Results for c = {c}")
        lines.append("")

        # Describe auto prior range
        auto_stats = auto_stats_by_c[c]
        if auto_stats:
            lines.append(f"**Auto prior range at c={c}:**")
            lines.append(f"- When c={c}, auto prior = {c} × π ranges from {auto_stats['min']:.4f} to {auto_stats['max']:.4f}")
            lines.append(f"- Mean: {auto_stats['mean']:.4f} ± {auto_stats['std']:.4f}")
            lines.append("")

        total_configs = 7 * 2 * 7 * num_seeds
        lines.append(f"*Mean ± Std across {total_configs} runs (7 datasets × 2 methods × 7 π × {num_seeds} seeds). **Bold** = best per metric.*")
        lines.append("")

        # Table
        lines.append(generate_table(aggregated_by_c[c], metrics, method_priors))
        lines.append("")

        # Rankings
        lines.append(f"### Rankings for c = {c}")
        lines.append("")
        lines.append(f"*{len(metrics)} metrics*")
        lines.append("")
        lines.append("| method_prior | Wins | Avg Rank |")
        lines.append("|--------------|------|----------|")

        sorted_rankings = sorted(rankings_by_c[c].items(), key=lambda x: x[1]['avg_rank'])
        for mp, stats in sorted_rankings:
            if mp == 'auto':
                prior_str = "auto"
            elif mp == 'true':
                prior_str = "true"
            else:
                prior_str = f"{mp:.4f}" if isinstance(mp, float) else str(mp)

            lines.append(f"| {prior_str} | {stats['wins']}/{stats['total_metrics']} | {stats['avg_rank']:.2f} |")

        lines.append("")
        lines.append("---")
        lines.append("")

    # Summary comparison
    lines.append("## Summary: Best method_prior by Label Frequency")
    lines.append("")
    lines.append("| c | Best method_prior | Wins | Avg Rank | Auto Prior Range |")
    lines.append("|---|-------------------|------|----------|------------------|")

    for c in c_values:
        sorted_rankings = sorted(rankings_by_c[c].items(), key=lambda x: x[1]['avg_rank'])
        best_mp = sorted_rankings[0][0]
        best_stats = sorted_rankings[0][1]

        if best_mp == 'auto':
            prior_str = "auto"
        elif best_mp == 'true':
            prior_str = "true"
        elif isinstance(best_mp, str):
            prior_str = best_mp
        else:
            prior_str = f"{best_mp:.4f}"

        auto_stats = auto_stats_by_c[c]
        auto_range_str = f"[{auto_stats['min']:.4f}, {auto_stats['max']:.4f}]"

        lines.append(f"| {c} | **{prior_str}** | {best_stats['wins']}/{best_stats['total_metrics']} | {best_stats['avg_rank']:.2f} | {auto_range_str} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Overall comparison
    lines.append("## Overall (All c values combined)")
    lines.append("")
    lines.append("For reference, here are the top 5 method_priors when aggregating across all c values:")
    lines.append("")
    lines.append("| Rank | method_prior | Wins | Avg Rank |")
    lines.append("|------|--------------|------|----------|")

    sorted_overall = sorted(overall_rankings.items(), key=lambda x: x[1]['avg_rank'])
    for i, (mp, stats) in enumerate(sorted_overall[:5], 1):
        if mp == 'auto':
            prior_str = "auto"
        elif mp == 'true':
            prior_str = "true"
        elif isinstance(mp, str):
            prior_str = mp
        else:
            prior_str = f"{mp:.4f}"

        lines.append(f"| {i} | {prior_str} | {stats['wins']}/{stats['total_metrics']} | {stats['avg_rank']:.2f} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Best prior per method
    lines.append("## Best method_prior per Method (Overall)")
    lines.append("")
    lines.append("*Based on Test AUC across all configurations (7 datasets × 3 c × 7 π)*")
    lines.append("")
    lines.append("| Method | Best method_prior | Mean AUC | Std AUC |")
    lines.append("|--------|-------------------|----------|---------|")

    # Sort methods alphabetically
    sorted_methods = sorted(best_prior_per_method.keys())
    for method in sorted_methods:
        best_prior, auc_mean, auc_std = best_prior_per_method[method]

        if best_prior == 'auto':
            prior_str = "auto"
        elif best_prior == 'true':
            prior_str = "true"
        elif isinstance(best_prior, str):
            prior_str = best_prior
        else:
            prior_str = f"{best_prior:.4f}"

        lines.append(f"| {method} | {prior_str} | {auc_mean:.3f} | {auc_std:.3f} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Best prior per method for each c
    for c in c_values:
        lines.append(f"## Best method_prior per Method at c = {c}")
        lines.append("")
        lines.append(f"*Based on Test AUC across {7 * 7 * num_seeds} configurations (7 datasets × 7 π × {num_seeds} seeds)*")
        lines.append("")
        lines.append("| Method | Best method_prior | Mean AUC | Std AUC |")
        lines.append("|--------|-------------------|----------|---------|")

        sorted_methods = sorted(best_prior_per_method_by_c[c].keys())
        for method in sorted_methods:
            best_prior, auc_mean, auc_std = best_prior_per_method_by_c[c][method]

            if best_prior == 'auto':
                prior_str = "auto"
            elif best_prior == 'true':
                prior_str = "true"
            elif isinstance(best_prior, str):
                prior_str = best_prior
            else:
                prior_str = f"{best_prior:.4f}"

            lines.append(f"| {method} | {prior_str} | {auc_mean:.3f} | {auc_std:.3f} |")

        lines.append("")
        lines.append("---")
        lines.append("")

    # Key insights
    lines.append("## Key Insights")
    lines.append("")
    lines.append("### Does the best method_prior change with label frequency?")
    lines.append("")

    best_by_c = {}
    for c in c_values:
        sorted_rankings = sorted(rankings_by_c[c].items(), key=lambda x: x[1]['avg_rank'])
        best_mp = sorted_rankings[0][0]
        best_by_c[c] = best_mp

        if best_mp == 'auto':
            prior_str = "auto"
        elif best_mp == 'true':
            prior_str = "true"
        elif isinstance(best_mp, str):
            prior_str = best_mp
        else:
            prior_str = f"{best_mp:.4f}"

        lines.append(f"- **c = {c}**: Best method_prior = {prior_str}")

    # Check if consistent
    unique_best = set(best_by_c.values())
    if len(unique_best) == 1:
        best_mp = list(unique_best)[0]
        if best_mp == 'auto':
            prior_str = "auto"
        elif best_mp == 'true':
            prior_str = "true"
        elif isinstance(best_mp, str):
            prior_str = best_mp
        else:
            prior_str = f"{best_mp:.4f}"
        lines.append("")
        lines.append(f"**✓ Consistent:** method_prior = {prior_str} is best across all label frequencies!")
    else:
        lines.append("")
        lines.append("**✗ Inconsistent:** Different label frequencies favor different method_prior values.")

    lines.append("")

    # Method-specific patterns
    lines.append("### Method-specific best priors")
    lines.append("")
    lines.append("Looking at best method_prior per method across all configurations:")
    lines.append("")

    # Count how many methods prefer each prior
    prior_counts = defaultdict(int)
    for method in sorted_methods:
        best_prior, _, _ = best_prior_per_method[method]
        prior_counts[best_prior] += 1

    # Show distribution
    lines.append("**Distribution of best method_prior values:**")
    sorted_priors = sorted(prior_counts.items(), key=lambda x: x[1], reverse=True)
    for prior, count in sorted_priors:
        if prior == 'auto':
            prior_str_temp = "auto"
        elif prior == 'true':
            prior_str_temp = "true"
        elif isinstance(prior, str):
            prior_str_temp = prior
        else:
            prior_str_temp = f"{prior:.4f}"

        lines.append(f"- **{prior_str_temp}**: {count}/{len(sorted_methods)} methods")

    lines.append("")

    # Check if any method's best prior changes dramatically with c
    lines.append("### Stability of best prior across c values")
    lines.append("")
    lines.append("Does each method's best prior change with label frequency?")
    lines.append("")
    lines.append("| Method | c=0.01 | c=0.5 | c=0.99 | Consistent? |")
    lines.append("|--------|--------|-------|--------|-------------|")

    for method in sorted_methods:
        priors_by_c = []
        for c in c_values:
            if method in best_prior_per_method_by_c[c]:
                best_prior, _, _ = best_prior_per_method_by_c[c][method]
                if best_prior == 'auto':
                    prior_str_temp = "auto"
                elif best_prior == 'true':
                    prior_str_temp = "true"
                elif isinstance(best_prior, str):
                    prior_str_temp = best_prior
                else:
                    prior_str_temp = f"{best_prior:.4f}"
                priors_by_c.append(prior_str_temp)
            else:
                priors_by_c.append("—")

        # Check consistency
        unique_priors = set(p for p in priors_by_c if p != "—")
        is_consistent = "✓" if len(unique_priors) == 1 else "✗"

        lines.append(f"| {method} | {priors_by_c[0]} | {priors_by_c[1]} | {priors_by_c[2]} | {is_consistent} |")

    lines.append("")

    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))

    print(f"✓ Generated {output_file}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY: Best method_prior by Label Frequency (c)")
    print("="*70)
    for c in c_values:
        sorted_rankings = sorted(rankings_by_c[c].items(), key=lambda x: x[1]['avg_rank'])
        best_mp = sorted_rankings[0][0]
        best_stats = sorted_rankings[0][1]

        if best_mp == 'auto':
            prior_str = "auto"
        elif best_mp == 'true':
            prior_str = "true"
        elif isinstance(best_mp, str):
            prior_str = best_mp
        else:
            prior_str = f"{best_mp:.4f}"

        auto_stats = auto_stats_by_c[c]
        print(f"c = {c}:")
        print(f"  Best method_prior: {prior_str}")
        print(f"  Wins: {best_stats['wins']}/{best_stats['total_metrics']}")
        print(f"  Avg rank: {best_stats['avg_rank']:.2f}")
        print(f"  Auto prior range: [{auto_stats['min']:.4f}, {auto_stats['max']:.4f}]")
        print()
    print("="*70)


if __name__ == "__main__":
    main()
