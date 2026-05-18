#!/usr/bin/env python3
"""Generate Phase 4 aggregated analysis: method_prior performance across all methods.

This analysis aggregates across all 10 methods to find which method_prior value
performs best overall, similar to how Phase 3 analyzed methods.
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
    """Load all Phase 4 results grouped by method_prior.

    Args:
        results_dir: Single seed directory (e.g., results_phase4/seed_42)
        methods_filter: Optional set of method names to include

    Returns:
        Dict mapping method_prior -> list of (method, metrics, config) tuples
    """
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


def aggregate_by_prior_range(results: Dict, low_threshold: float = 0.5) -> Tuple[Dict, Dict]:
    """Aggregate results separately for low and high true priors.

    Returns:
        (low_prior_aggregated, high_prior_aggregated)
    """
    low_results = defaultdict(list)
    high_results = defaultdict(list)

    for method_prior, runs in results.items():
        for method, metrics, config in runs:
            if config['true_prior'] < low_threshold:
                low_results[method_prior].append((method, metrics, config))
            else:
                high_results[method_prior].append((method, metrics, config))

    return aggregate_by_method_prior(low_results), aggregate_by_method_prior(high_results)


def compute_auto_prior_stats(results: Dict) -> Dict:
    """Compute statistics about what 'auto' prior values actually are.

    Returns dict with min, max, mean, std of auto priors (c × π)
    """
    auto_priors = []

    for method_prior, runs in results.items():
        if method_prior == 'auto':
            for method, metrics, config in runs:
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
    """Generate comprehensive Phase 4 analysis."""
    results_base = Path("results_phase4")
    vpu_methods = {'vpu_mean_prior', 'vpu_nomixup_mean_prior'}
    seeds = sorted([d.name for d in results_base.glob('seed_*')])
    num_seeds = len(seeds)

    output_file = Path("analysis/PHASE4_MULTISEED_AGGREGATED_ANALYSIS.md")

    print(f"Loading Phase 4 VPU results from {num_seeds} seeds...")
    results = load_phase4_results_multiseed(results_base, methods_filter=vpu_methods)
    print(f"Loaded results for {len(results)} method_prior values")

    print("Aggregating overall results...")
    overall_aggregated = aggregate_by_method_prior(results)

    print("Aggregating by prior range...")
    low_prior_aggregated, high_prior_aggregated = aggregate_by_prior_range(results)

    print("Computing auto prior statistics...")
    auto_stats = compute_auto_prior_stats(results)

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
        ('test_anice', 'ANICE ↓', False),
        ('oracle_cross_entropy', 'Oracle CE ↓', False),
    ]

    print("Computing rankings...")
    overall_rankings = compute_rankings(overall_aggregated, metrics, method_priors)
    low_rankings = compute_rankings(low_prior_aggregated, metrics, method_priors)
    high_rankings = compute_rankings(high_prior_aggregated, metrics, method_priors)

    configs_per_mp = 7 * 2 * 3 * 7 * num_seeds

    # Generate markdown document
    lines = []
    lines.append("# Phase 4 Aggregated Analysis - Which method_prior is Best? (Multi-Seed)")
    lines.append("")
    lines.append("**Analysis approach:** Aggregate VPU method performance across all seeds")
    lines.append("to identify which method_prior value is most robust overall.")
    lines.append("")
    lines.append("**Configuration:**")
    lines.append("- **Datasets**: 7 (20News, Connect4, FashionMNIST, IMDB, MNIST, Mushrooms, Spambase)")
    lines.append("- **Methods**: 2 VPU variants (vpu_mean_prior, vpu_nomixup_mean_prior)")
    lines.append("- **Label frequency (c)**: 3 values [0.01, 0.5, 0.99]")
    lines.append("- **True prior (π)**: 7 values [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]")
    lines.append(f"- **Seeds**: {num_seeds} {seeds}")
    lines.append(f"- **Total configurations**: 7 datasets × 2 methods × 3 c × 7 π × {num_seeds} seeds = {configs_per_mp:,} per method_prior")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Overall performance
    lines.append("## Overall Performance Across All Methods and Configurations")
    lines.append("")
    lines.append("*Mean ± Std across {configs_per_mp} runs (7 datasets × 2 methods × 3 c × 7 π × {num_seeds} seeds). **Bold** = best per metric.*")
    lines.append("")
    lines.append(generate_table(overall_aggregated, metrics, method_priors))
    lines.append("")
    lines.append("---")
    lines.append("")

    # Low priors
    lines.append("## Performance on Low Priors (π < 0.5)")
    lines.append("")
    lines.append("*Mean ± Std across configurations with π ∈ {0.01, 0.1, 0.3}. **Bold** = best per metric.*")
    lines.append("")
    lines.append(generate_table(low_prior_aggregated, metrics, method_priors))
    lines.append("")
    lines.append("---")
    lines.append("")

    # High priors
    lines.append("## Performance on High Priors (π ≥ 0.5)")
    lines.append("")
    lines.append("*Mean ± Std across configurations with π ∈ {0.5, 0.7, 0.9, 0.99}. **Bold** = best per metric.*")
    lines.append("")
    lines.append(generate_table(high_prior_aggregated, metrics, method_priors))
    lines.append("")
    lines.append("---")
    lines.append("")

    # Rankings
    lines.append("## Method_Prior Rankings")
    lines.append("")
    lines.append("### Overall Performance (All Configurations)")
    lines.append("")
    lines.append(f"*{len(metrics)} metrics*")
    lines.append("")
    lines.append("| method_prior | Wins | Avg Rank |")
    lines.append("|--------------|------|----------|")

    # Sort by avg rank
    sorted_rankings = sorted(overall_rankings.items(), key=lambda x: x[1]['avg_rank'])
    for mp, stats in sorted_rankings:
        if mp == 'auto':
            prior_str = "auto"
        elif mp == 'true':
            prior_str = "true"
        else:
            prior_str = f"{mp:.4f}" if isinstance(mp, float) else str(mp)

        lines.append(f"| {prior_str} | {stats['wins']}/{stats['total_metrics']} | {stats['avg_rank']:.2f} |")

    lines.append("")
    lines.append("### Performance on Low Priors (π < 0.5)")
    lines.append("")
    lines.append(f"*{len(metrics)} metrics*")
    lines.append("")
    lines.append("| method_prior | Wins | Avg Rank |")
    lines.append("|--------------|------|----------|")

    sorted_low = sorted(low_rankings.items(), key=lambda x: x[1]['avg_rank'])
    for mp, stats in sorted_low:
        if mp == 'auto':
            prior_str = "auto"
        elif mp == 'true':
            prior_str = "true"
        else:
            prior_str = f"{mp:.4f}" if isinstance(mp, float) else str(mp)

        lines.append(f"| {prior_str} | {stats['wins']}/{stats['total_metrics']} | {stats['avg_rank']:.2f} |")

    lines.append("")
    lines.append("### Performance on High Priors (π ≥ 0.5)")
    lines.append("")
    lines.append(f"*{len(metrics)} metrics*")
    lines.append("")
    lines.append("| method_prior | Wins | Avg Rank |")
    lines.append("|--------------|------|----------|")

    sorted_high = sorted(high_rankings.items(), key=lambda x: x[1]['avg_rank'])
    for mp, stats in sorted_high:
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

    # Key insights
    lines.append("## Key Insights")
    lines.append("")

    # Best overall
    best_overall = sorted_rankings[0][0]
    best_overall_str = "auto" if best_overall == 'auto' else ("true" if best_overall == 'true' else f"{best_overall:.4f}")
    lines.append(f"### Best Overall method_prior: **{best_overall_str}**")
    lines.append("")
    lines.append(f"- Achieves **{sorted_rankings[0][1]['wins']}/{sorted_rankings[0][1]['total_metrics']} wins** across {len(metrics)} metrics")
    lines.append(f"- Average rank: **{sorted_rankings[0][1]['avg_rank']:.2f}**")
    lines.append("")

    # 2/3 hypothesis
    rank_0_6667 = None
    for i, (mp, stats) in enumerate(sorted_rankings, 1):
        if mp == 0.6667:
            rank_0_6667 = i
            break

    lines.append("### Hypothesis Validation: Is 2/3 (0.6667) the Optimal Robust Prior?")
    lines.append("")
    if rank_0_6667:
        lines.append(f"- method_prior = 0.6667 ranks **#{rank_0_6667}/{len(sorted_rankings)}** overall")
        lines.append(f"- Achieves {overall_rankings[0.6667]['wins']}/{overall_rankings[0.6667]['total_metrics']} wins")
        lines.append(f"- Average rank: {overall_rankings[0.6667]['avg_rank']:.2f}")
    lines.append("")

    # Bias analysis
    lines.append("## Bias Analysis: The Auto Prior Distribution Problem")
    lines.append("")
    lines.append("### What is 'auto' actually computing?")
    lines.append("")
    lines.append("When method_prior='auto', the prior is computed as **c × π** (label frequency × true prior).")
    lines.append("")
    if auto_stats:
        lines.append(f"**Distribution of auto priors in Phase 4:**")
        lines.append(f"- Minimum: {auto_stats['min']:.4f}")
        lines.append(f"- Maximum: {auto_stats['max']:.4f}")
        lines.append(f"- Mean: {auto_stats['mean']:.4f}")
        lines.append(f"- Std: {auto_stats['std']:.4f}")
        lines.append("")

    lines.append("### Why this creates bias in fixed prior comparison")
    lines.append("")
    lines.append("The grid has uneven coverage:")
    lines.append("")
    lines.append("| c × π | Auto Prior | Example Configs |")
    lines.append("|-------|------------|-----------------|")
    lines.append("| 0.01 × 0.01 | 0.0001 | Extremely low |")
    lines.append("| 0.01 × 0.5 | 0.005 | Very low |")
    lines.append("| 0.5 × 0.5 | 0.25 | Moderate |")
    lines.append("| 0.99 × 0.99 | 0.9801 | Extremely high |")
    lines.append("")
    lines.append("**Problem:** A fixed method_prior (e.g., 0.5) is:")
    lines.append("- WAY TOO HIGH when true prior is 0.0001 (500× larger!)")
    lines.append("- About right when true prior is 0.25")
    lines.append("- WAY TOO LOW when true prior is 0.9801 (0.5× smaller)")
    lines.append("")
    lines.append("This asymmetry biases results:")
    lines.append("- Low fixed priors (0.01-0.3) fail catastrophically on high c×π configs")
    lines.append("- High fixed priors (0.7-0.99) fail catastrophically on low c×π configs")
    lines.append("- **Mid-range priors (0.3-0.6) minimize worst-case failure**")
    lines.append("")

    lines.append("### What would an unbiased experiment look like?")
    lines.append("")
    lines.append("**Option 1: Relative error analysis**")
    lines.append("- Instead of comparing method_prior to arbitrary fixed values")
    lines.append("- Compare method_prior to TRUE prior (c × π) as baseline")
    lines.append("- Measure: AUC(method_prior=X) - AUC(method_prior='true')")
    lines.append("- Question: Which constant X minimizes loss vs. truth?")
    lines.append("")
    lines.append("**Option 2: Stratified analysis**")
    lines.append("- Group configurations by true prior range: [0-0.1], [0.1-0.3], [0.3-0.7], [0.7-1.0]")
    lines.append("- Find best method_prior within each stratum")
    lines.append("- Check if a single method_prior wins across all strata")
    lines.append("")
    lines.append("**Option 3: Adaptive prior scaling**")
    lines.append("- Test priors as MULTIPLES of auto: {0.5×auto, 1×auto, 2×auto, 5×auto}")
    lines.append("- Question: Should we scale the computed prior up/down?")
    lines.append("- This preserves the relative structure across c×π space")
    lines.append("")

    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))

    print(f"✓ Generated {output_file}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Best overall method_prior: {best_overall_str}")
    print(f"  Wins: {sorted_rankings[0][1]['wins']}/{sorted_rankings[0][1]['total_metrics']}")
    print(f"  Avg rank: {sorted_rankings[0][1]['avg_rank']:.2f}")
    print()
    if rank_0_6667:
        print(f"method_prior=0.6667 (2/3 hypothesis):")
        print(f"  Rank: #{rank_0_6667}/{len(sorted_rankings)}")
        print(f"  Wins: {overall_rankings[0.6667]['wins']}/{overall_rankings[0.6667]['total_metrics']}")
        print(f"  Avg rank: {overall_rankings[0.6667]['avg_rank']:.2f}")
    print()
    if auto_stats:
        print(f"Auto prior distribution:")
        print(f"  Range: [{auto_stats['min']:.4f}, {auto_stats['max']:.4f}]")
        print(f"  Mean: {auto_stats['mean']:.4f} ± {auto_stats['std']:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
