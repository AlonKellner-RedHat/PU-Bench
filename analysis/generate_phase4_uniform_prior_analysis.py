#!/usr/bin/env python3
"""Phase 4 analysis with uniform effective-prior weighting.

The effective prior seen by methods is c × π. The experimental grid
overrepresents low effective priors (c=0.01 produces ep ∈ [0.0001, 0.0099]).

To answer "which method_prior is best if the effective prior is uniformly
distributed?", we:
1. Compute effective prior ep = c × π for each experiment
2. Bin experiments by ep into equal-width bins across [0, 1]
3. Compute mean performance within each bin
4. Average across bins with equal weight
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import beta as beta_dist
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def parse_experiment_name(exp_name: str) -> Optional[Dict]:
    parts = exp_name.split('_')
    dataset_idx = parts.index('case-control') if 'case-control' in parts else -1
    if dataset_idx < 0:
        return None

    dataset = '_'.join(parts[:dataset_idx])
    c = seed = true_prior = method_prior = None

    for part in parts[dataset_idx:]:
        if part.startswith('c'):
            try: c = float(part[1:])
            except ValueError: pass
        elif part.startswith('seed'):
            try: seed = int(part[4:])
            except ValueError: pass
        elif part.startswith('trueprior'):
            try: true_prior = float(part[9:])
            except ValueError: pass
        elif part.startswith('methodprior'):
            val = part[11:]
            if val == 'auto': method_prior = 'auto'
            elif val == 'true': method_prior = 'true'
            elif val == 'eplinear': method_prior = 'ep_linear'
            else:
                try: method_prior = float(val)
                except ValueError: pass

    if any(v is None for v in [c, seed, true_prior, method_prior]):
        return None

    return {'dataset': dataset, 'c': c, 'seed': seed,
            'true_prior': true_prior, 'method_prior': method_prior}


def load_results(results_base: Path, methods_filter=None) -> List[Dict]:
    """Load all results as flat list with effective prior computed."""
    records = []

    for seed_dir in sorted(results_base.glob('seed_*')):
        for json_file in seed_dir.glob('*.json'):
            with open(json_file) as f:
                data = json.load(f)

            params = parse_experiment_name(data.get('experiment', ''))
            if params is None:
                continue

            ep = params['c'] * params['true_prior']

            for method, md in data.get('runs', {}).items():
                if methods_filter and method not in methods_filter:
                    continue
                if 'best' not in md:
                    continue
                metrics = md['best'].get('metrics', {})
                if not metrics:
                    continue

                records.append({
                    'method': method,
                    'method_prior': params['method_prior'],
                    'effective_prior': ep,
                    'c': params['c'],
                    'true_prior': params['true_prior'],
                    'dataset': params['dataset'],
                    'seed': params['seed'],
                    'best_epoch': md['best'].get('epoch', float('nan')),
                    **{k: v for k, v in metrics.items()},
                })

    return records


def uniform_weighted_stats(records: List[Dict], method_priors: List,
                           metric: str, n_bins: int = 5
                           ) -> Tuple[Dict, Dict]:
    """Compute uniformly-weighted mean for each method_prior.

    Returns:
        (weighted_means, bin_info) where:
        - weighted_means: method_prior -> (mean, std, n_bins_with_data)
        - bin_info: dict with bin edges, counts, coverage
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)

    # Group records by (method_prior, bin)
    binned = defaultdict(lambda: defaultdict(list))
    bin_counts = defaultdict(int)

    for r in records:
        ep = r['effective_prior']
        val = r.get(metric)
        if val is None or np.isnan(val):
            continue

        bin_idx = min(int(ep * n_bins), n_bins - 1)
        mp = r['method_prior']
        binned[mp][bin_idx].append(val)
        bin_counts[bin_idx] += 1

    # Compute per-bin means, then average across bins
    weighted_means = {}
    for mp in method_priors:
        bin_means = []
        for b in range(n_bins):
            vals = binned[mp][b]
            if vals:
                bin_means.append(np.mean(vals))

        if bin_means:
            weighted_means[mp] = (
                float(np.mean(bin_means)),
                float(np.std(bin_means)),
                len(bin_means)
            )

    # Bin coverage info
    bin_info = {
        'edges': bin_edges,
        'counts': {b: bin_counts[b] for b in range(n_bins)},
        'total_records': len(records),
    }

    return weighted_means, bin_info


def naive_stats(records: List[Dict], method_priors: List,
                metric: str) -> Dict:
    """Compute naive (unweighted) mean for comparison."""
    grouped = defaultdict(list)
    for r in records:
        val = r.get(metric)
        if val is None or np.isnan(val):
            continue
        grouped[r['method_prior']].append(val)

    return {
        mp: (float(np.mean(grouped[mp])), float(np.std(grouped[mp])), len(grouped[mp]))
        for mp in method_priors if mp in grouped and grouped[mp]
    }


def run_analysis(records, output_file, plot_subdir, title_suffix, num_seeds, seeds):
    """Run the full analysis pipeline on a set of records."""

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
        ('test_oracle_ce', 'Oracle CE ↓', False),
        ('best_epoch', 'Epochs ↓', False),
    ]

    N_BINS = 5

    # Compute effective prior distribution
    eps = [r['effective_prior'] for r in records]
    ep_hist, _ = np.histogram(eps, bins=np.linspace(0, 1, N_BINS + 1))

    lines = []
    lines.append(f"# Phase 4 Analysis - Uniform Effective-Prior Weighting{title_suffix}")
    lines.append("")
    lines.append("## Why This Analysis?")
    lines.append("")
    lines.append("The effective prior seen by methods is **ep = c × π**. The experimental grid")
    lines.append("is heavily biased towards low effective priors:")
    lines.append("")
    lines.append("| Effective Prior Bin | Raw Count | % of Data |")
    lines.append("|---------------------|-----------|-----------|")
    for b in range(N_BINS):
        lo, hi = b / N_BINS, (b + 1) / N_BINS
        pct = 100 * ep_hist[b] / len(eps)
        lines.append(f"| [{lo:.3f}, {hi:.3f}) | {ep_hist[b]:,} | {pct:.1f}% |")
    lines.append("")
    lines.append("**Problem:** Low ep bins dominate. A naive average gives them disproportionate weight.")
    lines.append("")
    lines.append("**Solution:** Compute mean performance *within* each ep bin, then average across bins")
    lines.append("with equal weight. This answers: *if the effective prior were uniformly distributed,*")
    lines.append("*which method_prior would work best?*")
    lines.append("")
    lines.append("**Configuration:**")
    methods_in_data = sorted(set(r['method'] for r in records))
    lines.append(f"- **Methods**: {', '.join(methods_in_data)}")
    lines.append(f"- **Seeds**: {num_seeds} {seeds}")
    lines.append("- **Datasets**: 7 | **c**: 3 [0.01, 0.5, 0.99] | **π**: 7 [0.01–0.99]")
    lines.append(f"- **Effective prior bins**: {N_BINS} equal-width bins across [0, 1]")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Main comparison table: naive vs uniform-weighted
    lines.append("## Naive vs Uniform-Weighted Rankings (Test AUC)")
    lines.append("")

    naive = naive_stats(records, method_priors, 'test_auc')
    uniform, bin_info = uniform_weighted_stats(records, method_priors, 'test_auc', N_BINS)

    lines.append("| method_prior | Naive Mean | Naive Rank | Uniform Mean | Uniform Rank | Bins w/ Data |")
    lines.append("|--------------|------------|------------|--------------|--------------|--------------|")

    naive_sorted = sorted(naive.items(), key=lambda x: x[1][0], reverse=True)
    uniform_sorted = sorted(uniform.items(), key=lambda x: x[1][0], reverse=True)

    naive_ranks = {mp: i + 1 for i, (mp, _) in enumerate(naive_sorted)}
    uniform_ranks = {mp: i + 1 for i, (mp, _) in enumerate(uniform_sorted)}

    for mp in method_priors:
        if mp == 'auto': ps = "auto"
        elif mp == 'true': ps = "true"
        elif mp == 'ep_linear': ps = "(ep+1)/3"
        else: ps = f"{mp:.4f}"

        n_mean, n_std, n_count = naive.get(mp, (float('nan'), 0, 0))
        u_mean, u_std, u_bins = uniform.get(mp, (float('nan'), 0, 0))
        n_rank = naive_ranks.get(mp, '-')
        u_rank = uniform_ranks.get(mp, '-')

        n_str = f"{n_mean:.4f}" if not np.isnan(n_mean) else "—"
        u_str = f"{u_mean:.4f}" if not np.isnan(u_mean) else "—"

        # Bold the best
        if u_rank == 1: u_str = f"**{u_str}**"
        if n_rank == 1: n_str = f"**{n_str}**"

        lines.append(f"| {ps} | {n_str} | {n_rank} | {u_str} | {u_rank} | {u_bins}/{N_BINS} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Full uniform-weighted tables for all metrics
    lines.append("## Uniform-Weighted Performance Across All Metrics")
    lines.append("")
    lines.append(f"*Mean across {N_BINS} equal-width effective-prior bins. **Bold** = best per metric.*")
    lines.append("")

    # Header
    metric_labels = [label for _, label, _ in metrics]
    lines.append("| method_prior | " + " | ".join(metric_labels) + " |")
    lines.append("|" + "|".join(["---"] * (len(metrics) + 1)) + "|")

    # Compute uniform stats for all metrics
    all_uniform = {}
    for metric_name, _, _ in metrics:
        all_uniform[metric_name], _ = uniform_weighted_stats(
            records, method_priors, metric_name, N_BINS)

    # Find best per metric
    best_per_metric = {}
    for metric_name, _, higher_is_better in metrics:
        best_val = float('-inf') if higher_is_better else float('inf')
        best_mp = None
        for mp in method_priors:
            if mp not in all_uniform[metric_name]:
                continue
            val = all_uniform[metric_name][mp][0]
            is_better = (val > best_val) if higher_is_better else (val < best_val)
            if is_better:
                best_val = val
                best_mp = mp
        best_per_metric[metric_name] = best_mp

    # Rows
    for mp in method_priors:
        if mp == 'auto': ps = "auto"
        elif mp == 'true': ps = "true"
        elif mp == 'ep_linear': ps = "(ep+1)/3"
        else: ps = f"{mp:.4f}"

        row = f"| {ps} |"
        for metric_name, _, _ in metrics:
            stats = all_uniform[metric_name].get(mp)
            if stats:
                mean, std, n_bins = stats
                val_str = f"{mean:.3f} ± {std:.3f}"
                if best_per_metric[metric_name] == mp:
                    val_str = f"**{val_str}**"
            else:
                val_str = "—"
            row += f" {val_str} |"
        lines.append(row)

    lines.append("")
    lines.append("---")
    lines.append("")

    # Rankings
    lines.append("## Uniform-Weighted Rankings")
    lines.append("")
    lines.append(f"*{len(metrics)} metrics*")
    lines.append("")
    lines.append("| method_prior | Wins | Avg Rank |")
    lines.append("|--------------|------|----------|")

    rankings = {mp: [] for mp in method_priors}
    for metric_name, _, higher_is_better in metrics:
        vals = []
        for mp in method_priors:
            if mp in all_uniform[metric_name]:
                vals.append((mp, all_uniform[metric_name][mp][0]))
        vals.sort(key=lambda x: x[1], reverse=higher_is_better)
        for rank, (mp, _) in enumerate(vals, 1):
            rankings[mp].append(rank)

    sorted_rankings = sorted(
        [(mp, r) for mp, r in rankings.items() if r],
        key=lambda x: np.mean(x[1])
    )

    for mp, ranks in sorted_rankings:
        if mp == 'auto': ps = "auto"
        elif mp == 'true': ps = "true"
        elif mp == 'ep_linear': ps = "(ep+1)/3"
        else: ps = f"{mp:.4f}"

        wins = sum(1 for r in ranks if r == 1)
        avg = np.mean(ranks)
        lines.append(f"| {ps} | {wins}/{len(ranks)} | {avg:.2f} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Per-bin breakdown: best method_prior per effective-prior bin (AUC)
    lines.append("## Per-Bin Rankings (Test AUC)")
    lines.append("")
    lines.append("*Which method_prior works best in each effective-prior range?*")
    lines.append("")

    # Compute per-bin means for AUC
    bin_edges = np.linspace(0, 1, N_BINS + 1)
    binned_auc = defaultdict(lambda: defaultdict(list))
    for r in records:
        ep = r['effective_prior']
        val = r.get('test_auc')
        if val is None or np.isnan(val):
            continue
        bin_idx = min(int(ep * N_BINS), N_BINS - 1)
        binned_auc[r['method_prior']][bin_idx].append(val)

    # Header: method_prior | bin0 | bin1 | ... | binN
    bin_headers = [f"[{b/N_BINS:.2f}, {(b+1)/N_BINS:.2f})" for b in range(N_BINS)]
    lines.append("| method_prior | " + " | ".join(bin_headers) + " |")
    lines.append("|" + "|".join(["---"] * (N_BINS + 1)) + "|")

    # Find best method_prior per bin
    best_per_bin = {}
    for b in range(N_BINS):
        best_val = float('-inf')
        best_mp = None
        for mp in method_priors:
            vals = binned_auc[mp][b]
            if vals:
                mean = np.mean(vals)
                if mean > best_val:
                    best_val = mean
                    best_mp = mp
        best_per_bin[b] = best_mp

    # Rows
    for mp in method_priors:
        if mp == 'auto': ps = "auto"
        elif mp == 'true': ps = "true"
        elif mp == 'ep_linear': ps = "(ep+1)/3"
        else: ps = f"{mp:.4f}"

        row = f"| {ps} |"
        for b in range(N_BINS):
            vals = binned_auc[mp][b]
            if vals:
                mean = np.mean(vals)
                val_str = f"{mean:.3f}"
                if best_per_bin[b] == mp:
                    val_str = f"**{val_str}**"
            else:
                val_str = "—"
            row += f" {val_str} |"
        lines.append(row)

    lines.append("")

    # Summary: best per bin
    lines.append("**Best method_prior per bin:**")
    lines.append("")
    for b in range(N_BINS):
        lo, hi = b / N_BINS, (b + 1) / N_BINS
        mp = best_per_bin[b]
        if mp is None:
            ps = "—"
        elif mp == 'auto': ps = "auto"
        elif mp == 'true': ps = "true"
        elif mp == 'ep_linear': ps = "(ep+1)/3"
        else: ps = f"{mp:.4f}"
        lines.append(f"- **ep ∈ [{lo:.2f}, {hi:.2f})**: {ps}")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Per-bin breakdown: Oracle CE
    lines.append("## Per-Bin Rankings (Oracle Cross-Entropy)")
    lines.append("")
    lines.append("*Which method_prior achieves best calibration in each effective-prior range? (lower = better)*")
    lines.append("")

    binned_ce = defaultdict(lambda: defaultdict(list))
    for r in records:
        ep = r['effective_prior']
        val = r.get('test_oracle_ce')
        if val is None or np.isnan(val):
            continue
        bin_idx = min(int(ep * N_BINS), N_BINS - 1)
        binned_ce[r['method_prior']][bin_idx].append(val)

    lines.append("| method_prior | " + " | ".join(bin_headers) + " |")
    lines.append("|" + "|".join(["---"] * (N_BINS + 1)) + "|")

    best_ce_per_bin = {}
    for b in range(N_BINS):
        best_val = float('inf')
        best_mp = None
        for mp in method_priors:
            vals = binned_ce[mp][b]
            if vals:
                mean = np.mean(vals)
                if mean < best_val:
                    best_val = mean
                    best_mp = mp
        best_ce_per_bin[b] = best_mp

    for mp in method_priors:
        if mp == 'auto': ps = "auto"
        elif mp == 'true': ps = "true"
        elif mp == 'ep_linear': ps = "(ep+1)/3"
        else: ps = f"{mp:.4f}"

        row = f"| {ps} |"
        for b in range(N_BINS):
            vals = binned_ce[mp][b]
            if vals:
                mean = np.mean(vals)
                val_str = f"{mean:.3f}"
                if best_ce_per_bin[b] == mp:
                    val_str = f"**{val_str}**"
            else:
                val_str = "—"
            row += f" {val_str} |"
        lines.append(row)

    lines.append("")
    lines.append("**Best method_prior per bin (Oracle CE):**")
    lines.append("")
    for b in range(N_BINS):
        lo, hi = b / N_BINS, (b + 1) / N_BINS
        mp = best_ce_per_bin[b]
        if mp is None: ps = "—"
        elif mp == 'auto': ps = "auto"
        elif mp == 'true': ps = "true"
        elif mp == 'ep_linear': ps = "(ep+1)/3"
        else: ps = f"{mp:.4f}"
        lines.append(f"- **ep ∈ [{lo:.2f}, {hi:.2f})**: {ps}")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Scatter plots: metric vs effective prior, colored by method_prior
    plot_metrics = [
        ('test_auc', 'AUC', True),
        ('test_ap', 'AP', True),
        ('test_max_f1', 'Max F1', True),
        ('test_accuracy', 'Accuracy', True),
        ('test_f1', 'F1', True),
        ('test_precision', 'Precision', True),
        ('test_recall', 'Recall', True),
        ('test_ece', 'ECE', False),
        ('test_brier', 'Brier', False),
        ('test_oracle_ce', 'Oracle CE', False),
        ('test_anice', 'A-NICE', False),
        ('test_snice', 'S-NICE', False),
    ]

    plot_dir = Path(f"analysis/{plot_subdir}")
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Constant method_prior values only (exclude auto, true, ep_linear)
    constant_priors_for_plot = sorted([mp for mp in method_priors if isinstance(mp, float)])

    # Group records by (method_prior, effective_prior)
    ep_values_set = sorted(set(r['effective_prior'] for r in records))

    grouped = defaultdict(lambda: defaultdict(list))
    for r in records:
        grouped[r['method_prior']][r['effective_prior']].append(r)

    def beta_curve(x, a, b, scale, offset):
        """Scaled beta PDF for curve fitting."""
        pdf = beta_dist.pdf(x, a, b)
        return scale * pdf + offset

    def find_beta_peak(x_vals, y_vals, higher_is_better):
        """Fit a beta-shaped curve and return its peak (mode).

        For lower-is-better metrics, we negate y to find the minimum.
        """
        x = np.array(x_vals)
        y = np.array(y_vals)

        if not higher_is_better:
            y = -y

        # Normalize y to [0, 1] range for fitting
        y_min, y_max = y.min(), y.max()
        if y_max - y_min < 1e-10:
            return np.mean(x)
        y_norm = (y - y_min) / (y_max - y_min)

        try:
            popt, _ = curve_fit(
                beta_curve, x, y_norm,
                p0=[2.0, 2.0, 1.0, 0.0],
                bounds=([1.01, 1.01, -np.inf, -np.inf],
                        [50.0, 50.0, np.inf, np.inf]),
                maxfev=5000
            )
            a, b = popt[0], popt[1]
            mode = (a - 1) / (a + b - 2)
            return np.clip(mode, 0.01, 0.99)
        except:
            # Fallback: weighted average
            weights = y_norm - y_norm.min()
            if weights.sum() < 1e-10:
                return np.mean(x)
            return np.average(x, weights=weights)

    print("Generating beta-fit optimal-prior scatter plots...")
    lines.append("## Optimal Constant method_prior vs Effective Prior (Beta-Fit)")
    lines.append("")
    lines.append("*For each effective prior, a beta distribution is fit to metric(method_prior),*")
    lines.append("*and its mode gives the smoothed optimal prior (y-axis). This uses all 11 constant*")
    lines.append("*prior data points rather than just the argmax, reducing noise.*")
    lines.append("")

    for metric_key, metric_label, higher_is_better in plot_metrics:
        fig, ax = plt.subplots(figsize=(10, 6))

        # For each effective prior, fit beta and find peak
        opt_eps = []
        opt_priors = []
        for ep in ep_values_set:
            x_vals = []
            y_vals = []
            for mp in constant_priors_for_plot:
                vals = [r[metric_key] for r in grouped[mp].get(ep, [])
                        if r.get(metric_key) is not None and not np.isnan(r[metric_key])]
                if vals:
                    x_vals.append(mp)
                    y_vals.append(np.mean(vals))

            if len(x_vals) >= 4:
                peak = find_beta_peak(x_vals, y_vals, higher_is_better)
                opt_eps.append(ep)
                opt_priors.append(peak)

        ax.scatter(opt_eps, opt_priors, color='#377eb8', s=80, zorder=5,
                  edgecolors='white', linewidth=0.8, label='Beta-fit peak')

        # Best fitting line to the beta peaks
        if len(opt_eps) >= 2:
            coeffs = np.polyfit(opt_eps, opt_priors, 1)
            fit_x = np.linspace(0, 1, 100)
            fit_y = np.polyval(coeffs, fit_x)
            ax.plot(fit_x, fit_y, color='#377eb8', linewidth=2, linestyle='-',
                   alpha=0.6, label=f'Best fit: {coeffs[0]:.2f}·ep + {coeffs[1]:.2f}')

        # Reference lines
        ax.axhline(y=0.5, color='#984ea3', linewidth=1.5, linestyle=':',
                  label='0.5 (1/2)', alpha=0.7)
        ax.axhline(y=0.6, color='#ff7f00', linewidth=1.5, linestyle=':',
                  label='0.6 (3/5)', alpha=0.7)
        ax.axhline(y=0.6667, color='#4daf4a', linewidth=1.5, linestyle=':',
                  label='0.6667 (2/3)', alpha=0.7)

        ax.set_xlabel('Effective Prior (c × π)', fontsize=12)
        ax.set_ylabel('Optimal Constant method_prior (beta-fit peak)', fontsize=12)
        ax.set_title(f'Optimal Prior for {metric_label}', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        fname = f"phase4_optimal_{metric_key.replace('test_', '')}.png"
        fig.savefig(plot_dir / fname, dpi=150)
        plt.close(fig)

        lines.append(f"### {metric_label}")
        lines.append("")
        lines.append(f"![Optimal Prior for {metric_label}]({plot_subdir}/{fname})")
        lines.append("")

    lines.append("---")
    lines.append("")

    # Per-dataset best-fit lines
    datasets = sorted(set(r['dataset'] for r in records))

    # Group records by dataset
    by_dataset = defaultdict(list)
    for r in records:
        by_dataset[r['dataset']].append(r)

    lines.append("## Per-Dataset Best-Fit Lines (Beta-Fit)")
    lines.append("")
    lines.append("*How much does the optimal method_prior vary across datasets?*")
    lines.append("*Each row shows the best-fit line (slope·ep + intercept) for one dataset.*")
    lines.append("")

    # Header
    fit_metrics = [label for _, label, _ in plot_metrics]
    lines.append("| Dataset | " + " | ".join(fit_metrics) + " |")
    lines.append("|---------|" + "|".join(["---"] * len(plot_metrics)) + "|")

    # Compute per-dataset fits
    all_slopes = defaultdict(list)
    all_intercepts = defaultdict(list)

    for ds in datasets:
        ds_records = by_dataset[ds]

        # Group by (method_prior, effective_prior) for this dataset
        ds_grouped = defaultdict(lambda: defaultdict(list))
        for r in ds_records:
            ds_grouped[r['method_prior']][r['effective_prior']].append(r)

        ds_ep_values = sorted(set(r['effective_prior'] for r in ds_records))

        row = f"| {ds} |"

        for metric_key, metric_label, higher_is_better in plot_metrics:
            opt_eps = []
            opt_priors = []
            for ep in ds_ep_values:
                x_vals = []
                y_vals = []
                for mp in constant_priors_for_plot:
                    vals = [r[metric_key] for r in ds_grouped[mp].get(ep, [])
                            if r.get(metric_key) is not None and not np.isnan(r[metric_key])]
                    if vals:
                        x_vals.append(mp)
                        y_vals.append(np.mean(vals))

                if len(x_vals) >= 4:
                    peak = find_beta_peak(x_vals, y_vals, higher_is_better)
                    opt_eps.append(ep)
                    opt_priors.append(peak)

            if len(opt_eps) >= 2:
                coeffs = np.polyfit(opt_eps, opt_priors, 1)
                row += f" {coeffs[0]:.2f}·ep + {coeffs[1]:.2f} |"
                all_slopes[metric_key].append(coeffs[0])
                all_intercepts[metric_key].append(coeffs[1])
            else:
                row += " — |"

        lines.append(row)

    # Add mean ± std row
    row_mean = "| **Mean ± Std** |"
    for metric_key, _, _ in plot_metrics:
        slopes = all_slopes[metric_key]
        intercepts = all_intercepts[metric_key]
        if slopes:
            row_mean += f" {np.mean(slopes):.2f}±{np.std(slopes):.2f}·ep + {np.mean(intercepts):.2f}±{np.std(intercepts):.2f} |"
        else:
            row_mean += " — |"
    lines.append(row_mean)

    # Add combined (all datasets) row for reference
    row_all = "| **Combined** |"
    for metric_key, _, _ in plot_metrics:
        slopes = all_slopes[metric_key]
        intercepts = all_intercepts[metric_key]
        # Recompute from the full-dataset beta-fit (already done above, extract from plot)
        # Just use the mean of per-dataset as proxy — the actual combined was plotted above
        row_all += f" (see plots above) |"
    lines.append(row_all)

    lines.append("")
    lines.append("---")
    lines.append("")

    # Best constant method_prior per bin × metric
    constant_priors = [mp for mp in method_priors if isinstance(mp, float)]

    lines.append("## Best Constant method_prior per Bin × Metric")
    lines.append("")
    lines.append("*Rows = effective-prior bins, columns = metrics. Each cell shows the best constant*")
    lines.append("*method_prior value (excluding auto, true, ep_linear). Highlights which prior is optimal*")
    lines.append("*for each (regime, metric) combination.*")
    lines.append("")

    metric_labels_short = [label for _, label, _ in metrics]
    lines.append("| ep bin | " + " | ".join(metric_labels_short) + " |")
    lines.append("|" + "|".join(["---"] * (len(metrics) + 1)) + "|")

    # Build binned data for all metrics
    all_binned = {}
    for metric_name, _, _ in metrics:
        binned = defaultdict(lambda: defaultdict(list))
        for r in records:
            ep = r['effective_prior']
            val = r.get(metric_name)
            if val is None or np.isnan(val):
                continue
            bin_idx = min(int(ep * N_BINS), N_BINS - 1)
            binned[r['method_prior']][bin_idx].append(val)
        all_binned[metric_name] = binned

    for b in range(N_BINS):
        lo, hi = b / N_BINS, (b + 1) / N_BINS
        row = f"| [{lo:.2f}, {hi:.2f}) |"

        for metric_name, _, higher_is_better in metrics:
            best_val = float('-inf') if higher_is_better else float('inf')
            best_mp = None

            for mp in constant_priors:
                vals = all_binned[metric_name][mp][b]
                if vals:
                    mean = np.mean(vals)
                    is_better = (mean > best_val) if higher_is_better else (mean < best_val)
                    if is_better:
                        best_val = mean
                        best_mp = mp

            if best_mp is not None:
                row += f" {best_mp:.4f} |"
            else:
                row += " — |"

        lines.append(row)

    lines.append("")
    lines.append("---")
    lines.append("")

    # Key insight
    best_mp = sorted_rankings[0][0]
    best_ranks = sorted_rankings[0][1]
    if best_mp == 'auto': best_str = "auto"
    elif best_mp == 'true': best_str = "true"
    elif best_mp == 'ep_linear': best_str = "(ep+1)/3"
    else: best_str = f"{best_mp:.4f}"

    lines.append("## Key Finding")
    lines.append("")
    lines.append(f"**Best method_prior under uniform effective-prior weighting: {best_str}**")
    lines.append(f"- Wins: {sum(1 for r in best_ranks if r == 1)}/{len(best_ranks)} metrics")
    lines.append(f"- Average rank: {np.mean(best_ranks):.2f}")
    lines.append("")

    # Compare naive vs uniform winner
    naive_best = naive_sorted[0][0]
    if naive_best == 'auto': naive_str = "auto"
    elif naive_best == 'true': naive_str = "true"
    elif naive_best == 'ep_linear': naive_str = "(ep+1)/3"
    else: naive_str = f"{naive_best:.4f}"

    if best_mp == naive_best:
        lines.append(f"This matches the naive (unweighted) winner ({naive_str}), confirming the result is robust.")
    else:
        lines.append(f"**This differs from the naive winner ({naive_str})**, showing the bias correction matters!")
    lines.append("")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))

    print(f"✓ Generated {output_file}")
    print(f"  Best (uniform-weighted): {best_str}")
    print(f"  Best (naive):            {naive_str}")


def main():
    results_base = Path("results_phase4")
    seeds = sorted([d.name for d in results_base.glob('seed_*')])
    num_seeds = len(seeds)

    print(f"Loading Phase 4 results from {num_seeds} seeds...")

    # Load all VPU records
    vpu_methods = {'vpu_mean_prior', 'vpu_nomixup_mean_prior'}
    all_records = load_results(results_base, methods_filter=vpu_methods)
    print(f"Loaded {len(all_records):,} total records")

    # Combined report (both methods)
    print("\n=== Both VPU methods ===")
    run_analysis(
        all_records,
        output_file=Path("analysis/PHASE4_MULTISEED_UNIFORM_PRIOR_ANALYSIS.md"),
        plot_subdir="plots_phase4",
        title_suffix=" (Multi-Seed)",
        num_seeds=num_seeds, seeds=seeds
    )

    # VPU-MP only (with mixup)
    vpu_mp_records = [r for r in all_records if r['method'] == 'vpu_mean_prior']
    print(f"\n=== vpu_mean_prior only ({len(vpu_mp_records):,} records) ===")
    run_analysis(
        vpu_mp_records,
        output_file=Path("analysis/PHASE4_MULTISEED_UNIFORM_PRIOR_VPU_MP.md"),
        plot_subdir="plots_phase4_vpu_mp",
        title_suffix=" — VPU-MP (with mixup)",
        num_seeds=num_seeds, seeds=seeds
    )

    # VPU-nomix-MP only (without mixup)
    vpu_nomix_records = [r for r in all_records if r['method'] == 'vpu_nomixup_mean_prior']
    print(f"\n=== vpu_nomixup_mean_prior only ({len(vpu_nomix_records):,} records) ===")
    run_analysis(
        vpu_nomix_records,
        output_file=Path("analysis/PHASE4_MULTISEED_UNIFORM_PRIOR_VPU_NOMIX_MP.md"),
        plot_subdir="plots_phase4_vpu_nomix_mp",
        title_suffix=" — VPU-nomix-MP (no mixup)",
        num_seeds=num_seeds, seeds=seeds
    )


if __name__ == "__main__":
    main()
