#!/usr/bin/env python3
"""Monitor Phase 3 progress in real-time."""

import json
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime

RESULTS_DIR = Path("results_phase3")
EXPECTED_PER_SEED = 4802

def count_method_runs(seed):
    """Count actual method runs by parsing JSON files."""
    seed_dir = RESULTS_DIR / f"seed_{seed}"
    if not seed_dir.exists():
        return 0, defaultdict(int)

    method_counts = defaultdict(int)
    total = 0

    for json_file in seed_dir.glob("*.json"):
        is_auto = "methodprior_auto" in json_file.name

        try:
            with open(json_file) as f:
                data = json.load(f)

            for method_key, method_data in data.get("runs", {}).items():
                if method_key in ["vpu_mean_prior", "vpu_nomixup_mean_prior"]:
                    hyperparams = method_data.get("hyperparameters", {})
                    method_prior = hyperparams.get("method_prior")

                    if is_auto or method_prior == "auto":
                        full_method_name = f"{method_key}_auto"
                    elif method_prior is None:
                        full_method_name = method_key
                    else:
                        full_method_name = f"{method_key}_{method_prior}"
                else:
                    full_method_name = method_key

                method_counts[full_method_name] += 1
                total += 1

        except Exception:
            continue

    return total, method_counts

print("=" * 80)
print("PHASE 3 PROGRESS MONITOR")
print("=" * 80)
print()

# Initial counts
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

seeds = [42, 456, 789, 1024, 2048]
total_actual = 0
total_missing = 0

for seed in seeds:
    actual, method_counts = count_method_runs(seed)
    missing = EXPECTED_PER_SEED - actual
    pct = actual * 100.0 / EXPECTED_PER_SEED

    total_actual += actual
    total_missing += missing

    status = "✓" if missing == 0 else f"✗ {missing} missing"
    print(f"Seed {seed:4d}: {actual:4,} / {EXPECTED_PER_SEED:,} ({pct:5.1f}%)  {status}")

total_expected = EXPECTED_PER_SEED * 5
pct = total_actual * 100.0 / total_expected

print()
print("=" * 80)
print(f"TOTAL: {total_actual:,} / {total_expected:,} ({pct:.1f}%)")
print(f"Missing: {total_missing:,} method runs")
print("=" * 80)

# Show method breakdown for incomplete seeds
print()
print("Method Breakdown for Incomplete Seeds:")
print("-" * 80)

for seed in [456, 789, 1024, 2048]:
    actual, method_counts = count_method_runs(seed)
    missing = EXPECTED_PER_SEED - actual

    if missing > 0:
        print(f"\nSeed {seed} (missing {missing}):")

        # Expected counts per method
        expected_base = 343  # 7 datasets × 7 c × 7 π
        expected_mp = 343    # Each mean-prior variant

        expected_counts = {
            "vpu": expected_base,
            "vpu_nomixup": expected_base,
            "oracle_bce": expected_base,
            "pn_naive": expected_base,
            "vpu_mean_prior_auto": expected_mp,
            "vpu_mean_prior_0.353": expected_mp,
            "vpu_mean_prior_0.5": expected_mp,
            "vpu_mean_prior_0.69": expected_mp,
            "vpu_mean_prior_1.0": expected_mp,
            "vpu_nomixup_mean_prior_auto": expected_mp,
            "vpu_nomixup_mean_prior_0.353": expected_mp,
            "vpu_nomixup_mean_prior_0.5": expected_mp,
            "vpu_nomixup_mean_prior_0.69": expected_mp,
            "vpu_nomixup_mean_prior_1.0": expected_mp,
        }

        for method in sorted(expected_counts.keys()):
            expected = expected_counts[method]
            actual_count = method_counts.get(method, 0)
            method_missing = expected - actual_count

            if method_missing > 0:
                print(f"  {method:35s}: {actual_count:3d} / {expected:3d} ({method_missing:3d} missing)")
