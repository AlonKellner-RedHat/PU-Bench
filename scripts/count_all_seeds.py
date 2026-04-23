#!/usr/bin/env python3
"""Count method runs for all Phase 3 seeds."""

import json
from pathlib import Path
from collections import defaultdict

def count_actual_method_runs(seed):
    """Count actual method runs by parsing JSON files."""

    seed_dir = Path(f"results_phase3/seed_{seed}")
    if not seed_dir.exists():
        return set(), defaultdict(int), set()

    method_runs = set()
    method_counts = defaultdict(int)
    configs = set()

    for json_file in seed_dir.glob("*.json"):
        is_auto = "methodprior_auto" in json_file.name

        parts = json_file.stem.split("_")
        dataset = parts[0]

        c_str = [p for p in parts if p.startswith("c") and len(p) > 1 and p[1].isdigit()][0]
        c = float(c_str[1:])

        pi_str = [p for p in parts if p.startswith("trueprior")][0]
        pi = float(pi_str[9:])

        configs.add((dataset, c, pi))

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

                run_id = (dataset, c, pi, full_method_name)
                method_runs.add(run_id)
                method_counts[full_method_name] += 1

        except Exception as e:
            continue

    return method_runs, method_counts, configs

EXPECTED_CONFIGS = 343
EXPECTED_METHODS = 14
EXPECTED_TOTAL = 4802

print("=" * 80)
print("PHASE 3 ALL SEEDS: METHOD RUN COUNT")
print("=" * 80)
print()

total_actual = 0
total_missing = 0

for seed in [42, 456, 789, 1024, 2048]:
    method_runs, method_counts, configs = count_actual_method_runs(seed)

    actual = len(method_runs)
    missing = EXPECTED_TOTAL - actual
    pct = actual * 100.0 / EXPECTED_TOTAL

    total_actual += actual
    total_missing += missing

    status = "✓ COMPLETE" if missing == 0 else f"✗ {missing} missing"
    print(f"Seed {seed:4d}: {actual:4,} / {EXPECTED_TOTAL:,} ({pct:5.1f}%)  {status}")

print()
print("=" * 80)
print(f"TOTAL ACROSS ALL SEEDS:")
total_expected = EXPECTED_TOTAL * 5
pct = total_actual * 100.0 / total_expected
print(f"  Actual:   {total_actual:,} / {total_expected:,} ({pct:.1f}%)")
print(f"  Missing:  {total_missing:,} method runs")
print("=" * 80)
