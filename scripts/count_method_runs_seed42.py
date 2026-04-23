#!/usr/bin/env python3
"""
Count ACTUAL method runs for Phase 3 Seed 42 by parsing JSON files.
Each file may contain multiple method runs.
"""

import json
from pathlib import Path
from collections import defaultdict

def count_actual_method_runs(seed):
    """Count actual method runs by parsing JSON files."""

    seed_dir = Path(f"results_phase3/seed_{seed}")

    # Track unique method runs: (dataset, c, pi, method_name)
    method_runs = set()

    # Also count by method name
    method_counts = defaultdict(int)

    # Count by configuration
    configs = set()

    for json_file in seed_dir.glob("*.json"):
        # Check if this is a methodprior_auto file
        is_auto = "methodprior_auto" in json_file.name
        # Parse filename
        parts = json_file.stem.split("_")
        dataset = parts[0]

        # Extract c
        c_str = [p for p in parts if p.startswith("c") and len(p) > 1 and p[1].isdigit()][0]
        c = float(c_str[1:])

        # Extract pi
        pi_str = [p for p in parts if p.startswith("trueprior")][0]
        pi = float(pi_str[9:])

        configs.add((dataset, c, pi))

        # Load JSON and count methods
        try:
            with open(json_file) as f:
                data = json.load(f)

            for method_key, method_data in data.get("runs", {}).items():
                # For mean-prior methods, get the actual prior value from hyperparameters
                if method_key in ["vpu_mean_prior", "vpu_nomixup_mean_prior"]:
                    hyperparams = method_data.get("hyperparameters", {})
                    method_prior = hyperparams.get("method_prior")

                    # Create unique method identifier with prior
                    if is_auto or method_prior == "auto":
                        # Files with methodprior_auto have None in hyperparameters
                        full_method_name = f"{method_key}_auto"
                    elif method_prior is None:
                        full_method_name = method_key  # Shouldn't happen for mean_prior methods
                    else:
                        full_method_name = f"{method_key}_{method_prior}"
                else:
                    full_method_name = method_key

                # Create unique identifier for this method run
                run_id = (dataset, c, pi, full_method_name)
                method_runs.add(run_id)
                method_counts[full_method_name] += 1

        except Exception as e:
            print(f"Error reading {json_file.name}: {e}")
            continue

    return method_runs, method_counts, configs

# Expected cartesian product
EXPECTED_CONFIGS = 7 * 7 * 7  # 7 datasets × 7 c × 7 π = 343
EXPECTED_METHODS = 14  # 4 base + 10 mean-prior variants
EXPECTED_TOTAL = EXPECTED_CONFIGS * EXPECTED_METHODS  # 343 × 14 = 4,802

print("=" * 70)
print("PHASE 3 SEED 42: ACTUAL METHOD RUN COUNT")
print("=" * 70)
print()

method_runs, method_counts, configs = count_actual_method_runs(42)

print(f"Configurations found: {len(configs)} / {EXPECTED_CONFIGS}")
print()

# List all method names found
all_methods = sorted(method_counts.keys())

print("Method Runs by Method Name:")
print("-" * 70)

# Group by type
base_methods = ["vpu", "vpu_nomixup", "oracle_bce", "pn_naive"]
for method in base_methods:
    count = method_counts.get(method, 0)
    status = "✓" if count == EXPECTED_CONFIGS else f"✗ MISSING {EXPECTED_CONFIGS - count}"
    print(f"  {method:30s}: {count:4d} / {EXPECTED_CONFIGS:4d}  {status}")

print()
print("Mean-Prior Method Runs (vpu_mean_prior):")
for prior in ["auto", "0.353", "0.5", "0.69", "1.0"]:
    method = f"vpu_mean_prior_{prior}"
    count = method_counts.get(method, 0)
    status = "✓" if count == EXPECTED_CONFIGS else f"✗ MISSING {EXPECTED_CONFIGS - count}"
    print(f"  {method:30s}: {count:4d} / {EXPECTED_CONFIGS:4d}  {status}")

print()
print("Mean-Prior Method Runs (vpu_nomixup_mean_prior):")
for prior in ["auto", "0.353", "0.5", "0.69", "1.0"]:
    method = f"vpu_nomixup_mean_prior_{prior}"
    count = method_counts.get(method, 0)
    status = "✓" if count == EXPECTED_CONFIGS else f"✗ MISSING {EXPECTED_CONFIGS - count}"
    print(f"  {method:30s}: {count:4d} / {EXPECTED_CONFIGS:4d}  {status}")

print()
print("=" * 70)
print(f"TOTAL METHOD RUNS: {len(method_runs):,} / {EXPECTED_TOTAL:,}")
missing = EXPECTED_TOTAL - len(method_runs)
print(f"MISSING: {missing:,} method runs")
print("=" * 70)

if missing == 0:
    print()
    print("✅ Seed 42 is COMPLETE! All method runs present.")
else:
    print()
    print(f"⚠️  Seed 42 is INCOMPLETE. Missing {missing} method runs.")
    print()

    # Find which methods are missing
    print("Missing method runs by method:")
    for method in all_methods:
        expected = EXPECTED_CONFIGS
        actual = method_counts.get(method, 0)
        if actual < expected:
            print(f"  {method}: {expected - actual} missing")
