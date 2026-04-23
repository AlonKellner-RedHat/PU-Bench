#!/usr/bin/env python3
"""
Count actual method runs in Phase 3 results by parsing JSON files.
Each file may contain multiple method runs.
"""

import json
from pathlib import Path
from collections import defaultdict

def count_method_runs(results_dir, seed):
    """Count actual method runs by parsing all JSON files for a seed."""

    seed_dir = Path(results_dir) / f"seed_{seed}"

    # Track method runs by (dataset, c, pi, method)
    method_runs = set()

    # Also track by method name for summary
    method_counts = defaultdict(int)

    for json_file in seed_dir.glob("*.json"):
        # Parse filename to get dataset, c, pi
        parts = json_file.stem.split("_")

        # Dataset name
        dataset = parts[0]

        # Extract c value
        c_str = [p for p in parts if p.startswith("c") and p[1:2].isdigit()][0]
        c = float(c_str[1:])

        # Extract true prior
        pi_str = [p for p in parts if p.startswith("trueprior")][0]
        pi = float(pi_str[9:])

        # Load JSON and count methods
        try:
            with open(json_file) as f:
                data = json.load(f)

            for method_key in data.get("runs", {}).keys():
                # Create unique identifier for this method run
                run_id = (dataset, c, pi, method_key)
                method_runs.add(run_id)
                method_counts[method_key] += 1

        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            continue

    return method_runs, method_counts

def main():
    results_dir = Path("results_phase3")
    seed = 42

    print(f"Counting method runs for seed {seed}...")
    print()

    method_runs, method_counts = count_method_runs(results_dir, seed)

    print(f"Actual Method Runs by Method:")
    print("=" * 60)

    # Group by method type
    base_methods = ["vpu", "vpu_nomixup", "oracle_bce", "pn_naive"]
    mean_prior_methods = ["vpu_mean_prior", "vpu_nomixup_mean_prior"]

    print("\nBase Methods (expected: 343 each):")
    for method in base_methods:
        count = method_counts.get(method, 0)
        print(f"  {method:25s}: {count:3d} / 343")

    print("\nMean-Prior Methods (expected: 343 each for all 5 prior variants):")
    for method in mean_prior_methods:
        count = method_counts.get(method, 0)
        print(f"  {method:25s}: {count:3d} / 1,715 (all priors combined)")

    print("\nTotal method runs:")
    total_actual = len(method_runs)
    total_expected = 343 * 14  # 343 configs × 14 methods
    print(f"  Actual:   {total_actual:,}")
    print(f"  Expected: {total_expected:,}")
    print(f"  Missing:  {total_expected - total_actual:,}")
    print()

    # Calculate missing by method
    print("Missing Method Runs:")
    print("=" * 60)

    # For base methods
    base_missing = 0
    for method in base_methods:
        expected = 343
        actual = method_counts.get(method, 0)
        missing = expected - actual
        if missing > 0:
            print(f"  {method:25s}: {missing} missing")
            base_missing += missing

    # For mean-prior methods, need to check each prior variant
    # by looking at unique configurations
    configs_by_method = defaultdict(set)
    for (dataset, c, pi, method_key) in method_runs:
        configs_by_method[method_key].add((dataset, c, pi))

    mean_prior_missing = 0
    for base_method in mean_prior_methods:
        print(f"\n  {base_method}:")
        for prior in ["auto", "0.353", "0.5", "0.69", "1.0"]:
            # Count how many configs have this method with this prior
            # This is tricky - need to look at actual method runs
            # For now, just report what we know from file counts
            pass

    print()

    # Better breakdown by checking file counts
    print("Breakdown by File Type:")
    print("=" * 60)

    seed_dir = Path("results_phase3/seed_42")

    base_files = len(list(seed_dir.glob("*.json")))
    base_files_no_mp = len([f for f in seed_dir.glob("*.json") if "methodprior" not in f.name])

    print(f"Base method files (no methodprior): {base_files_no_mp} / 343")
    print(f"  Each contains 4 methods × 1 config = 4 method runs")
    print(f"  Total base method runs: {base_files_no_mp * 4} / {343 * 4}")
    print()

    for prior in ["auto", "0.353", "0.5", "0.69", "1.0"]:
        if prior == "auto":
            pattern = "*methodprior_auto.json"
        else:
            pattern = f"*methodprior{prior}.json"

        count = len(list(seed_dir.glob(pattern)))
        print(f"methodprior={prior}: {count} / 343 files")
        print(f"  Each contains 2 methods × 1 config = 2 method runs")
        print(f"  Total method runs: {count * 2} / {343 * 2}")
        print()

    print("Summary:")
    print("=" * 60)
    print(f"Total actual method runs: {total_actual:,}")
    print(f"Total expected: {total_expected:,}")
    print(f"MISSING: {total_expected - total_actual:,} method runs")

if __name__ == "__main__":
    main()
