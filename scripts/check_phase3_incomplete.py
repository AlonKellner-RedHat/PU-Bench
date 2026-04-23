#!/usr/bin/env python3
"""
Check Phase 3 for incomplete method runs using the same logic as fixed --resume.
"""

import json
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("results_phase3")

def check_method_complete(method_data):
    """Check if a method run is complete (same logic as fixed _method_already_completed)."""
    if "best" not in method_data:
        return False, "Missing 'best' field"

    if "metrics" not in method_data.get("best", {}):
        return False, "Missing 'metrics' in best"

    metrics = method_data["best"]["metrics"]
    if "test_auc" not in metrics:
        return False, "Missing 'test_auc' metric"

    return True, None

def check_incomplete_files():
    """Check all Phase 3 files for incomplete method runs."""

    print("=" * 80)
    print("PHASE 3: CHECKING FOR INCOMPLETE METHOD RUNS")
    print("=" * 80)
    print()

    incomplete_by_seed = defaultdict(list)
    incomplete_methods_by_seed = defaultdict(lambda: defaultdict(int))
    total_files_by_seed = defaultdict(int)
    total_incomplete_by_seed = defaultdict(int)

    for seed in [42, 456, 789, 1024, 2048]:
        seed_dir = RESULTS_DIR / f"seed_{seed}"
        if not seed_dir.exists():
            continue

        for json_file in seed_dir.glob("*.json"):
            total_files_by_seed[seed] += 1

            try:
                with open(json_file) as f:
                    data = json.load(f)

                runs = data.get("runs", {})
                file_has_incomplete = False
                incomplete_methods_in_file = []

                for method, method_data in runs.items():
                    is_complete, reason = check_method_complete(method_data)

                    if not is_complete:
                        file_has_incomplete = True
                        incomplete_methods_in_file.append((method, reason))
                        incomplete_methods_by_seed[seed][method] += 1

                if file_has_incomplete:
                    incomplete_by_seed[seed].append({
                        "file": json_file.name,
                        "methods": incomplete_methods_in_file
                    })
                    total_incomplete_by_seed[seed] += 1

            except Exception as e:
                print(f"Error reading {json_file.name}: {e}")
                total_incomplete_by_seed[seed] += 1

    # Report
    print("Results by Seed:")
    print("-" * 80)

    all_complete = True
    for seed in [42, 456, 789, 1024, 2048]:
        total = total_files_by_seed[seed]
        incomplete = total_incomplete_by_seed[seed]
        status = "✓" if incomplete == 0 else "✗"

        print(f"{status} Seed {seed}: {total:4d} files, {incomplete:4d} incomplete")

        if incomplete > 0:
            all_complete = False
            # Show incomplete methods summary for this seed
            for method, count in sorted(incomplete_methods_by_seed[seed].items()):
                print(f"    - {method:30s}: {count:3d} incomplete runs")

    print()
    print("=" * 80)

    if all_complete:
        print("✅ All Phase 3 files are complete!")
    else:
        total_incomplete_files = sum(total_incomplete_by_seed.values())
        print(f"⚠️  ACTION REQUIRED: {total_incomplete_files} files have incomplete method runs")
        print()
        print("These files will be re-processed with the fixed --resume logic.")
        print("The fixed logic will:")
        print("  1. Detect incomplete methods in existing files")
        print("  2. Re-run only those incomplete methods")
        print("  3. Update the files with complete results")

    print("=" * 80)

    return all_complete

if __name__ == "__main__":
    check_incomplete_files()
