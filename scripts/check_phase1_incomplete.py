#!/usr/bin/env python3
"""
Check Phase 1 Extended for incomplete method runs.
"""

import json
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("results_phase1_extended")

# Expected methods per configuration (7 datasets × 10 seeds × 3 c values = 210)
# Base methods without mean_prior variants
BASE_METHODS = [
    "nnpu", "nnpusb", "bbepu", "lbe", "puet", "distpu", "pulda", "selfpu",
    "p3mixe", "p3mixc", "robustpu", "holisticpu", "lagam", "pulcpbf",
    "vaepu", "pan", "cgenpu", "pn_naive", "oracle_bce",
    "vpu", "vpu_nomixup"
]

# Mean-prior methods with 5 prior values each
MEAN_PRIOR_METHODS = ["vpu_mean_prior", "vpu_nomixup_mean_prior"]
PRIORS = ["auto", "0.353", "0.5", "0.69", "1.0"]

def check_method_complete(method_data):
    """Check if a method run is complete."""
    if "best" not in method_data:
        return False, "Missing 'best' field"

    if "metrics" not in method_data.get("best", {}):
        return False, "Missing 'metrics' in best"

    metrics = method_data["best"]["metrics"]
    if "test_auc" not in metrics:
        return False, "Missing 'test_auc' metric"

    return True, None

def check_incomplete_files():
    """Check all Phase 1 Extended files for incomplete method runs."""

    print("=" * 80)
    print("PHASE 1 EXTENDED: CHECKING FOR INCOMPLETE METHOD RUNS")
    print("=" * 80)
    print()

    incomplete_files = []
    incomplete_methods = defaultdict(list)
    total_files = 0
    total_incomplete = 0

    for json_file in RESULTS_DIR.glob("seed_*/**.json"):
        total_files += 1

        try:
            with open(json_file) as f:
                data = json.load(f)

            runs = data.get("runs", {})
            file_incomplete = False

            for method, method_data in runs.items():
                is_complete, reason = check_method_complete(method_data)

                if not is_complete:
                    file_incomplete = True
                    incomplete_methods[method].append({
                        "file": str(json_file.relative_to(RESULTS_DIR)),
                        "reason": reason
                    })

            if file_incomplete:
                incomplete_files.append(json_file.relative_to(RESULTS_DIR))
                total_incomplete += 1

        except Exception as e:
            print(f"Error reading {json_file.name}: {e}")
            incomplete_files.append(json_file.relative_to(RESULTS_DIR))
            total_incomplete += 1

    # Report
    print(f"Total files checked: {total_files:,}")
    print(f"Incomplete files: {total_incomplete:,}")
    print()

    if total_incomplete == 0:
        print("✅ All Phase 1 Extended files are complete!")
        return

    print("⚠️  INCOMPLETE FILES FOUND")
    print()

    # Show incomplete methods summary
    print("Incomplete Methods by Type:")
    print("-" * 80)

    for method in sorted(incomplete_methods.keys()):
        count = len(incomplete_methods[method])
        print(f"  {method:30s}: {count:3d} incomplete runs")

        # Show first few reasons
        reasons = {}
        for item in incomplete_methods[method]:
            reason = item["reason"]
            if reason not in reasons:
                reasons[reason] = []
            reasons[reason].append(item["file"])

        for reason, files in list(reasons.items())[:3]:
            print(f"    - {reason}: {len(files)} files")
            for f in files[:2]:
                print(f"      • {f}")

    print()
    print("=" * 80)
    print(f"ACTION REQUIRED: {total_incomplete} files need to be re-run")
    print("=" * 80)

if __name__ == "__main__":
    check_incomplete_files()
