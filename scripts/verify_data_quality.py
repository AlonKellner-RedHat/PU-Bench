#!/usr/bin/env python3
"""Verify data quality - check for NaN, inf, out-of-range values"""

import json
import math
from pathlib import Path
from collections import defaultdict

results_dir = Path("results_comprehensive")
all_files = list(results_dir.glob("seed_*/*.json"))

issues = defaultdict(list)

print(f"Checking {len(all_files)} result files for data quality...")

for json_file in all_files:
    try:
        with open(json_file) as f:
            data = json.load(f)

        for method, run_data in data.get("runs", {}).items():
            metrics = run_data.get("best", {}).get("metrics", {})

            # Check key metrics for validity
            checks = [
                ("test_ap", 0.0, 1.0),
                ("test_auc", 0.0, 1.0),
                ("test_f1", 0.0, 1.0),
                ("test_accuracy", 0.0, 1.0),
                ("test_precision", 0.0, 1.0),
                ("test_recall", 0.0, 1.0),
                ("test_ece", 0.0, 1.0),
                ("test_mce", 0.0, 1.0),
                ("test_brier", 0.0, 2.0),  # Brier can be > 1 in edge cases
            ]

            for metric_name, min_val, max_val in checks:
                value = metrics.get(metric_name)

                if value is None:
                    continue  # Skip missing metrics

                # Check for NaN
                if isinstance(value, float) and math.isnan(value):
                    issues["nan"].append((str(json_file), method, metric_name))

                # Check for inf
                elif isinstance(value, float) and math.isinf(value):
                    issues["inf"].append((str(json_file), method, metric_name))

                # Check for out of range
                elif not (min_val <= value <= max_val):
                    issues["out_of_range"].append((str(json_file), method, metric_name, value))

    except Exception as e:
        issues["read_error"].append((str(json_file), str(e)))

print("\n" + "="*80)
print("DATA QUALITY CHECK")
print("="*80)

has_issues = False

if issues["nan"]:
    print(f"\n✗ Found {len(issues['nan'])} NaN values:")
    for file, method, metric in issues["nan"][:10]:
        print(f"  - {file}: {method}.{metric}")
    has_issues = True

if issues["inf"]:
    print(f"\n✗ Found {len(issues['inf'])} Inf values:")
    for file, method, metric in issues["inf"][:10]:
        print(f"  - {file}: {method}.{metric}")
    has_issues = True

if issues["out_of_range"]:
    print(f"\n✗ Found {len(issues['out_of_range'])} out-of-range values:")
    for file, method, metric, value in issues["out_of_range"][:10]:
        print(f"  - {file}: {method}.{metric} = {value}")
    has_issues = True

if issues["read_error"]:
    print(f"\n✗ Found {len(issues['read_error'])} read errors:")
    for file, error in issues["read_error"][:10]:
        print(f"  - {file}: {error}")
    has_issues = True

if not has_issues:
    print("\n✓ All data quality checks passed!")
    exit(0)
else:
    exit(1)
