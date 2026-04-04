#!/usr/bin/env python3
"""Verify all results have oracle_ce metric"""

import json
from pathlib import Path

results_dir = Path("results_comprehensive")
all_files = list(results_dir.glob("seed_*/*.json"))

missing_oracle_ce = []

print(f"Checking {len(all_files)} result files for oracle_ce...")

for json_file in all_files:
    try:
        with open(json_file) as f:
            data = json.load(f)

        for method, run_data in data.get("runs", {}).items():
            metrics = run_data.get("best", {}).get("metrics", {})

            # Check if oracle_ce is present and not None/NaN
            if "test_oracle_ce" not in metrics:
                missing_oracle_ce.append((str(json_file), method, "missing"))
            elif metrics["test_oracle_ce"] is None:
                missing_oracle_ce.append((str(json_file), method, "None"))
            elif str(metrics["test_oracle_ce"]).lower() == "nan":
                missing_oracle_ce.append((str(json_file), method, "NaN"))
    except Exception as e:
        print(f"⚠ Error reading {json_file}: {e}")

print("\n" + "="*80)
print("ORACLE CE VERIFICATION")
print("="*80)

if missing_oracle_ce:
    print(f"\n✗ Found {len(missing_oracle_ce)} runs with missing/invalid oracle_ce:")
    for file, method, issue in missing_oracle_ce[:20]:  # Show first 20
        print(f"  - {file}: {method} ({issue})")

    if len(missing_oracle_ce) > 20:
        print(f"  ... and {len(missing_oracle_ce) - 20} more")

    exit(1)
else:
    print("\n✓ All runs have valid oracle_ce metric!")
    exit(0)
