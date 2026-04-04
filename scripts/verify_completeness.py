#!/usr/bin/env python3
"""Verify all expected experiments completed successfully"""

import json
from pathlib import Path
from collections import defaultdict

# Expected experiment counts per method
expected_per_method = {
    "vpu": 675,  # 9 datasets × 5 seeds × 3 c × 5 true_priors × 1 = 675
    "vpu_nomixup": 675,
    "vpu_mean_prior": 2025,  # 9 × 5 × 3 × 5 × 3 (method_prior: 0.5, auto, 1.0) = 2025
    "vpu_nomixup_mean_prior": 2025,
    "nnpu": 675,
    "distpu": 675,
}

expected_total = sum(expected_per_method.values())  # 6,750

results_dir = Path("results_comprehensive")

# Count actual results per method
actual_counts = defaultdict(int)
all_files = list(results_dir.glob("seed_*/*.json"))

print(f"Scanning {len(all_files)} result files...")

for json_file in all_files:
    try:
        with open(json_file) as f:
            data = json.load(f)

        for method in data.get("runs", {}):
            actual_counts[method] += 1
    except Exception as e:
        print(f"⚠ Error reading {json_file}: {e}")

# Compare actual vs expected
print("\n" + "="*80)
print("COMPLETENESS CHECK")
print("="*80)

all_complete = True
for method, expected in sorted(expected_per_method.items()):
    actual = actual_counts.get(method, 0)
    status = "✓" if actual == expected else "✗"
    pct = (actual / expected * 100) if expected > 0 else 0

    print(f"{status} {method:25s}: {actual:5d} / {expected:5d} ({pct:6.2f}%)")

    if actual != expected:
        all_complete = False

actual_total = sum(actual_counts.values())
print("-" * 80)
print(f"{'✓' if actual_total == expected_total else '✗'} {'TOTAL':25s}: {actual_total:5d} / {expected_total:5d} ({actual_total/expected_total*100:6.2f}%)")

if all_complete:
    print("\n✓ All experiments completed successfully!")
    exit(0)
else:
    print("\n✗ Some experiments are missing. Check logs for errors.")
    exit(1)
