#!/usr/bin/env python3
"""Count method runs in Phase 1 Extended (not just files)."""

import json
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("results_phase1_extended")

# Expected dimensions
DATASETS = ["MNIST", "FashionMNIST", "IMDB", "20News", "Mushrooms", "Spambase", "Connect4"]
SEEDS = [42, 456, 789, 1024, 2048, 3000, 4096, 5555, 6789, 8192]
C_VALUES = [0.1, 0.3, 0.5]

# Base methods (run with method_prior=null): 21 methods
BASE_METHODS = [
    "nnpu", "nnpusb", "bbepu", "lbe", "puet", "distpu", "pulda", "selfpu",
    "p3mixe", "p3mixc", "robustpu", "holisticpu", "lagam", "pulcpbf",
    "vaepu", "pan", "cgenpu", "pn_naive", "oracle_bce",
    "vpu", "vpu_nomixup"
]

# Mean-prior methods: 2 methods × 5 priors = 10 variants
MEAN_PRIOR_METHODS = ["vpu_mean_prior", "vpu_nomixup_mean_prior"]
PRIORS = ["auto", "0.353", "0.5", "0.69", "1.0"]

# Expected: 7 datasets × 10 seeds × 3 c × (21 base + 10 mean-prior) = 6,510
EXPECTED_TOTAL = len(DATASETS) * len(SEEDS) * len(C_VALUES) * (len(BASE_METHODS) + len(MEAN_PRIOR_METHODS) * len(PRIORS))

def count_actual_method_runs():
    """Count actual method runs by parsing JSON files."""

    method_runs = set()
    method_counts = defaultdict(int)
    configs = set()

    for seed_dir in RESULTS_DIR.glob("seed_*"):
        seed = int(seed_dir.name.split("_")[1])

        for json_file in seed_dir.glob("*.json"):
            # Parse filename to extract config
            parts = json_file.stem.split("_")
            dataset = parts[0]

            # Find c value
            c_str = [p for p in parts if p.startswith("c") and len(p) > 1 and p[1].isdigit()]
            if not c_str:
                continue
            c = float(c_str[0][1:])

            configs.add((dataset, seed, c))

            try:
                with open(json_file) as f:
                    data = json.load(f)

                for method_key, method_data in data.get("runs", {}).items():
                    # For mean-prior methods, distinguish by prior value
                    if method_key in MEAN_PRIOR_METHODS:
                        hyperparams = method_data.get("hyperparameters", {})
                        method_prior = hyperparams.get("method_prior")

                        # Check filename for methodprior indicator
                        is_auto = "methodprior_auto" in json_file.name

                        if is_auto or method_prior == "auto":
                            full_method_name = f"{method_key}_auto"
                        elif method_prior is None:
                            full_method_name = method_key
                        else:
                            full_method_name = f"{method_key}_{method_prior}"
                    else:
                        full_method_name = method_key

                    run_id = (dataset, seed, c, full_method_name)
                    method_runs.add(run_id)
                    method_counts[full_method_name] += 1

            except Exception as e:
                continue

    return method_runs, method_counts, configs

print("=" * 80)
print("PHASE 1 EXTENDED: METHOD RUN COUNT")
print("=" * 80)
print()

method_runs, method_counts, configs = count_actual_method_runs()

actual = len(method_runs)
missing = EXPECTED_TOTAL - actual
pct = actual * 100.0 / EXPECTED_TOTAL if EXPECTED_TOTAL > 0 else 0

print(f"Expected: {EXPECTED_TOTAL:,} method runs")
print(f"Actual:   {actual:,} method runs")
print(f"Missing:  {missing:,} method runs")
print(f"Progress: {pct:.1f}%")
print()

# Expected counts per method
expected_per_base = len(DATASETS) * len(SEEDS) * len(C_VALUES)  # 7 × 10 × 3 = 210
expected_per_mp_variant = expected_per_base  # 210

print("=" * 80)
print("Method Breakdown:")
print("=" * 80)
print()

# Check base methods
print("Base Methods (expected 210 each):")
for method in sorted(BASE_METHODS):
    actual_count = method_counts.get(method, 0)
    method_missing = expected_per_base - actual_count
    status = "✓" if method_missing == 0 else f"✗ {method_missing} missing"
    print(f"  {method:25s}: {actual_count:3d} / {expected_per_base:3d}  {status}")

print()
print("Mean-Prior Methods (expected 210 each):")
for method in sorted(MEAN_PRIOR_METHODS):
    for prior in PRIORS:
        if prior == "auto":
            full_name = f"{method}_auto"
        else:
            full_name = f"{method}_{prior}"

        actual_count = method_counts.get(full_name, 0)
        method_missing = expected_per_mp_variant - actual_count
        status = "✓" if method_missing == 0 else f"✗ {method_missing} missing"
        print(f"  {full_name:35s}: {actual_count:3d} / {expected_per_mp_variant:3d}  {status}")

print()
print("=" * 80)
if missing == 0:
    print("✅ Phase 1 Extended is COMPLETE!")
else:
    print(f"⚠️  Phase 1 Extended is INCOMPLETE: {missing:,} method runs missing")
print("=" * 80)
