#!/usr/bin/env python3
"""Validate prior robustness experiment results

Checks:
1. All expected experiments completed
2. All JSON files have required fields (method_prior, true_prior)
3. Data isolation (no contamination in results/)
"""

import json
from pathlib import Path
from collections import defaultdict


def validate_results():
    """Validate robustness experiment completeness and data isolation"""

    # Expected counts
    datasets = ["MNIST", "FashionMNIST", "IMDB", "20News", "Mushrooms", "Spambase"]
    seeds = [42, 456, 789]
    c_values = [0.1, 0.5, 0.9]
    methods = ["vpu_nomixup", "vpu_nomixup_mean", "vpu_nomixup_mean_prior"]
    prior_values = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, "auto"]

    # Expected experiments per dataset/seed/c: 3 methods × 7 priors = 21
    # But vpu_nomixup and vpu_nomixup_mean don't use priors, so they appear once
    # vpu_nomixup_mean_prior appears 7 times (one per prior value)
    # So: 2 + 7 = 9 method runs per (dataset, seed, c) combination

    expected_total = len(datasets) * len(seeds) * len(c_values) * 9  # 486

    print("=" * 80)
    print("Prior Robustness Results Validation")
    print("=" * 80)
    print(f"\nExpected configuration:")
    print(f"  Datasets: {len(datasets)} ({', '.join(datasets)})")
    print(f"  Seeds: {len(seeds)} ({seeds})")
    print(f"  C values: {len(c_values)} ({c_values})")
    print(f"  Methods: {len(methods)}")
    print(f"  Prior values: {len(prior_values)}")
    print(f"  Expected method runs: {expected_total}")
    print()

    # Check results_robustness/
    results_dir = Path("results_robustness")
    if not results_dir.exists():
        print(f"✗ {results_dir} does not exist!")
        return

    json_files = list(results_dir.glob("seed_*/*methodprior*.json"))
    print(f"Found {len(json_files)} result files in {results_dir}/")

    if len(json_files) == 0:
        print("✗ No result files found!")
        return

    # Validate each file
    valid_files = 0
    missing_fields = []
    prior_errors = defaultdict(int)

    for json_file in json_files:
        try:
            with open(json_file) as f:
                data = json.load(f)

            # Check for required fields in each method's run data
            # JSON structure: {experiment, updated_at, runs: {method_name: {hyperparameters, dataset, ...}}}
            file_valid = False

            for method_name, method_data in data.get('runs', {}).items():
                hyperparams = method_data.get("hyperparameters", {})
                dataset_info = method_data.get("dataset", {})

                has_method_prior = "method_prior" in hyperparams
                has_true_prior = "prior" in dataset_info.get("train", {})

                if has_method_prior and has_true_prior:
                    file_valid = True

                    # Calculate prior error
                    method_prior = hyperparams["method_prior"]
                    true_prior = dataset_info["train"]["prior"]

                    if method_prior is not None and method_prior != "auto":
                        error = abs(method_prior - true_prior)
                        prior_errors[f"{json_file.name}:{method_name}"] = error

                    break  # Just check one method per file
                else:
                    if not has_method_prior:
                        missing_fields.append((json_file.name, "method_prior"))
                    if not has_true_prior:
                        missing_fields.append((json_file.name, "true_prior"))

            if file_valid:
                valid_files += 1

        except Exception as e:
            print(f"✗ Error reading {json_file}: {e}")

    print(f"\n✓ Valid files: {valid_files}/{len(json_files)}")

    if missing_fields:
        print(f"\n✗ Files with missing fields: {len(missing_fields)}")
        for fname, field in missing_fields[:10]:
            print(f"  - {fname}: missing {field}")
        if len(missing_fields) > 10:
            print(f"  ... and {len(missing_fields) - 10} more")
    else:
        print("✓ All files have required fields (method_prior, true_prior)")

    # Check prior error distribution
    if prior_errors:
        errors = sorted(prior_errors.values())
        print(f"\nPrior error statistics ({len(errors)} experiments with misspecified priors):")
        print(f"  Min: {min(errors):.3f}")
        print(f"  Max: {max(errors):.3f}")
        print(f"  Mean: {sum(errors)/len(errors):.3f}")

    # Check data isolation
    print("\n" + "=" * 80)
    print("Data Isolation Check")
    print("=" * 80)

    original_results = Path("results")
    if original_results.exists():
        contamination = list(original_results.glob("seed_*/*methodprior*.json"))
        if contamination:
            print(f"✗ CONTAMINATION DETECTED: Found {len(contamination)} robustness files in results/")
            for f in contamination[:5]:
                print(f"  - {f}")
        else:
            print("✓ Data isolation verified: no robustness files in results/")
    else:
        print("⚠ results/ directory does not exist")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    coverage_pct = (valid_files / expected_total * 100) if expected_total > 0 else 0
    print(f"Completion: {valid_files}/{expected_total} ({coverage_pct:.1f}%)")

    if valid_files == expected_total and not missing_fields:
        print("✓ All robustness experiments completed successfully!")
    else:
        print(f"⚠ Incomplete: {expected_total - valid_files} experiments remaining")


if __name__ == "__main__":
    validate_results()
