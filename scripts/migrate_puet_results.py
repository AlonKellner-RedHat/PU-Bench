#!/usr/bin/env python3
"""
Migrate PUET results from results/ to results_phase1_extended/.

This script finds all PUET method runs in the old results/ directory
and merges them into the corresponding JSON files in results_phase1_extended/.
"""

import json
import glob
import os
import shutil
from datetime import datetime

def migrate_puet_results(source_dir="results", target_dir="results_phase1_extended", dry_run=False):
    """
    Migrate PUET results from source_dir to target_dir.

    Args:
        source_dir: Source directory (default: "results")
        target_dir: Target directory (default: "results_phase1_extended")
        dry_run: If True, only report what would be done without making changes
    """
    print("=" * 70)
    print("PUET RESULTS MIGRATION")
    print("=" * 70)
    print()
    print(f"Source: {source_dir}/")
    print(f"Target: {target_dir}/")
    print(f"Mode:   {'DRY RUN (no changes)' if dry_run else 'LIVE (will modify files)'}")
    print()

    migrated_count = 0
    skipped_count = 0
    error_count = 0

    # Find all JSON files in source directory
    source_pattern = os.path.join(source_dir, "seed_*", "*.json")
    source_files = glob.glob(source_pattern)

    print(f"Found {len(source_files)} JSON files in {source_dir}/")
    print()

    for source_file in source_files:
        try:
            # Load source JSON
            with open(source_file, 'r', encoding='utf-8') as f:
                source_data = json.load(f)

            # Check if it has PUET runs
            if "puet" not in source_data.get("runs", {}):
                continue

            # Extract PUET run
            puet_run = source_data["runs"]["puet"]
            experiment_name = source_data.get("experiment", "")

            # Determine target file path
            # Extract seed from path: results/seed_42/Exp.json -> seed_42
            path_parts = source_file.split(os.sep)
            seed_dir = path_parts[-2]  # e.g., "seed_42"
            filename = path_parts[-1]   # e.g., "MNIST_case-control_random_c0.1_seed42.json"

            target_file = os.path.join(target_dir, seed_dir, filename)

            # Check if target directory exists
            target_seed_dir = os.path.join(target_dir, seed_dir)
            if not os.path.exists(target_seed_dir):
                if not dry_run:
                    os.makedirs(target_seed_dir, exist_ok=True)
                    print(f"  Created directory: {target_seed_dir}")

            # Load or create target JSON
            target_data = {
                "experiment": experiment_name,
                "updated_at": None,
                "runs": {}
            }

            if os.path.exists(target_file):
                with open(target_file, 'r', encoding='utf-8') as f:
                    target_data = json.load(f)

            # Check if PUET already exists in target
            if "puet" in target_data.get("runs", {}):
                print(f"  ⏭  SKIP: {experiment_name} (PUET already in target)")
                skipped_count += 1
                continue

            # Merge PUET run into target
            target_data["runs"]["puet"] = puet_run
            target_data["updated_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

            if not dry_run:
                # Write updated target file
                with open(target_file, 'w', encoding='utf-8') as f:
                    json.dump(target_data, f, ensure_ascii=False, indent=2)

            print(f"  ✅ MIGRATE: {experiment_name}")
            print(f"      From: {source_file}")
            print(f"      To:   {target_file}")
            migrated_count += 1

        except Exception as e:
            print(f"  ❌ ERROR: {source_file}")
            print(f"      {str(e)}")
            error_count += 1

    print()
    print("=" * 70)
    print("MIGRATION SUMMARY")
    print("=" * 70)
    print(f"  Migrated: {migrated_count} PUET runs")
    print(f"  Skipped:  {skipped_count} (already in target)")
    print(f"  Errors:   {error_count}")
    print()

    if dry_run:
        print("NOTE: This was a dry run. Run with --live to apply changes.")
    else:
        print("✅ Migration complete!")

    return migrated_count, skipped_count, error_count


if __name__ == "__main__":
    import sys

    dry_run = True
    if "--live" in sys.argv:
        dry_run = False

    migrate_puet_results(dry_run=dry_run)
