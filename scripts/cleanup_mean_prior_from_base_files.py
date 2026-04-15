#!/usr/bin/env python3
"""
Remove mean_prior method runs from base files (files without methodprior suffix).

Mean_prior methods should only appear in methodprior_auto and methodprior0.5 files,
not in base files. This script cleans up the incorrect runs that were created due
to the filtering bug.
"""

import json
from pathlib import Path

def cleanup_mean_prior_from_base_files(results_dir="results_phase1_extended", dry_run=False):
    """
    Remove mean_prior methods from base files.

    Args:
        results_dir: Results directory to clean
        dry_run: If True, only report what would be done without making changes
    """
    results_path = Path(results_dir)

    print("=" * 70)
    print("CLEANUP: Remove mean_prior methods from base files")
    print("=" * 70)
    print()
    print(f"Results directory: {results_dir}")
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'LIVE (will modify files)'}")
    print()

    modified_count = 0
    removed_methods_count = 0

    # Find all base files (no methodprior in filename)
    base_files = [f for f in results_path.glob("seed_*/*.json") if "methodprior" not in f.name]

    print(f"Found {len(base_files)} base files")
    print()

    for json_file in base_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            runs = data.get("runs", {})
            mean_prior_methods = [m for m in runs.keys() if "mean_prior" in m]

            if mean_prior_methods:
                # Remove mean_prior methods
                for method in mean_prior_methods:
                    if not dry_run:
                        del runs[method]
                    removed_methods_count += 1

                # Save modified file
                if not dry_run:
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)

                print(f"  {'[DRY RUN] ' if dry_run else ''}Modified: {json_file.name}")
                print(f"    Removed methods: {mean_prior_methods}")
                modified_count += 1

        except Exception as e:
            print(f"  ERROR: {json_file.name}")
            print(f"    {str(e)}")

    print()
    print("=" * 70)
    print("CLEANUP SUMMARY")
    print("=" * 70)
    print(f"  Modified files: {modified_count}")
    print(f"  Removed method runs: {removed_methods_count}")
    print()

    if dry_run:
        print("NOTE: This was a dry run. Run with --live to apply changes.")
    else:
        print("✅ Cleanup complete!")

    return modified_count, removed_methods_count


if __name__ == "__main__":
    import sys

    dry_run = True
    if "--live" in sys.argv:
        dry_run = False

    cleanup_mean_prior_from_base_files(dry_run=dry_run)
