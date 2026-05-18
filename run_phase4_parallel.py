#!/usr/bin/env python3
"""Phase 4 runner with ThreadPoolExecutor + subprocess-based dynamic task queue.

Architecture:
- Main process creates dataloader groups
- ThreadPoolExecutor manages N worker threads
- Each thread pulls a group and spawns a subprocess to process it
- Subprocess processes all experiments in the group (cache benefit!)
- Subprocess exits, releasing all memory (no restart needed!)
- Thread pulls next group and repeats

This gives us:
- Dynamic load balancing (threads pull from queue)
- Memory leak isolation (subprocess per group)
- Cache efficiency (contiguous processing within group)
- No restart interruptions
"""

import sys
import json
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.run_param_sweep import load_dataset_config, expand_dataset_grid


def _build_experiment_name(dataset_class, data_cfg, method, target_prev_train, method_prior):
    """Build experiment name to check completion status."""
    c = data_cfg.get("labeled_ratio")
    scn = data_cfg.get("scenario")
    strat = data_cfg.get("selection_strategy")
    seed = data_cfg.get("random_seed")

    base_name = f"{dataset_class}_{scn}_{strat}_c{c:g}_seed{seed}"

    if target_prev_train is not None:
        base_name += f"_trueprior{target_prev_train:g}"

    if method_prior is not None:
        if method_prior == "auto":
            base_name += "_methodpriorauto"
        elif method_prior == "true":
            base_name += "_methodpriortrue"
        elif method_prior == "ep_linear":
            base_name += "_methodprioreplinear"
        else:
            base_name += f"_methodprior{method_prior:g}"

    return base_name


def _is_method_completed(exp_name, method, seed, output_dir):
    """Check if a method has already completed for this experiment."""
    result_file = Path(f"{output_dir}/seed_{seed}/{exp_name}.json")
    if not result_file.exists():
        return False

    try:
        with open(result_file, "r") as f:
            data = json.load(f)

        if method not in data.get("runs", {}):
            return False

        method_data = data["runs"][method]

        if "best" not in method_data:
            return False

        if "metrics" not in method_data.get("best", {}):
            return False

        metrics = method_data["best"]["metrics"]
        if "test_auc" not in metrics:
            return False

        return True
    except:
        return False


def create_dataloader_groups(config_paths, methods, output_dir):
    """Create groups of experiments sharing the same dataloader.

    Filters out already-completed experiments before grouping to avoid
    subprocess overhead for skipped work.
    """
    import copy

    all_experiments = []
    filtered_experiments = []
    total_generated = 0
    already_complete = 0

    # Load all dataset configs
    for cfg_path in config_paths:
        dataset_cfg = load_dataset_config(cfg_path)
        dataset_class, data_runs = expand_dataset_grid(dataset_cfg)

        for data_cfg in data_runs:
            target_prevalence_train_values = data_cfg.get("target_prevalence_train_values", [None])
            method_prior_values = data_cfg.get("method_prior_values", [None])

            for target_prev_train in target_prevalence_train_values:
                for method_prior in method_prior_values:
                    for method in methods:
                        # Methods with prior support
                        METHODS_WITH_PRIOR_SUPPORT = {
                            'vpu_mean_prior', 'vpu_nomixup_mean_prior',
                            'nnpu', 'nnpusb', 'lbe', 'distpu', 'selfpu',
                            'p3mixe', 'p3mixc', 'robustpu'
                        }

                        supports_method_prior = method in METHODS_WITH_PRIOR_SUPPORT
                        if supports_method_prior != (method_prior is not None):
                            continue

                        exp_config = {
                            "dataset_class": dataset_class,
                            "data_cfg": copy.deepcopy(data_cfg),
                            "target_prev_train": target_prev_train,
                            "method_prior": method_prior,
                            "method": method,
                            "seed": data_cfg.get("random_seed"),
                        }

                        total_generated += 1

                        # Check if already complete
                        exp_name = _build_experiment_name(
                            dataset_class, data_cfg, method,
                            target_prev_train, method_prior
                        )

                        if _is_method_completed(exp_name, method, data_cfg.get("random_seed"), output_dir):
                            already_complete += 1
                        else:
                            filtered_experiments.append(exp_config)

    # Report filtering results
    print(f"Experiment filtering:")
    print(f"  Total generated: {total_generated:,}")
    print(f"  Already complete: {already_complete:,} ({100*already_complete/total_generated:.1f}%)")
    print(f"  Remaining: {len(filtered_experiments):,}")

    # Group by dataloader configuration (only remaining experiments)
    dataloader_groups = defaultdict(list)
    for exp in filtered_experiments:
        group_key = (
            exp["dataset_class"],
            exp["data_cfg"]["labeled_ratio"],
            exp["data_cfg"]["random_seed"],
            exp.get("target_prev_train") or 0,
        )
        dataloader_groups[group_key].append(exp)

    groups = list(dataloader_groups.values())
    avg_per_group = len(filtered_experiments) / len(groups) if len(groups) > 0 else 0

    print(f"Created {len(groups)} dataloader groups from {len(filtered_experiments):,} remaining experiments")
    print(f"Average {avg_per_group:.1f} experiments per group")

    return groups


def process_group_in_subprocess(group_info):
    """Process a group of experiments in a subprocess."""
    group_idx, group_experiments, output_dir, methods_str = group_info

    # Get group description
    first_exp = group_experiments[0]
    group_desc = f"{first_exp['dataset_class']}_c{first_exp['data_cfg']['labeled_ratio']}_pi{first_exp.get('target_prev_train', 'N/A')}"

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Group {group_idx}: Starting {group_desc} ({len(group_experiments)} experiments)")

    # Write group to temp file
    temp_dir = Path(output_dir) / "temp_groups"
    temp_dir.mkdir(parents=True, exist_ok=True)
    group_file = temp_dir / f"group_{group_idx:04d}.json"

    with open(group_file, 'w') as f:
        json.dump(group_experiments, f)

    # Launch subprocess to process this group
    cmd = [
        sys.executable,
        "run_train.py",
        "--dataset-config", "DUMMY",  # Will be ignored, we load from group file
        "--methods", methods_str,
        "--output-dir", output_dir,
        "--resume",
        "--process-group", str(group_file)
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )

        # Clean up temp file
        group_file.unlink(missing_ok=True)

        if result.returncode == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Group {group_idx}: ✓ Completed {group_desc}")
            return group_idx, "success", None
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Group {group_idx}: ✗ Failed {group_desc}")
            print(f"  stderr: {result.stderr[:500]}")
            return group_idx, "failed", result.stderr

    except Exception as e:
        group_file.unlink(missing_ok=True)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Group {group_idx}: ✗ Error {group_desc}: {e}")
        return group_idx, "error", str(e)


def main():
    # Configuration
    OUTPUT_DIR = "results_phase4"
    NUM_WORKERS = 12

    CONFIGS = [
        "config/phase4/mnist_phase4.yaml",
        "config/phase4/fashionmnist_phase4.yaml",
        "config/phase4/imdb_phase4.yaml",
        "config/phase4/20news_phase4.yaml",
        "config/phase4/mushrooms_phase4.yaml",
        "config/phase4/spambase_phase4.yaml",
        "config/phase4/connect4_phase4.yaml"
    ]

    METHODS = [
        "nnpu", "nnpusb", "lbe", "distpu", "selfpu",
        "p3mixe", "p3mixc", "robustpu",
        "vpu_mean_prior", "vpu_nomixup_mean_prior"
    ]

    print("=" * 70)
    print("Phase 4: Prior Robustness with Dynamic Task Queue")
    print("=" * 70)
    print(f"Workers: {NUM_WORKERS} threads")
    print(f"Architecture: ThreadPoolExecutor + subprocess per group")
    print(f"Benefits: Dynamic load balancing + automatic memory cleanup")
    print("=" * 70)
    print()

    # Create groups
    groups = create_dataloader_groups(CONFIGS, METHODS, OUTPUT_DIR)

    # Prepare tasks
    methods_str = ",".join(METHODS)
    tasks = [
        (idx, group, OUTPUT_DIR, methods_str)
        for idx, group in enumerate(groups)
    ]

    # Process with ThreadPoolExecutor
    start_time = datetime.now()
    completed = 0
    failed = 0

    print(f"\nProcessing {len(tasks)} groups with {NUM_WORKERS} workers...")
    print()

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_group_in_subprocess, task): task[0]
            for task in tasks
        }

        # Process as they complete
        for future in as_completed(futures):
            group_idx, status, error = future.result()

            if status == "success":
                completed += 1
            else:
                failed += 1

            # Progress update
            total_done = completed + failed
            pct = 100 * total_done / len(tasks)
            elapsed = (datetime.now() - start_time).total_seconds() / 3600
            rate = total_done / elapsed if elapsed > 0 else 0
            remaining = len(tasks) - total_done
            eta_hours = remaining / rate if rate > 0 else 0

            print(f"Progress: {total_done}/{len(tasks)} ({pct:.1f}%) | "
                  f"✓ {completed} ✗ {failed} | "
                  f"Rate: {rate:.1f} groups/hour | "
                  f"ETA: {eta_hours:.1f}h")

    # Summary
    print()
    print("=" * 70)
    print(f"Phase 4 Complete!")
    print(f"  Completed: {completed}")
    print(f"  Failed: {failed}")
    print(f"  Total time: {(datetime.now() - start_time).total_seconds() / 3600:.1f} hours")
    print("=" * 70)


if __name__ == "__main__":
    main()
