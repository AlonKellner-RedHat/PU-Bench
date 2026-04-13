#!/usr/bin/env python3
"""
Recompute Metrics from Saved Scores

Loads NPZ files with raw model outputs and recomputes all metrics.
Useful for adding new metrics without re-training.

Usage:
    python analysis/recompute_metrics.py \
        --results-dir results_phase1_extended \
        --output results_phase1_extended_recomputed.json
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)

# Import calibration metrics computation from train_utils
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from train.train_utils import compute_calibration_metrics


def load_scores_npz(npz_path: Path) -> dict:
    """Load raw outputs from NPZ file."""
    data = np.load(npz_path)
    return {
        "train": {
            "y_true": data["train_y_true"],
            "y_pred": data["train_y_pred"],
            "y_scores": data["train_y_scores"],
        },
        "val": {
            "y_true": data["val_y_true"],
            "y_pred": data["val_y_pred"],
            "y_scores": data["val_y_scores"],
        },
        "test": {
            "y_true": data["test_y_true"],
            "y_pred": data["test_y_pred"],
            "y_scores": data["test_y_scores"],
        },
    }


def compute_metrics_from_outputs(y_true, y_pred, y_scores, prior=None) -> dict:
    """Compute all metrics from raw outputs."""
    # Handle empty arrays (e.g., when validation set doesn't exist)
    if len(y_true) == 0:
        return {
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "auc": float("nan"),
            "ap": float("nan"),
            "max_f1": float("nan"),
            "oracle_ce": float("nan"),
            "ece": float("nan"),
            "mce": float("nan"),
            "brier": float("nan"),
            "anice": float("nan"),
            "snice": float("nan"),
        }

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    # AUC and AP (require at least 2 classes)
    if len(np.unique(y_true)) > 1:
        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_scores))
        except Exception:
            metrics["auc"] = float("nan")

        try:
            metrics["ap"] = float(average_precision_score(y_true, y_scores))
        except Exception:
            metrics["ap"] = float("nan")

        # Max F1 (threshold-optimized)
        try:
            precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
            f1_scores = np.where(
                (precision + recall) > 0,
                2 * (precision * recall) / (precision + recall),
                0
            )
            metrics["max_f1"] = float(np.max(f1_scores))
        except Exception:
            metrics["max_f1"] = float("nan")
    else:
        metrics["auc"] = float("nan")
        metrics["ap"] = float("nan")
        metrics["max_f1"] = float("nan")

    # Calibration metrics
    if len(np.unique(y_true)) > 1:
        try:
            cal_metrics = compute_calibration_metrics(y_true, y_scores, n_bins=15)
            metrics.update(cal_metrics)  # ece, mce, brier, anice, snice
        except Exception:
            metrics.update({
                "ece": float("nan"),
                "mce": float("nan"),
                "brier": float("nan"),
                "anice": float("nan"),
                "snice": float("nan"),
            })
    else:
        metrics.update({
            "ece": float("nan"),
            "mce": float("nan"),
            "brier": float("nan"),
            "anice": float("nan"),
            "snice": float("nan"),
        })

    # Oracle CE (uses true labels)
    if len(np.unique(y_true)) > 1:
        try:
            # Convert logits to probabilities
            y_probs = 1.0 / (1.0 + np.exp(-y_scores))
            # Clip to avoid log(0)
            y_probs = np.clip(y_probs, 1e-7, 1 - 1e-7)
            # Binary cross-entropy
            ce = -np.mean(
                y_true * np.log(y_probs) + (1 - y_true) * np.log(1 - y_probs)
            )
            metrics["oracle_ce"] = float(ce)
        except Exception:
            metrics["oracle_ce"] = float("nan")
    else:
        metrics["oracle_ce"] = float("nan")

    return metrics


def compute_timing_metrics(method_data):
    """Compute timing-based metrics from method results."""
    timing_metrics = {}

    # Extract timing info
    duration_sec = method_data.get("timing", {}).get("duration_seconds", None)
    global_epochs = method_data.get("global_epochs", None)
    best_epoch = method_data.get("best", {}).get("epoch", None)

    if duration_sec is not None and global_epochs is not None and best_epoch is not None:
        # Total convergence time (wall clock)
        timing_metrics["total_time"] = duration_sec

        # Average time per epoch
        timing_metrics["avg_time_per_epoch"] = duration_sec / global_epochs if global_epochs > 0 else None

        # Time to best epoch (when best validation performance was achieved)
        timing_metrics["time_to_best"] = (best_epoch / global_epochs) * duration_sec if global_epochs > 0 else None

        # Efficiency ratio (lower = converged earlier)
        timing_metrics["efficiency_ratio"] = best_epoch / global_epochs if global_epochs > 0 else None

        # Convergence speed (epochs)
        timing_metrics["epochs_to_best"] = best_epoch
        timing_metrics["total_epochs"] = global_epochs

    return timing_metrics


def recompute_all_metrics(results_dir: Path, output_json: Path):
    """Recompute metrics for all experiments in results directory."""

    recomputed_results = {}

    # Iterate over all seed directories
    for seed_dir in sorted(results_dir.glob("seed_*")):
        seed = int(seed_dir.name.split("_")[1])

        # Iterate over all experiment JSON files
        for json_file in sorted(seed_dir.glob("*.json")):
            exp_name = json_file.stem

            # Load original JSON to get hyperparameters and metadata
            try:
                with open(json_file) as f:
                    original_data = json.load(f)
            except Exception as e:
                print(f"WARNING: Could not load {json_file}: {e}")
                continue

            # Recompute metrics for each method
            exp_results = {"experiments": {}}
            for method in original_data.get("runs", {}):
                # New format includes experiment name to avoid collisions
                scores_path = seed_dir / "scores" / f"{exp_name}_{method}_scores.npz"

                if not scores_path.exists():
                    print(f"WARNING: Scores not found for {exp_name}/{method}, skipping")
                    continue

                try:
                    # Load raw outputs
                    scores = load_scores_npz(scores_path)

                    # Recompute metrics for train/val/test
                    train_metrics = compute_metrics_from_outputs(
                        scores["train"]["y_true"],
                        scores["train"]["y_pred"],
                        scores["train"]["y_scores"],
                    )
                    val_metrics = compute_metrics_from_outputs(
                        scores["val"]["y_true"],
                        scores["val"]["y_pred"],
                        scores["val"]["y_scores"],
                    )
                    test_metrics = compute_metrics_from_outputs(
                        scores["test"]["y_true"],
                        scores["test"]["y_pred"],
                        scores["test"]["y_scores"],
                    )

                    # Store with train_/val_/test_ prefixes
                    method_metrics = {}
                    for split, metrics in [("train", train_metrics), ("val", val_metrics), ("test", test_metrics)]:
                        for metric_name, value in metrics.items():
                            method_metrics[f"{split}_{metric_name}"] = value

                    # Compute timing metrics from original data
                    timing_metrics = compute_timing_metrics(original_data["runs"][method])
                    method_metrics.update(timing_metrics)

                    exp_results["experiments"][method] = {
                        "metrics": method_metrics,
                        "hyperparameters": original_data["runs"][method].get("hyperparameters", {}),
                        "timing": original_data["runs"][method].get("timing", {}),
                        "global_epochs": original_data["runs"][method].get("global_epochs", None),
                        "best_epoch": original_data["runs"][method].get("best", {}).get("epoch", None),
                    }
                except Exception as e:
                    print(f"ERROR: Failed to recompute metrics for {exp_name}/{method}: {e}")
                    continue

            recomputed_results[f"{exp_name}_seed{seed}"] = exp_results

    # Save recomputed metrics
    with open(output_json, "w") as f:
        json.dump(recomputed_results, f, indent=2)

    print(f"\nRecomputed metrics saved to: {output_json}")
    print(f"Total experiments processed: {len(recomputed_results)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Recompute metrics from saved NPZ score files"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing experiment results with scores/ subdirectories"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file for recomputed metrics"
    )
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"ERROR: Results directory not found: {args.results_dir}")
        sys.exit(1)

    recompute_all_metrics(args.results_dir, args.output)
