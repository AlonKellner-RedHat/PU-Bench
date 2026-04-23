#!/usr/bin/env python3
"""
Test the fixed --resume logic to ensure it properly detects incomplete method runs.
"""

import json
import tempfile
import shutil
from pathlib import Path
import sys

# Add current directory to path to import run_train
sys.path.insert(0, str(Path(__file__).parent))
from run_train import _method_already_completed

def test_resume_logic():
    """Test various scenarios for method completion detection."""

    # Create temporary directory for test
    test_dir = Path(tempfile.mkdtemp())
    seed_dir = test_dir / "seed_42"
    seed_dir.mkdir(parents=True)

    print("=" * 80)
    print("TESTING FIXED --RESUME LOGIC")
    print("=" * 80)
    print()

    # Test Case 1: File doesn't exist
    print("Test 1: File doesn't exist")
    result = _method_already_completed("nonexistent_exp", "vpu", 42, str(test_dir))
    assert result == False, "Should return False when file doesn't exist"
    print("  ✓ Correctly returns False")
    print()

    # Test Case 2: File exists but method not in runs
    print("Test 2: File exists but method not in runs")
    test_file = seed_dir / "test_exp.json"
    with open(test_file, "w") as f:
        json.dump({"runs": {}}, f)

    result = _method_already_completed("test_exp", "vpu", 42, str(test_dir))
    assert result == False, "Should return False when method not in runs"
    print("  ✓ Correctly returns False")
    print()

    # Test Case 3: Method in runs but no 'best' field
    print("Test 3: Method in runs but no 'best' field")
    with open(test_file, "w") as f:
        json.dump({
            "runs": {
                "vpu": {
                    "epochs": 10
                }
            }
        }, f)

    result = _method_already_completed("test_exp", "vpu", 42, str(test_dir))
    assert result == False, "Should return False when 'best' field missing"
    print("  ✓ Correctly returns False (incomplete - no 'best' field)")
    print()

    # Test Case 4: Has 'best' but no 'metrics'
    print("Test 4: Has 'best' but no 'metrics'")
    with open(test_file, "w") as f:
        json.dump({
            "runs": {
                "vpu": {
                    "best": {
                        "epoch": 5
                    }
                }
            }
        }, f)

    result = _method_already_completed("test_exp", "vpu", 42, str(test_dir))
    assert result == False, "Should return False when 'metrics' missing"
    print("  ✓ Correctly returns False (incomplete - no 'metrics')")
    print()

    # Test Case 5: Has 'metrics' but no 'test_auc'
    print("Test 5: Has 'metrics' but no 'test_auc'")
    with open(test_file, "w") as f:
        json.dump({
            "runs": {
                "vpu": {
                    "best": {
                        "metrics": {
                            "test_f1": 0.85
                        }
                    }
                }
            }
        }, f)

    result = _method_already_completed("test_exp", "vpu", 42, str(test_dir))
    assert result == False, "Should return False when 'test_auc' missing"
    print("  ✓ Correctly returns False (incomplete - no 'test_auc')")
    print()

    # Test Case 6: Complete method run
    print("Test 6: Complete method run with all required fields")
    with open(test_file, "w") as f:
        json.dump({
            "runs": {
                "vpu": {
                    "best": {
                        "metrics": {
                            "test_auc": 0.92,
                            "test_f1": 0.85,
                            "test_accuracy": 0.88
                        }
                    },
                    "epochs": 25
                }
            }
        }, f)

    result = _method_already_completed("test_exp", "vpu", 42, str(test_dir))
    assert result == True, "Should return True when method is complete"
    print("  ✓ Correctly returns True (complete method run)")
    print()

    # Test Case 7: Multiple methods, one complete, one incomplete
    print("Test 7: File with mixed complete/incomplete methods")
    with open(test_file, "w") as f:
        json.dump({
            "runs": {
                "vpu": {
                    "best": {
                        "metrics": {
                            "test_auc": 0.92
                        }
                    }
                },
                "vpu_nomixup": {
                    # Incomplete - no 'best' field
                    "epochs": 10
                }
            }
        }, f)

    result_complete = _method_already_completed("test_exp", "vpu", 42, str(test_dir))
    result_incomplete = _method_already_completed("test_exp", "vpu_nomixup", 42, str(test_dir))

    assert result_complete == True, "vpu should be detected as complete"
    assert result_incomplete == False, "vpu_nomixup should be detected as incomplete"
    print("  ✓ Correctly identifies vpu as complete")
    print("  ✓ Correctly identifies vpu_nomixup as incomplete")
    print()

    # Cleanup
    shutil.rmtree(test_dir)

    print("=" * 80)
    print("✅ ALL TESTS PASSED - Resume logic correctly detects incomplete methods")
    print("=" * 80)
    print()
    print("The fixed --resume behavior now:")
    print("  1. Checks if file exists")
    print("  2. Checks if method is in 'runs'")
    print("  3. Checks if 'best' field exists")
    print("  4. Checks if 'metrics' exists in 'best'")
    print("  5. Checks if 'test_auc' exists in 'metrics'")
    print()
    print("This ensures only COMPLETE method experiments are skipped.")

if __name__ == "__main__":
    test_resume_logic()
