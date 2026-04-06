#!/bin/bash
# Summarize optimization results

echo "=========================================="
echo "Optimization Results Summary"
echo "=========================================="
echo ""

extract_time() {
    local logfile=$1
    grep "Training (VPU):" "$logfile" 2>/dev/null | \
        grep -oE '[0-9]+\.[0-9]+s/it' | \
        tail -5 | \
        awk -F's/' '{sum+=$1; count++} END {if(count>0) printf "%.1f", sum/count; else print "N/A"}'
}

extract_total_time() {
    local logfile=$1
    grep "Training (VPU):" "$logfile" 2>/dev/null | \
        tail -1 | \
        grep -oE '\[[0-9]+:[0-9]+<' | \
        sed 's/\[//;s/<//'
}

echo "=== Per-Epoch Timing (average of last 5 epochs) ==="
echo ""

baseline_cifar=$(extract_time profiling_output/cifar10_cprofile.log)
mps_opt_cifar=$(extract_time profiling_output/cifar10_mps_optimized.log)

baseline_alzh=$(extract_time profiling_output/alzheimermri_cprofile.log)
mps_opt_alzh=$(extract_time profiling_output/alzheimermri_mps_optimized.log)

echo "CIFAR10:"
echo "  Baseline:      ${baseline_cifar}s/epoch"
echo "  MPS-Optimized: ${mps_opt_cifar}s/epoch"

if [[ "$baseline_cifar" != "N/A" && "$mps_opt_cifar" != "N/A" ]]; then
    speedup=$(echo "scale=2; $baseline_cifar / $mps_opt_cifar" | bc)
    if (( $(echo "$speedup > 1" | bc -l) )); then
        pct=$(echo "scale=1; ($speedup - 1) * 100" | bc)
        echo "  → ${pct}% FASTER"
    else
        pct=$(echo "scale=1; (1 - $speedup) * 100" | bc)
        echo "  → ${pct}% slower"
    fi
fi

echo ""
echo "AlzheimerMRI:"
echo "  Baseline:      ${baseline_alzh}s/epoch"
echo "  MPS-Optimized: ${mps_opt_alzh}s/epoch"

if [[ "$baseline_alzh" != "N/A" && "$mps_opt_alzh" != "N/A" ]]; then
    speedup=$(echo "scale=2; $baseline_alzh / $mps_opt_alzh" | bc)
    if (( $(echo "$speedup > 1" | bc -l) )); then
        pct=$(echo "scale=1; ($speedup - 1) * 100" | bc)
        echo "  → ${pct}% FASTER"
    else
        pct=$(echo "scale=1; (1 - $speedup) * 100" | bc)
        echo "  → ${pct}% slower"
    fi
fi

echo ""
echo "=== Total Runtime ==="
baseline_total=$(extract_total_time profiling_output/cifar10_cprofile.log)
mps_opt_total=$(extract_total_time profiling_output/cifar10_mps_optimized.log)

echo "CIFAR10 Baseline:      $baseline_total"
echo "CIFAR10 MPS-Optimized: $mps_opt_total"

echo ""
echo "=== Optimizations Applied ==="
echo "  ✓ Platform detection (MPS vs CUDA)"
echo "  ✓ num_workers=2 for MPS (avoid contention)"
echo "  ✓ pin_memory=False for MPS (not supported)"
echo "  ✓ persistent_workers=True"
