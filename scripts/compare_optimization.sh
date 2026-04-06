#!/bin/bash
# Compare baseline vs optimized profiling runs

echo "=========================================="
echo "Optimization Comparison"
echo "=========================================="
echo ""

# Check if processes are running
echo "=== Status ==="
cifar_opt_pid=$(cat profiling_output/cifar10_optimized.pid 2>/dev/null)
alzh_opt_pid=$(cat profiling_output/alzheimermri_optimized.pid 2>/dev/null)

if ps -p $cifar_opt_pid > /dev/null 2>&1; then
    etime=$(ps -o etime= -p $cifar_opt_pid | tr -d ' ')
    echo "✓ CIFAR10 optimized: Running ($etime)"
else
    echo "✗ CIFAR10 optimized: Completed/Stopped"
fi

if ps -p $alzh_opt_pid > /dev/null 2>&1; then
    etime=$(ps -o etime= -p $alzh_opt_pid | tr -d ' ')
    echo "✓ AlzheimerMRI optimized: Running ($etime)"
else
    echo "✗ AlzheimerMRI optimized: Completed/Stopped"
fi

echo ""
echo "=== Completion Times (when both finish) ==="

# Extract completion times from logs
if grep -q "✔ DONE" profiling_output/cifar10_cprofile.log 2>/dev/null; then
    baseline_time=$(grep "Training (VPU):" profiling_output/cifar10_cprofile.log | tail -1 | grep -oE '[0-9]+:[0-9]+' | head -1)
    echo "CIFAR10 Baseline: $baseline_time"
fi

if grep -q "✔ DONE" profiling_output/cifar10_optimized.log 2>/dev/null; then
    optimized_time=$(grep "Training (VPU):" profiling_output/cifar10_optimized.log | tail -1 | grep -oE '[0-9]+:[0-9]+' | head -1)
    echo "CIFAR10 Optimized: $optimized_time"
fi

echo ""
echo "Monitor logs:"
echo "  Baseline:  tail -f profiling_output/cifar10_cprofile.log"
echo "  Optimized: tail -f profiling_output/cifar10_optimized.log"
