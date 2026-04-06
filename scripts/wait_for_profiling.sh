#!/bin/bash
# Wait for all profiling tests to complete, then analyze

tests="gpu_preload eval_interval combined"

echo "Waiting for profiling tests to complete..."
echo "(This will take ~20-25 minutes)"
echo ""

while true; do
    all_done=true
    for test in $tests; do
        pidfile="profiling_output/cifar10_${test}.pid"
        if [ -f "$pidfile" ]; then
            pid=$(cat "$pidfile")
            if ps -p $pid > /dev/null 2>&1; then
                all_done=false
                break
            fi
        fi
    done
    
    if [ "$all_done" = true ]; then
        echo "All tests completed!"
        break
    fi
    
    echo -n "."
    sleep 30
done

echo ""
echo ""
echo "Running comparison analysis..."
bash scripts/compare_all_optimizations.sh

echo ""
echo "Analyzing cProfile data..."
for test in $tests; do
    if [ -f "profiling_output/cifar10_${test}.prof" ]; then
        echo ""
        echo "=== $test ==="
        uv run python scripts/analyze_profiling.py "profiling_output/cifar10_${test}.prof" 2>&1 | grep -A5 "tensor.to\|OPTIMIZATION PRIORITY"
    fi
done
