#!/bin/bash
# Compare all optimization test results

echo "=========================================="
echo "Optimization Profiling Results"
echo "=========================================="
echo ""

extract_avg_time() {
    local logfile=$1
    # Extract times from epochs 3-10 (skip first 2, stop before early stopping)
    grep "Training (VPU):" "$logfile" 2>/dev/null | \
        grep -oE '[0-9]+\.[0-9]+s/it' | \
        head -10 | tail -7 | \
        awk -F's/' '{sum+=$1; count++} END {if(count>0) printf "%.1f", sum/count; else print "N/A"}'
}

echo "Configuration                  | Avg Time/Epoch | vs Baseline | Notes"
echo "-------------------------------|----------------|-------------|-------"

baseline=$(extract_avg_time profiling_output/cifar10_fixed.log)
gpu_preload=$(extract_avg_time profiling_output/cifar10_gpu_preload.log)
eval_interval=$(extract_avg_time profiling_output/cifar10_eval_interval.log)
combined=$(extract_avg_time profiling_output/cifar10_combined.log)

printf "Baseline (MPS-aware)           | %6ss         | -           | num_workers=2\n" "$baseline"

if [[ "$gpu_preload" != "N/A" ]]; then
    if (( $(echo "$baseline > 0" | bc -l) )); then
        speedup=$(echo "scale=2; $baseline / $gpu_preload" | bc)
        if (( $(echo "$speedup > 1" | bc -l) )); then
            pct=$(echo "scale=0; ($speedup - 1) * 100" | bc)
            printf "GPU Preload                    | %6ss         | +%d%% faster | Data on GPU\n" "$gpu_preload" "$pct"
        else
            pct=$(echo "scale=0; (1 - $speedup) * 100" | bc)
            printf "GPU Preload                    | %6ss         | -%d%% slower | Data on GPU\n" "$gpu_preload" "$pct"
        fi
    fi
fi

if [[ "$eval_interval" != "N/A" ]]; then
    if (( $(echo "$baseline > 0" | bc -l) )); then
        speedup=$(echo "scale=2; $baseline / $eval_interval" | bc)
        if (( $(echo "$speedup > 1" | bc -l) )); then
            pct=$(echo "scale=0; ($speedup - 1) * 100" | bc)
            printf "Eval Interval=5                | %6ss         | +%d%% faster | Eval every 5 epochs\n" "$eval_interval" "$pct"
        else
            pct=$(echo "scale=0; (1 - $speedup) * 100" | bc)
            printf "Eval Interval=5                | %6ss         | -%d%% slower | Eval every 5 epochs\n" "$eval_interval" "$pct"
        fi
    fi
fi

if [[ "$combined" != "N/A" ]]; then
    if (( $(echo "$baseline > 0" | bc -l) )); then
        speedup=$(echo "scale=2; $baseline / $combined" | bc)
        if (( $(echo "$speedup > 1" | bc -l) )); then
            pct=$(echo "scale=0; ($speedup - 1) * 100" | bc)
            printf "Combined (both)                | %6ss         | +%d%% faster | GPU + eval=5\n" "$combined" "$pct"
        else
            pct=$(echo "scale=0; (1 - $speedup) * 100" | bc)
            printf "Combined (both)                | %6ss         | -%d%% slower | GPU + eval=5\n" "$combined" "$pct"
        fi
    fi
fi

echo ""
echo "Phase 2 Impact (1,500 experiments on MPS):"
echo "  Baseline:       $(echo "$baseline * 28 * 1500 / 60" | bc) hours"
if [[ "$combined" != "N/A" ]] && (( $(echo "$combined > 0" | bc -l) )); then
    echo "  With Combined:  $(echo "$combined * 28 * 1500 / 60" | bc) hours"
fi
