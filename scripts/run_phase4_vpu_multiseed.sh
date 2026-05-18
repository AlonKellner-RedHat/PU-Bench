#!/bin/bash
# Phase 4: VPU multi-seed with automatic 2-hour worker restarts

OUTPUT_DIR="results_phase4"
SHUFFLE_SEED=77777
NUM_WORKERS=12
RESTART_INTERVAL=7200  # 2 hours in seconds

# Expected: 7 datasets × 10 seeds × 3 c × 7 π × 13 method_prior × 2 methods = 38220
# (seed 42 VPU already done, resume will skip)
EXPECTED_TOTAL=38220

echo "============================================"
echo "Phase 4: VPU Multi-Seed with Auto Restarts"
echo "============================================"
echo ""
echo "Configuration:"
echo "  Methods: vpu_mean_prior, vpu_nomixup_mean_prior"
echo "  Seeds: 10 (42 + 9 new)"
echo "  Workers: ${NUM_WORKERS}"
echo "  Restart interval: 2 hours"
echo "  Resume mode: enabled"
echo ""

mkdir -p "$OUTPUT_DIR"
mkdir -p logs/phase4

CONFIGS="config/phase4/mnist_phase4.yaml config/phase4/fashionmnist_phase4.yaml config/phase4/imdb_phase4.yaml config/phase4/20news_phase4.yaml config/phase4/mushrooms_phase4.yaml config/phase4/spambase_phase4.yaml config/phase4/connect4_phase4.yaml"

ALL_METHODS="vpu_mean_prior,vpu_nomixup_mean_prior"

function start_workers() {
    local restart_num=$1
    echo "[$(date +%H:%M:%S)] Starting worker batch #${restart_num}..."

    for worker_id in $(seq 0 $((NUM_WORKERS - 1))); do
        .venv/bin/python run_train.py \
            --dataset-config $CONFIGS \
            --methods "$ALL_METHODS" \
            --output-dir "$OUTPUT_DIR" \
            --shuffle-seed "$SHUFFLE_SEED" \
            --num-workers "$NUM_WORKERS" \
            --worker-id "$worker_id" \
            --resume \
            > "logs/phase4/worker_${worker_id}_vpu_multiseed.log" 2>&1 &
    done

    echo "[$(date +%H:%M:%S)] All ${NUM_WORKERS} workers started (batch #${restart_num})"
}

function stop_workers() {
    echo "[$(date +%H:%M:%S)] Stopping workers for restart..."
    pkill -f "run_train.py.*results_phase4" 2>/dev/null
    sleep 3
    pkill -9 -f "run_train.py.*results_phase4" 2>/dev/null
    sleep 1
}

function report_progress() {
    local vpu_methods=0
    for d in "$OUTPUT_DIR"/seed_*/; do
        [ -d "$d" ] || continue
        local count=$(find "$d" -name "*.json" -exec python3 -c "
import json,sys
for f in sys.argv[1:]:
    try:
        d=json.load(open(f))
        for m in ['vpu_mean_prior','vpu_nomixup_mean_prior']:
            if m in d.get('runs',{}) and 'best' in d['runs'][m]:
                print(1)
    except: pass
" {} + 2>/dev/null | wc -l)
        vpu_methods=$((vpu_methods + count))
    done
    local percent=$(echo "scale=1; $vpu_methods * 100 / $EXPECTED_TOTAL" | bc)
    echo "[$(date +%H:%M:%S)] VPU methods completed: $vpu_methods / $EXPECTED_TOTAL ($percent%)"
}

trap "echo ''; echo 'Stopping all workers...'; stop_workers; echo 'Cleanup complete.'; exit 0" SIGINT SIGTERM

restart_count=0

echo "Starting with automatic restarts every 2 hours..."
echo ""

while true; do
    restart_count=$((restart_count + 1))

    start_workers $restart_count
    report_progress

    echo "[$(date +%H:%M:%S)] Next restart in 2 hours..."

    for i in $(seq 1 240); do
        sleep 30
        if [ $((i % 4)) -eq 0 ]; then
            report_progress
        fi
    done

    stop_workers
    sleep 5
done
