#!/bin/bash
# Phase 4 with automatic 30-minute worker restarts to prevent memory leaks

OUTPUT_DIR="results_phase4"
SHUFFLE_SEED=77777
NUM_WORKERS=12
RESTART_INTERVAL=1800  # 30 minutes in seconds

echo "============================================"
echo "Phase 4: Prior Robustness Grid Analysis"
echo "with 30-minute automatic worker restarts"
echo "============================================"
echo ""
echo "Configuration:"
echo "  Datasets: 7 (MNIST, FashionMNIST, IMDB, 20News, Mushrooms, Spambase, Connect4)"
echo "  Methods: 10 (8 baseline prior-based + 2 VPU variants)"
echo "  Workers: ${NUM_WORKERS}"
echo "  Restart interval: 30 minutes"
echo "  Total: ~19,110 experiments"
echo ""

mkdir -p "$OUTPUT_DIR"
mkdir -p logs/phase4

# Config files for 7 Phase 4 datasets (space-separated)
CONFIGS="config/phase4/mnist_phase4.yaml config/phase4/fashionmnist_phase4.yaml config/phase4/imdb_phase4.yaml config/phase4/20news_phase4.yaml config/phase4/mushrooms_phase4.yaml config/phase4/spambase_phase4.yaml config/phase4/connect4_phase4.yaml"

ALL_METHODS="nnpu,nnpusb,lbe,distpu,selfpu,p3mixe,p3mixc,robustpu,vpu_mean_prior,vpu_nomixup_mean_prior"

# Function to start all workers
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
            > "logs/phase4/worker_${worker_id}.log" 2>&1 &

        worker_pids[$worker_id]=$!
    done

    echo "[$(date +%H:%M:%S)] All ${NUM_WORKERS} workers started (batch #${restart_num})"
}

# Function to stop all workers
function stop_workers() {
    echo "[$(date +%H:%M:%S)] Stopping workers for restart..."
    pkill -f "run_train.py.*phase4" 2>/dev/null
    sleep 3

    # Force kill if still running
    pkill -9 -f "run_train.py.*phase4" 2>/dev/null
    sleep 1

    local remaining=$(ps aux | grep "run_train.py" | grep -v grep | wc -l | tr -d ' ')
    echo "[$(date +%H:%M:%S)] Workers stopped. Remaining: $remaining"
}

# Function to check progress
function report_progress() {
    local completed=$(find "$OUTPUT_DIR" -name "*.json" | wc -l | tr -d ' ')
    local percent=$(echo "scale=2; $completed * 100 / 19110" | bc)
    local mem=$(ps aux | grep "run_train.py" | grep -v grep | awk '{sum+=$6} END {print sum/1024}' 2>/dev/null || echo "0")
    echo "[$(date +%H:%M:%S)] Progress: $completed / 19,110 ($percent%) | RAM: ${mem} MB"
}

# Main loop
restart_count=0
start_time=$(date +%s)

echo ""
echo "Starting Phase 4 with automatic restarts every 30 minutes..."
echo "Press Ctrl+C to stop gracefully"
echo ""

# Trap to handle Ctrl+C
trap "echo ''; echo 'Stopping all workers...'; stop_workers; echo 'Cleanup complete.'; exit 0" SIGINT SIGTERM

while true; do
    restart_count=$((restart_count + 1))

    # Start workers
    start_workers $restart_count
    report_progress

    # Wait for restart interval (30 minutes)
    echo "[$(date +%H:%M:%S)] Next restart in 30 minutes..."

    # Sleep in small increments to allow responsive Ctrl+C
    for i in $(seq 1 60); do
        sleep 30
        if [ $((i % 2)) -eq 0 ]; then
            report_progress
        fi
    done

    # Stop workers before restarting
    stop_workers

    # Brief pause before restart
    sleep 5
done
