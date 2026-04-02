#!/bin/bash
#
# Parallel VPU Variants Rerun
#
# Runs experiments in parallel with proper process tracking
# Expected speedup: 2-3x (running 2-3 experiments simultaneously)
#

set -e

# Process tracking
PIDFILE="/tmp/vpu_parallel_pids.txt"
LOGDIR="logs/vpu_rerun/parallel"
STATUSFILE="${LOGDIR}/status.txt"

# Cleanup function
cleanup() {
    echo ""
    echo "Caught interrupt signal - cleaning up..."

    if [ -f "$PIDFILE" ]; then
        echo "Killing all worker processes..."
        while IFS= read -r pid; do
            if ps -p "$pid" > /dev/null 2>&1; then
                echo "  Killing PID $pid"
                kill "$pid" 2>/dev/null || true
            fi
        done < "$PIDFILE"

        # Wait a moment, then force kill if needed
        sleep 2
        while IFS= read -r pid; do
            if ps -p "$pid" > /dev/null 2>&1; then
                echo "  Force killing PID $pid"
                kill -9 "$pid" 2>/dev/null || true
            fi
        done < "$PIDFILE"

        rm -f "$PIDFILE"
    fi

    echo "Cleanup complete."
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

METHODS="vpu vpu_mean vpu_mean_prior vpu_nomixup vpu_nomixup_mean vpu_nomixup_mean_prior"

# Datasets to run
DATASETS=(
    "config/vpu_rerun/mnist.yaml"
    "config/vpu_rerun/fashionmnist.yaml"
    "config/vpu_rerun/imdb.yaml"
    "config/vpu_rerun/20news.yaml"
    "config/vpu_rerun/mushrooms.yaml"
    "config/vpu_rerun/spambase.yaml"
)

echo "========================================="
echo "VPU Parallel Execution"
echo "========================================="
echo ""
echo "Process tracking:"
echo "  PID file: $PIDFILE"
echo "  Status:   $STATUSFILE"
echo "  Logs:     $LOGDIR/"
echo ""
echo "To stop: Ctrl+C or run: bash scripts/kill_vpu_parallel.sh"
echo ""
echo "========================================="
echo ""

# Create directories
mkdir -p "$LOGDIR"

# Clear old PID file
rm -f "$PIDFILE"

# Initialize status file
echo "VPU Parallel Run - Started at $(date)" > "$STATUSFILE"
echo "" >> "$STATUSFILE"

# Function to run a single dataset config
run_dataset() {
    local config=$1
    local dataset_name=$(basename $config .yaml)
    local logfile="${LOGDIR}/${dataset_name}.log"

    # Record start time
    local start_time=$(date +%s)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting $dataset_name (PID: $$)" | tee -a "$STATUSFILE"

    # Run training
    python run_train.py \
        --dataset-config "$config" \
        --methods $METHODS \
        --resume \
        > "$logfile" 2>&1

    local exit_code=$?
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed $dataset_name (${duration}s)" | tee -a "$STATUSFILE"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED $dataset_name (exit code: $exit_code)" | tee -a "$STATUSFILE"
    fi

    return $exit_code
}

# Export function and variables
export -f run_dataset
export METHODS
export LOGDIR
export STATUSFILE

# Check if GNU parallel is available
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel (3 jobs at once)"
    echo ""

    # Run with GNU parallel, tracking PIDs
    printf '%s\n' "${DATASETS[@]}" | \
        parallel --jobs 3 --line-buffer --tagstring '[{= $_=basename($_,".yaml") =}]' \
        "run_dataset {}" &

    MAIN_PID=$!
    echo "$MAIN_PID" >> "$PIDFILE"

    # Also track child processes
    sleep 1
    pgrep -P "$MAIN_PID" >> "$PIDFILE" 2>/dev/null || true

    # Wait for completion
    wait "$MAIN_PID"

else
    echo "GNU parallel not found - using bash background jobs (2 jobs)"
    echo "  Install with: brew install parallel"
    echo ""

    # Track all background PIDs
    PIDS=()

    # Run datasets in batches of 2
    for ((i=0; i<${#DATASETS[@]}; i+=2)); do
        # Start first job
        run_dataset "${DATASETS[i]}" &
        pid1=$!
        echo "$pid1" >> "$PIDFILE"
        PIDS+=($pid1)

        echo "Started job ${DATASETS[i]} (PID: $pid1)"

        # Start second job if exists
        if [ $((i+1)) -lt ${#DATASETS[@]} ]; then
            run_dataset "${DATASETS[i+1]}" &
            pid2=$!
            echo "$pid2" >> "$PIDFILE"
            PIDS+=($pid2)

            echo "Started job ${DATASETS[i+1]} (PID: $pid2)"
            echo "  Waiting for batch to complete..."

            # Wait for both to complete
            wait $pid1 2>/dev/null || echo "Job $pid1 failed or was killed"
            wait $pid2 2>/dev/null || echo "Job $pid2 failed or was killed"
        else
            # Only one job left
            echo "  Waiting for final job..."
            wait $pid1 2>/dev/null || echo "Job $pid1 failed or was killed"
        fi

        echo ""
    done
fi

# Cleanup PID file
rm -f "$PIDFILE"

echo ""
echo "========================================="
echo "Parallel rerun complete at $(date)"
echo "========================================="
echo ""
echo "Status summary:"
cat "$STATUSFILE"
echo ""
echo "Individual logs: $LOGDIR/*.log"
echo ""
