#!/bin/bash
# Benchmark different worker counts to find optimal parallelization

OUTPUT_DIR="results_phase1_extended"
SHUFFLE_SEED=54321
CONFIGS="config/comprehensive/mnist_comprehensive.yaml config/comprehensive/fashionmnist_comprehensive.yaml config/comprehensive/imdb_comprehensive.yaml config/comprehensive/20news_comprehensive.yaml config/comprehensive/mushrooms_comprehensive.yaml config/comprehensive/spambase_comprehensive.yaml config/comprehensive/connect4_comprehensive.yaml"
METHODS="nnpu,nnpusb,bbepu,lbe,puet,distpu,pulda,selfpu,p3mixe,p3mixc,robustpu,holisticpu,lagam,pulcpbf,vaepu,pan,cgenpu,pn_naive,oracle_bce,vpu,vpu_nomixup,vpu_nomixup_mean_prior,vpu_mean_prior"

NUM_WORKERS=$1
BENCHMARK_DURATION=$2  # seconds

echo "=========================================="
echo "Benchmarking with ${NUM_WORKERS} workers"
echo "Duration: ${BENCHMARK_DURATION} seconds"
echo "=========================================="

# Record start state
START_COUNT=$(find "$OUTPUT_DIR" -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
echo "Experiments completed at start: $START_COUNT"

# Launch workers
for worker_id in $(seq 0 $((NUM_WORKERS - 1))); do
    python run_train.py \
        --dataset-config $CONFIGS \
        --methods "$METHODS" \
        --output-dir "$OUTPUT_DIR" \
        --shuffle-seed "$SHUFFLE_SEED" \
        --num-workers "$NUM_WORKERS" \
        --worker-id "$worker_id" \
        --resume \
        > "logs/phase1_extended/benchmark_worker_${worker_id}.log" 2>&1 &
done

echo "Workers launched. PID list:"
pgrep -f "run_train.py.*phase1_extended" | head -${NUM_WORKERS}

# Wait for benchmark duration
sleep "$BENCHMARK_DURATION"

# Kill workers
pkill -f "run_train.py.*phase1_extended"
sleep 2

# Record end state
END_COUNT=$(find "$OUTPUT_DIR" -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
COMPLETED=$((END_COUNT - START_COUNT))

echo ""
echo "=========================================="
echo "Benchmark Results (${NUM_WORKERS} workers)"
echo "=========================================="
echo "Duration: ${BENCHMARK_DURATION}s"
echo "Completed at start: $START_COUNT"
echo "Completed at end: $END_COUNT"
echo "New experiments: $COMPLETED"
echo "Rate: $(echo "scale=2; $COMPLETED * 3600 / $BENCHMARK_DURATION" | bc) experiments/hour"
echo "Avg time: $(echo "scale=1; $BENCHMARK_DURATION / $COMPLETED" | bc) seconds/experiment"
echo "=========================================="
