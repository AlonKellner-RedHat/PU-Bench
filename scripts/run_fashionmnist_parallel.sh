#!/bin/bash
#
# Parallel FashionMNIST Final Sprint - 3 seed groups
#

METHODS="vpu vpu_mean vpu_mean_prior vpu_nomixup vpu_nomixup_mean vpu_nomixup_mean_prior"

echo "========================================="
echo "FashionMNIST Parallel Final Sprint"
echo "========================================="
echo ""
echo "Running 3 seed groups in parallel:"
echo "  - seed_456 (5 experiments)"
echo "  - seed_789 (42 experiments)"
echo "  - seed_2024 (42 experiments)"
echo ""
echo "Total: 89 experiments"
echo "========================================="
echo ""

mkdir -p logs/vpu_rerun/final

# Run all 3 in parallel
python run_train.py \
    --dataset-config config/vpu_rerun/fashionmnist_seed456.yaml \
    --methods $METHODS \
    --resume \
    > logs/vpu_rerun/final/fashionmnist_seed456.log 2>&1 &
PID1=$!
echo "Started seed_456 (PID: $PID1)"

python run_train.py \
    --dataset-config config/vpu_rerun/fashionmnist_seed789.yaml \
    --methods $METHODS \
    --resume \
    > logs/vpu_rerun/final/fashionmnist_seed789.log 2>&1 &
PID2=$!
echo "Started seed_789 (PID: $PID2)"

python run_train.py \
    --dataset-config config/vpu_rerun/fashionmnist_seed2024.yaml \
    --methods $METHODS \
    --resume \
    > logs/vpu_rerun/final/fashionmnist_seed2024.log 2>&1 &
PID3=$!
echo "Started seed_2024 (PID: $PID3)"

echo ""
echo "All 3 workers started!"
echo "PIDs: $PID1 $PID2 $PID3"
echo ""
echo "Monitor with: ps aux | grep run_train.py"
echo "Logs: logs/vpu_rerun/final/*.log"
echo ""
echo "Waiting for completion..."

# Wait for all to finish
wait $PID1
wait $PID2
wait $PID3

echo ""
echo "========================================="
echo "All FashionMNIST experiments complete!"
echo "========================================="
