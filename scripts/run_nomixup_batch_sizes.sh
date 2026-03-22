#!/bin/bash

echo "========================================================================"
echo "Batch Size Sensitivity Testing - WITHOUT MixUp"
echo "========================================================================"
echo ""
echo "Testing hypothesis: Does batch size sensitivity change without MixUp?"
echo ""
echo "Previous WITH MixUp results:"
echo "  - VPU and VPU-Mean showed equal sensitivity (-0.6% AP at batch 2)"
echo "  - Dataset characteristics dominated over method choice"
echo "  - MNIST improved at small batches, IMDB degraded at small batches"
echo ""
echo "WITHOUT MixUp we can test batch size 1 (which requires batch >= 2 with MixUp)"
echo ""
echo "Batch sizes: 1, 2, 4, 8, 16, 64, 256"
echo "Datasets: MNIST, IMDB"
echo "Methods: vpu_nomixup, vpu_nomixup_mean (at each batch size)"
echo "Seeds: 42, 123"
echo "Total runs: 7 batch sizes × 2 datasets × 2 methods × 2 seeds = 56 runs"
echo "Estimated time: ~8-12 hours"
echo "========================================================================"
echo ""

python run_train.py \
  --dataset-config \
    config/datasets_batch_size/batch_validation_mnist.yaml \
    config/datasets_batch_size/batch_validation_imdb.yaml \
  --methods \
    vpu_nomixup_batch1 vpu_nomixup_batch2 vpu_nomixup_batch4 vpu_nomixup_batch8 \
    vpu_nomixup_batch16 vpu_nomixup_batch64 vpu_nomixup \
    vpu_nomixup_mean_batch1 vpu_nomixup_mean_batch2 vpu_nomixup_mean_batch4 vpu_nomixup_mean_batch8 \
    vpu_nomixup_mean_batch16 vpu_nomixup_mean_batch64 vpu_nomixup_mean \
  --resume
