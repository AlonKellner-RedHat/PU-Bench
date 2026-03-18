#!/bin/bash
# Run benchmark for VPU-Mean-Prior and VPU-NoMixUp-Mean-Prior variants
# Tests on 6 datasets (excluding CIFAR10, AlzheimerMRI, Connect4)
# with 5 seeds (42, 123, 456, 789, 2024) across all c and prior values

set -e

echo "======================================================================"
echo "VPU Prior Variants Benchmark"
echo "======================================================================"
echo ""
echo "Methods: vpu_mean_prior, vpu_nomixup_mean_prior"
echo "Datasets: MNIST, FashionMNIST, IMDB, 20News, Spambase, Mushrooms"
echo "Seeds: 42, 123, 456, 789, 2024"
echo "Configs: vary_c + vary_prior for each dataset"
echo ""
echo "Total expected runs: 6 datasets × 12 configs × 2 methods × 5 seeds = 720 runs"
echo "======================================================================"
echo ""

# Run with multiseed configs (seeds 123, 456, 789, 2024)
python run_train.py \
  --dataset-config \
    config/datasets_multiseed/vary_c_mnist_multiseed.yaml \
    config/datasets_multiseed/vary_prior_mnist_multiseed.yaml \
    config/datasets_multiseed/vary_c_fashionmnist_multiseed.yaml \
    config/datasets_multiseed/vary_prior_fashionmnist_multiseed.yaml \
    config/datasets_multiseed/vary_c_imdb_multiseed.yaml \
    config/datasets_multiseed/vary_prior_imdb_multiseed.yaml \
    config/datasets_multiseed/vary_c_20news_multiseed.yaml \
    config/datasets_multiseed/vary_prior_20news_multiseed.yaml \
    config/datasets_multiseed/vary_c_spambase_multiseed.yaml \
    config/datasets_multiseed/vary_prior_spambase_multiseed.yaml \
    config/datasets_multiseed/vary_c_mushrooms_multiseed.yaml \
    config/datasets_multiseed/vary_prior_mushrooms_multiseed.yaml \
  --methods vpu_mean_prior vpu_nomixup_mean_prior \
  --resume

echo ""
echo "======================================================================"
echo "Multi-seed benchmark complete (seeds 123, 456, 789, 2024)"
echo "======================================================================"
echo ""
echo "Now running with seed 42..."
echo ""

# Also run with seed 42 configs (using exploration configs which have seed 42)
python run_train.py \
  --dataset-config \
    config/datasets_exploration/vary_c_mnist.yaml \
    config/datasets_exploration/vary_prior_mnist.yaml \
    config/datasets_exploration/vary_c_fashionmnist.yaml \
    config/datasets_exploration/vary_prior_fashionmnist.yaml \
    config/datasets_exploration/vary_c_imdb.yaml \
    config/datasets_exploration/vary_prior_imdb.yaml \
    config/datasets_exploration/vary_c_20news.yaml \
    config/datasets_exploration/vary_prior_20news.yaml \
    config/datasets_exploration/vary_c_spambase.yaml \
    config/datasets_exploration/vary_prior_spambase.yaml \
    config/datasets_exploration/vary_c_mushrooms.yaml \
    config/datasets_exploration/vary_prior_mushrooms.yaml \
  --methods vpu_mean_prior vpu_nomixup_mean_prior \
  --resume

echo ""
echo "======================================================================"
echo "All benchmarks complete!"
echo "======================================================================"
echo ""
echo "Results stored in:"
echo "  - results/seed_42/"
echo "  - results/seed_123/"
echo "  - results/seed_456/"
echo "  - results/seed_789/"
echo "  - results/seed_2024/"
echo ""
