#!/bin/bash
# Generate multi-seed configs for all datasets

SEEDS="[123, 456, 789, 2024]"
C_VALUES="[0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]"
PRIOR_VALUES="[0.1, 0.3, 0.5, 0.7, 0.9]"

# Function to create vary_c config
create_vary_c() {
    local dataset=$1
    local embeddings_path=$2

    cat > "config/datasets_multiseed/vary_c_${dataset,,}_multiseed.yaml" << EOF
dataset_class: $dataset
data_dir: ./datasets
random_seeds: $SEEDS

c_values: $C_VALUES
scenarios: [case-control]
selection_strategies: ["random"]

val_ratio: 0.01
target_prevalence: null
with_replacement: true
also_print_dataset_stats: false
EOF

    if [ ! -z "$embeddings_path" ]; then
        cat >> "config/datasets_multiseed/vary_c_${dataset,,}_multiseed.yaml" << EOF

# SBERT encoding settings - use precomputed embeddings!
sbert_model_name: all-MiniLM-L6-v2
sbert_embeddings_path: $embeddings_path
EOF
    fi

    cat >> "config/datasets_multiseed/vary_c_${dataset,,}_multiseed.yaml" << EOF

label_scheme:
  true_positive_label: 1
  true_negative_label: 0
  pu_labeled_label: 1
  pu_unlabeled_label: -1
EOF
}

# Function to create vary_prior config
create_vary_prior() {
    local dataset=$1
    local embeddings_path=$2

    cat > "config/datasets_multiseed/vary_prior_${dataset,,}_multiseed.yaml" << EOF
dataset_class: $dataset
data_dir: ./datasets
random_seeds: $SEEDS

c_values: [0.1]
scenarios: [case-control]
selection_strategies: ["random"]

val_ratio: 0.01
target_prevalence: $PRIOR_VALUES
with_replacement: true
also_print_dataset_stats: false
EOF

    if [ ! -z "$embeddings_path" ]; then
        cat >> "config/datasets_multiseed/vary_prior_${dataset,,}_multiseed.yaml" << EOF

# SBERT encoding settings - use precomputed embeddings!
sbert_model_name: all-MiniLM-L6-v2
sbert_embeddings_path: $embeddings_path
EOF
    fi

    cat >> "config/datasets_multiseed/vary_prior_${dataset,,}_multiseed.yaml" << EOF

label_scheme:
  true_positive_label: 1
  true_negative_label: 0
  pu_labeled_label: 1
  pu_unlabeled_label: -1
EOF
}

# Create configs for all datasets
echo "Creating multi-seed configs..."

# Image datasets
create_vary_c "MNIST" ""
create_vary_prior "MNIST" ""

create_vary_c "FashionMNIST" ""
create_vary_prior "FashionMNIST" ""

# Text datasets with cached embeddings
create_vary_c "IMDB" "./scripts/embeddings/imdb_sbert_embeddings.npz"
create_vary_prior "IMDB" "./scripts/embeddings/imdb_sbert_embeddings.npz"

create_vary_c "20News" "./scripts/embeddings/20news_sbert_embeddings.npz"
create_vary_prior "20News" "./scripts/embeddings/20news_sbert_embeddings.npz"

# Tabular datasets
create_vary_c "Connect4" ""
create_vary_prior "Connect4" ""

create_vary_c "Spambase" ""
create_vary_prior "Spambase" ""

create_vary_c "Mushrooms" ""
create_vary_prior "Mushrooms" ""

echo "Done! Created configs in config/datasets_multiseed/"
ls -1 config/datasets_multiseed/
