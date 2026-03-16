#!/bin/bash
# Run improved gradient matching meta-learning

cd "$(dirname "$0")/.."

echo "Starting improved gradient matching meta-learning..."
echo "Output will be saved to: gradient_matching_output_improved/"
echo ""

python -u train_gradient_matching_improved.py

echo ""
echo "Training complete!"
echo "Check gradient_matching_output_improved/ for results"
