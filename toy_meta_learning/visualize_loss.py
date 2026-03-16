#!/usr/bin/env python3
"""Visualize learned loss parameters."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load checkpoints
best_ckpt = torch.load('gradient_matching_output/best_checkpoint.pt', map_location='cpu', weights_only=False)
final_ckpt = torch.load('gradient_matching_output/final_learned_loss.pt', map_location='cpu', weights_only=False)

best_state = best_ckpt['loss_state_dict']
final_state = final_ckpt

best_w = best_state['linear.weight'].numpy()
final_w = final_state['linear.weight'].numpy()

feature_names = [
    "M_p", "M_u", "M_p/B", "M_u/B", "V", "V_p", "V_u",
    "log(V)", "log(V_p)", "log(V_u)", "log(1-V)", "log(1-V_p)", "log(1-V_u)"
]

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Weight heatmap (Best)
ax = axes[0, 0]
im = ax.imshow(best_w.T, aspect='auto', cmap='RdBu_r', vmin=-0.3, vmax=0.3)
ax.set_xlabel('Hidden Unit', fontsize=12)
ax.set_ylabel('Input Feature', fontsize=12)
ax.set_title('Best Checkpoint (iter 240) - Weight Matrix [13 × 64]', fontsize=14, fontweight='bold')
ax.set_yticks(range(13))
ax.set_yticklabels(feature_names, fontsize=10)
plt.colorbar(im, ax=ax, label='Weight Value')

# 2. Weight heatmap (Final)
ax = axes[0, 1]
im = ax.imshow(final_w.T, aspect='auto', cmap='RdBu_r', vmin=-0.3, vmax=0.3)
ax.set_xlabel('Hidden Unit', fontsize=12)
ax.set_ylabel('Input Feature', fontsize=12)
ax.set_title('Final Checkpoint (iter 500) - Weight Matrix [13 × 64]', fontsize=14, fontweight='bold')
ax.set_yticks(range(13))
ax.set_yticklabels(feature_names, fontsize=10)
plt.colorbar(im, ax=ax, label='Weight Value')

# 3. Feature importance comparison
ax = axes[1, 0]
best_importance = np.abs(best_w).mean(axis=0)
final_importance = np.abs(final_w).mean(axis=0)
x = np.arange(13)
width = 0.35
ax.bar(x - width/2, best_importance, width, label='Best (iter 240)', alpha=0.8)
ax.bar(x + width/2, final_importance, width, label='Final (iter 500)', alpha=0.8)
ax.set_xlabel('Input Feature', fontsize=12)
ax.set_ylabel('Importance (Mean |Weight|)', fontsize=12)
ax.set_title('Feature Importance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=9)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 4. Weight change heatmap
ax = axes[1, 1]
weight_change = final_w - best_w
im = ax.imshow(weight_change.T, aspect='auto', cmap='RdBu_r', vmin=-0.1, vmax=0.1)
ax.set_xlabel('Hidden Unit', fontsize=12)
ax.set_ylabel('Input Feature', fontsize=12)
ax.set_title('Weight Change (Final - Best)', fontsize=14, fontweight='bold')
ax.set_yticks(range(13))
ax.set_yticklabels(feature_names, fontsize=10)
plt.colorbar(im, ax=ax, label='Weight Change')

plt.tight_layout()
plt.savefig('gradient_matching_output/learned_loss_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization to gradient_matching_output/learned_loss_analysis.png")

# Create second figure: X1 vs X2 analysis
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# X1 weights (units 0-31)
ax = axes[0]
x1_weights = best_w[:32, :].T
im = ax.imshow(x1_weights, aspect='auto', cmap='RdBu_r', vmin=-0.3, vmax=0.3)
ax.set_xlabel('X1 Hidden Unit (0-31)', fontsize=12)
ax.set_ylabel('Input Feature', fontsize=12)
ax.set_title('X1 Component - First Multiplicand', fontsize=14, fontweight='bold')
ax.set_yticks(range(13))
ax.set_yticklabels(feature_names, fontsize=10)
plt.colorbar(im, ax=ax, label='Weight Value')

# X2 weights (units 32-63)
ax = axes[1]
x2_weights = best_w[32:, :].T
im = ax.imshow(x2_weights, aspect='auto', cmap='RdBu_r', vmin=-0.3, vmax=0.3)
ax.set_xlabel('X2 Hidden Unit (32-63)', fontsize=12)
ax.set_ylabel('Input Feature', fontsize=12)
ax.set_title('X2 Component - Second Multiplicand', fontsize=14, fontweight='bold')
ax.set_yticks(range(13))
ax.set_yticklabels(feature_names, fontsize=10)
plt.colorbar(im, ax=ax, label='Weight Value')

plt.tight_layout()
plt.savefig('gradient_matching_output/learned_loss_x1_x2.png', dpi=150, bbox_inches='tight')
print("✓ Saved X1/X2 analysis to gradient_matching_output/learned_loss_x1_x2.png")

print("\nVisualization complete!")
