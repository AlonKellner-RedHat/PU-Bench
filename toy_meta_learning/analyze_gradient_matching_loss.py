#!/usr/bin/env python3
"""Analyze the learned loss from gradient matching meta-learning.

Loads the final learned loss and analyzes:
- Weight patterns and structure
- Feature importance
- Comparison to known good loss functions (BCE, PUDRa)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from loss.neural_pu_loss import NeuralPULoss
from tasks.gaussian_task import GaussianBlobTask


def analyze_loss_parameters(learned_loss: NeuralPULoss):
    """Analyze learned loss parameters."""

    print("=" * 70)
    print("LEARNED LOSS PARAMETER ANALYSIS")
    print("=" * 70)

    # Get weights and bias
    weights = learned_loss.linear.weight.detach().cpu().numpy()  # [hidden_dim, 13]
    bias = learned_loss.linear.bias.detach().cpu().numpy()  # [hidden_dim]

    print(f"\nLinear layer shape: {weights.shape}")
    print(f"Bias shape: {bias.shape}")

    # Feature names (13 input features)
    feature_names = [
        'M_p', 'M_u', 'M_p/B', 'M_u/B', 'V', 'V_p', 'V_u',
        'log(V)', 'log(V_p)', 'log(V_u)',
        'log(1-V)', 'log(1-V_p)', 'log(1-V_u)'
    ]

    # Compute feature importance (sum of absolute weights per feature)
    feature_importance = np.abs(weights).sum(axis=0)  # [13]

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE (sum of absolute weights)")
    print("=" * 70)

    for idx in np.argsort(feature_importance)[::-1]:
        print(f"  {feature_names[idx]:15s}: {feature_importance[idx]:8.4f}")

    # Analyze weight statistics per feature
    print("\n" + "=" * 70)
    print("WEIGHT STATISTICS PER FEATURE")
    print("=" * 70)
    print(f"{'Feature':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 70)

    for i, name in enumerate(feature_names):
        feature_weights = weights[:, i]
        print(f"{name:<15} {feature_weights.mean():10.4f} {feature_weights.std():10.4f} "
              f"{feature_weights.min():10.4f} {feature_weights.max():10.4f}")

    # Overall statistics
    print("\n" + "=" * 70)
    print("OVERALL PARAMETER STATISTICS")
    print("=" * 70)

    all_params = np.concatenate([weights.flatten(), bias])
    print(f"  Total parameters: {len(all_params)}")
    print(f"  Near-zero (<0.01): {np.sum(np.abs(all_params) < 0.01)} ({np.sum(np.abs(all_params) < 0.01)/len(all_params)*100:.1f}%)")
    print(f"  Mean: {all_params.mean():.6f}")
    print(f"  Std: {all_params.std():.6f}")
    print(f"  Min: {all_params.min():.6f}")
    print(f"  Max: {all_params.max():.6f}")

    # Analyze bias
    print("\n" + "=" * 70)
    print("BIAS VECTOR ANALYSIS")
    print("=" * 70)
    print(f"  Mean: {bias.mean():.6f}")
    print(f"  Std: {bias.std():.6f}")
    print(f"  Min: {bias.min():.6f}")
    print(f"  Max: {bias.max():.6f}")
    print(f"  Near-zero (<0.01): {np.sum(np.abs(bias) < 0.01)} / {len(bias)}")

    return weights, bias, feature_names, feature_importance


def test_on_sample_task(learned_loss: NeuralPULoss):
    """Test learned loss on a sample task."""

    print("\n" + "=" * 70)
    print("TESTING ON SAMPLE GAUSSIAN TASK")
    print("=" * 70)

    # Create a sample task
    task = GaussianBlobTask(
        num_dimensions=2,
        mean_separation=2.5,
        std=1.0,
        prior=0.5,
        labeling_freq=0.3,
        num_samples=1000,
        seed=42,
        mode='pu',
        negative_labeling_freq=0.3,
    )

    # Get a batch
    dataloaders = task.get_dataloaders(batch_size=64, num_train=1000)
    batch = next(iter(dataloaders['train']))
    x, y_true, y_pu = batch

    # Create dummy predictions (logits)
    # Simulate predictions at different confidence levels
    device = 'cpu'

    test_logits = [
        torch.full((64,), -2.0),  # Low confidence (p~0.12)
        torch.full((64,), 0.0),   # Medium confidence (p=0.5)
        torch.full((64,), 2.0),   # High confidence (p~0.88)
    ]

    print("\nLoss values at different prediction confidence levels:")
    print(f"{'Logit':>10} {'Probability':>15} {'PU Loss':>15}")
    print("-" * 45)

    for logit_value in test_logits:
        prob = torch.sigmoid(logit_value[0]).item()
        loss = learned_loss(logit_value, y_pu, mode='pu')
        print(f"{logit_value[0].item():10.2f} {prob:15.4f} {loss.item():15.6f}")

    # Test gradient behavior
    print("\n" + "=" * 70)
    print("GRADIENT BEHAVIOR TEST")
    print("=" * 70)

    logits = torch.zeros(64, requires_grad=True)
    loss = learned_loss(logits, y_pu, mode='pu')
    loss.backward()

    print(f"Loss at logit=0: {loss.item():.6f}")
    print(f"Gradient mean: {logits.grad.mean().item():.6f}")
    print(f"Gradient std: {logits.grad.std().item():.6f}")
    print(f"Gradient min: {logits.grad.min().item():.6f}")
    print(f"Gradient max: {logits.grad.max().item():.6f}")


def visualize_weights(weights, feature_names):
    """Visualize weight matrix."""

    print("\n" + "=" * 70)
    print("GENERATING WEIGHT VISUALIZATION")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Heatmap of full weight matrix
    ax = axes[0, 0]
    im = ax.imshow(weights.T, aspect='auto', cmap='RdBu_r',
                   vmin=-np.abs(weights).max(), vmax=np.abs(weights).max())
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)
    ax.set_xlabel('Hidden Unit')
    ax.set_ylabel('Input Feature')
    ax.set_title('Weight Matrix Heatmap')
    plt.colorbar(im, ax=ax)

    # 2. Feature importance bar chart
    ax = axes[0, 1]
    feature_importance = np.abs(weights).sum(axis=0)
    sorted_idx = np.argsort(feature_importance)
    ax.barh(range(len(feature_names)), feature_importance[sorted_idx])
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel('Total Absolute Weight')
    ax.set_title('Feature Importance')
    ax.grid(axis='x', alpha=0.3)

    # 3. Weight distribution histogram
    ax = axes[1, 0]
    ax.hist(weights.flatten(), bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Count')
    ax.set_title('Weight Distribution')
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. Per-feature weight statistics
    ax = axes[1, 1]
    feature_means = weights.mean(axis=0)
    feature_stds = weights.std(axis=0)
    x_pos = range(len(feature_names))
    ax.errorbar(x_pos, feature_means, yerr=feature_stds, fmt='o', capsize=5)
    ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_ylabel('Weight Value')
    ax.set_title('Mean Weight per Feature (± std)')
    ax.grid(alpha=0.3)

    plt.tight_layout()

    output_path = Path('gradient_matching_output/learned_loss_weights.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Weight visualization saved to {output_path}")

    plt.close()


def main():
    # Load final learned loss
    loss_path = Path('gradient_matching_output/final_learned_loss.pt')

    if not loss_path.exists():
        print(f"Error: {loss_path} not found!")
        print("Make sure gradient matching training completed successfully.")
        return

    print("=" * 70)
    print("GRADIENT MATCHING LEARNED LOSS ANALYSIS")
    print("=" * 70)
    print(f"\nLoading learned loss from: {loss_path}")

    # Initialize and load
    learned_loss = NeuralPULoss(
        hidden_dim=64,
        eps=1e-7,
        l05_lambda=0.001,
        init_mode='xavier_uniform',
        init_scale=1.0,
    )

    learned_loss.load_state_dict(torch.load(loss_path))
    learned_loss.eval()

    print("✓ Learned loss loaded successfully")
    print(f"\n{learned_loss}")

    # Analyze parameters
    weights, bias, feature_names, feature_importance = analyze_loss_parameters(learned_loss)

    # Test on sample task
    test_on_sample_task(learned_loss)

    # Visualize weights
    visualize_weights(weights, feature_names)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    # Key insights
    print("\nKEY INSIGHTS:")
    print("-" * 70)

    # Top features
    top_k = 5
    top_indices = np.argsort(feature_importance)[::-1][:top_k]
    print(f"\nTop {top_k} most important features:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank}. {feature_names[idx]:<15} (importance: {feature_importance[idx]:.4f})")

    # Sparsity
    all_params = np.concatenate([weights.flatten(), bias])
    sparsity_pct = np.sum(np.abs(all_params) < 0.01) / len(all_params) * 100
    print(f"\nParameter sparsity: {sparsity_pct:.1f}% near zero")

    if sparsity_pct < 20:
        print("  → Low sparsity: Loss uses most parameters actively")
    elif sparsity_pct > 50:
        print("  → High sparsity: Loss has learned a sparse solution")
    else:
        print("  → Medium sparsity: Some parameters pruned")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
