#!/usr/bin/env python3
"""Initialize SimpleNeuralPULoss to approximate VPU loss.

VPU loss: L = log(E_all[φ(x)]) - E_P[log φ(x)]
        = logsumexp(log φ(x)) - log(N) - mean(log φ_p(x))

Where φ(x) = sigmoid(x), log φ(x) = -softplus(-x)

SimpleNeuralPULoss has 40 features in 5 groups of 8:
  Group 1 (Base): 1, 1/B, p, p/B, log(p), log(p)/B, log(1-p), log(1-p)/B
  Group 2 (M_p masked): same 8 features * M_p
  Group 3 (M_u masked): same 8 features * M_u
  Group 4 (M_p normalized): same 8 features * (M_p/M_p.sum())
  Group 5 (M_u normalized): same 8 features * (M_u/M_u.sum())

Strategy:
- VPU term 1: logsumexp(log φ) ≈ sum(log φ) when φ values similar
  Use Group 1, feature 4 (log(p)) - sums to approximate log(mean(φ))

- VPU term 2: -mean(log φ_p)
  Use Group 4, feature 4 (M_p_norm * log(p)) - sums to mean of log φ over positives

We'll initialize linear layer and W1/W2 to produce VPU-like behavior.
"""

import torch
import torch.nn as nn
import numpy as np
from loss.simple_neural_pu_loss import SimpleNeuralPULoss
from loss.baseline_losses import VPUNoMixUpLoss
from tasks.gaussian_task import GaussianBlobTask


def initialize_to_vpu(loss_module: SimpleNeuralPULoss) -> SimpleNeuralPULoss:
    """Initialize SimpleNeuralPULoss to EXACTLY match VPU.

    VPU: L = log(mean(φ)) - mean_p(log(φ))
         = log(sum(p)/B) - sum_p(log(p))/N_p
         = log(sum(p)) - log(B) - mean_p(log(p))

    Mapping to SimpleNeuralPULoss architecture:
    - Feature 2: p (probability) → A2[0] → log(sum(p))
    - Feature 0: 1 (ones) → A2[1] → log(B)
    - Feature 28: M_p_norm * log(p) → A1[0] → -mean_p(log(p))

    Args:
        loss_module: SimpleNeuralPULoss instance

    Returns:
        Modified loss_module with exact VPU initialization
    """
    hidden_dim = loss_module.hidden_dim
    half = hidden_dim // 2

    with torch.no_grad():
        # Zero out all weights
        loss_module.linear.weight.zero_()
        loss_module.linear.bias.zero_()
        loss_module.W1.zero_()
        loss_module.W2.zero_()

        # Feature indices in 40-feature input:
        # Group 1 (Base, 0-7): 1, 1/B, p, p/B, log(p), log(p)/B, log(1-p), log(1-p)/B
        #   Feature 0: ones
        #   Feature 2: p
        # Group 4 (M_p normalized, 24-31): M_p_norm * [same 8 features]
        #   Feature 28: M_p_norm * log(p)  (24 + 4)

        # Term 1: -mean_p(log(p)) [LINEAR TERM, no log]
        # Feature 28 sums to mean_p(log(p)), negate it
        loss_module.linear.weight[0, 28] = 1.0  # A1[0] = mean_p(log(p))
        loss_module.W1[0] = -1.0  # Multiply by -1

        # Term 2: log(sum(p)) [LOG TERM]
        # Feature 2 sums to sum(p), then take log
        loss_module.linear.weight[half, 2] = 1.0  # A2[0] = sum(p)
        loss_module.W2[0] = 1.0  # log(sum(p))

        # Term 3: -log(B) [LOG TERM]
        # Feature 0 sums to B, then take log and negate
        loss_module.linear.weight[half + 1, 0] = 1.0  # A2[1] = B
        loss_module.W2[1] = -1.0  # -log(B)

    return loss_module


def test_vpu_initialization():
    """Test VPU initialization on sample data."""

    # Create sample task
    task = GaussianBlobTask(
        num_dimensions=2,
        mean_separation=2.0,
        std=1.0,
        prior=0.5,
        num_samples=1000,
        seed=42,
        mode='pu',
        negative_labeling_freq=0.3,
    )

    # Generate batch
    x, y_true, y_pu = task.generate_data(num_samples=100, seed=42)

    # Create simple MLP
    from models.simple_mlp import SimpleMLP
    model = SimpleMLP(2, [32, 32])
    outputs = model(x).squeeze(-1)

    # Test original VPU
    vpu_loss = VPUNoMixUpLoss()
    vpu_value = vpu_loss(outputs, y_pu)

    # Test initialized SimpleNeuralPULoss
    simple_loss = SimpleNeuralPULoss(hidden_dim=128)
    simple_loss = initialize_to_vpu(simple_loss)
    simple_value = simple_loss(outputs, y_pu)

    print(f"VPU loss value: {vpu_value.item():.6f}")
    print(f"Initialized SimpleNeuralPULoss value: {simple_value.item():.6f}")
    print(f"Ratio: {simple_value.item() / vpu_value.item():.4f}")

    # Test gradients - need fresh model outputs
    model.zero_grad()
    outputs_grad_test = model(x).squeeze(-1)
    vpu_loss(outputs_grad_test, y_pu).backward()
    # Get gradients from model parameters instead
    vpu_grad_params = torch.cat([p.grad.flatten() for p in model.parameters()]).clone()

    model.zero_grad()
    outputs_grad_test2 = model(x).squeeze(-1)
    simple_loss(outputs_grad_test2, y_pu).backward()
    simple_grad_params = torch.cat([p.grad.flatten() for p in model.parameters()]).clone()

    cos_sim = torch.nn.functional.cosine_similarity(
        vpu_grad_params.unsqueeze(0),
        simple_grad_params.unsqueeze(0)
    )

    print(f"\nGradient cosine similarity: {cos_sim.item():.4f}")
    print(f"VPU gradient norm: {vpu_grad_params.norm().item():.6f}")
    print(f"Simple gradient norm: {simple_grad_params.norm().item():.6f}")


if __name__ == '__main__':
    test_vpu_initialization()
