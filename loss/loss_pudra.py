"""PUDRa (Positive-Unlabeled Density Ratio) Loss

This module implements the PUDRa loss function, which estimates the density ratio
D(x) = P(x|y=1)/P(x) using the unbiased PU risk estimator with Point Process/
Generalized KL loss functions.

Mathematical Foundation:
    L_PUDRa = π * E_P[-log(g(x))] + E_U[g(x)]

    where:
    - π is the class prior P(y=1)
    - g(x) is a non-negative function (sigmoid or softplus of model output)
    - E_P is expectation over positive samples
    - E_U is expectation over unlabeled samples

Key Properties:
    - Strictly convex and non-negative
    - No clipping constraints required (unlike nnPU)
    - Term 1 maximizes log-likelihood at positive points
    - Term 2 minimizes total density integral over domain
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PUDRALoss(nn.Module):
    """PUDRa loss for Positive-Unlabeled learning.

    Implements the Point Process / Generalized KL formulation:
        L = π * E_P[-log(g(x))] + E_U[g(x)]

    Args:
        prior (float): Class prior π = P(y=1), must be in (0, 1)
        activation (str): Activation function for g(x):
            - "sigmoid": g(x) = sigmoid(x), bounded [0,1], converges to P(y=1|x)
            - "softplus": g(x) = softplus(x), unbounded [0,∞), converges to density ratio
        epsilon (float): Small constant for numerical stability in log computation
    """

    def __init__(self, prior: float, activation: str = "sigmoid", epsilon: float = 1e-7):
        super().__init__()

        if not 0 < prior < 1:
            raise ValueError(f"Class prior must be in (0, 1), got {prior}")

        if activation not in ["sigmoid", "softplus"]:
            raise ValueError(f"activation must be 'sigmoid' or 'softplus', got '{activation}'")

        self.prior = float(prior)
        self.activation = activation
        self.epsilon = float(epsilon)

        # Label constants (following PU-Bench convention)
        self.positive = 1
        self.unlabeled = -1

    def forward(self, outputs: torch.Tensor, pu_labels: torch.Tensor,
                weights: torch.Tensor = None) -> torch.Tensor:
        """Compute PUDRa loss.

        Args:
            outputs: Model logits, shape [N]
            pu_labels: PU labels where 1=positive, -1=unlabeled, shape [N]
            weights: Optional sample weights, shape [N] (currently unused)

        Returns:
            Scalar loss tensor
        """
        outputs = outputs.view(-1)
        pu_labels = pu_labels.view(-1)

        # Separate positive and unlabeled samples
        positive_mask = pu_labels == self.positive
        unlabeled_mask = pu_labels == self.unlabeled

        n_positive = max(1, positive_mask.sum().item())
        n_unlabeled = max(1, unlabeled_mask.sum().item())

        # Apply activation to get g(x)
        if self.activation == "sigmoid":
            g = torch.sigmoid(outputs)
        else:  # softplus
            g = F.softplus(outputs)

        # Compute positive risk: π * E_P[-log(g(x))]
        # Add epsilon for numerical stability to prevent log(0)
        g_positive = g[positive_mask]
        positive_risk = self.prior * torch.mean(-torch.log(g_positive + self.epsilon))

        # Compute unlabeled risk: E_U[g(x)]
        g_unlabeled = g[unlabeled_mask]
        unlabeled_risk = torch.mean(g_unlabeled)

        # Total PUDRa loss (strictly non-negative and convex)
        loss = positive_risk + unlabeled_risk

        return loss
