"""PUDRaSB (Positive-Unlabeled Density Ratio with Selection Bias) Loss

This module implements the PUDRaSB loss function, which combines:
- PUDRa's elegant Point Process/Generalized KL loss formulation
- nnPUSB's scalar propensity weighting for selection bias handling

Mathematical Foundation:
    L_PUDRaSB = w * π * E_P[-log(g(x))] + E_U[g(x)]

    where:
    - w is the propensity weight (scalar, default=1.0 for SCAR)
    - π is the class prior P(y=1)
    - g(x) is a non-negative function (sigmoid or softplus of model output)
    - E_P is expectation over positive samples
    - E_U is expectation over unlabeled samples

Key Properties:
    - Strictly convex and non-negative (like PUDRa)
    - No clipping constraints required
    - Under SCAR with w=1.0: identical to PUDRa
    - Ready for SAR: can tune w for bias correction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PUDRaSBLoss(nn.Module):
    """PUDRaSB loss for Positive-Unlabeled learning with selection bias handling.

    Implements the weighted Point Process / Generalized KL formulation:
        L = w * π * E_P[-log(g(x))] + E_U[g(x)]

    Args:
        prior (float): Class prior π = P(y=1), must be in (0, 1)
        activation (str): Activation function for g(x):
            - "sigmoid": g(x) = sigmoid(x), bounded [0,1], converges to P(y=1|x)
            - "softplus": g(x) = softplus(x), unbounded [0,∞), converges to density ratio
        weight (float): Propensity weight for selection bias correction (default=1.0 for SCAR)
        epsilon (float): Small constant for numerical stability in log computation
    """

    def __init__(
        self,
        prior: float,
        activation: str = "sigmoid",
        weight: float = 1.0,
        epsilon: float = 1e-7,
    ):
        super().__init__()

        if not 0 < prior < 1:
            raise ValueError(f"Class prior must be in (0, 1), got {prior}")

        if activation not in ["sigmoid", "softplus"]:
            raise ValueError(
                f"activation must be 'sigmoid' or 'softplus', got '{activation}'"
            )

        if weight <= 0:
            raise ValueError(f"weight must be positive, got {weight}")

        self.prior = float(prior)
        self.activation = activation
        self.weight = float(weight)
        self.epsilon = float(epsilon)

        # Label constants (following PU-Bench convention)
        self.positive = 1
        self.unlabeled = -1

    def forward(
        self,
        outputs: torch.Tensor,
        pu_labels: torch.Tensor,
        weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute PUDRaSB loss.

        Args:
            outputs: Model logits, shape [N]
            pu_labels: PU labels where 1=positive, -1=unlabeled, shape [N]
            weights: Optional instance-dependent weights, shape [N] (currently unused,
                     uses scalar self.weight instead)

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

        # Compute positive risk: w * π * E_P[-log(g(x))]
        # Add epsilon for numerical stability to prevent log(0)
        g_positive = g[positive_mask]
        positive_risk = (
            self.weight * self.prior * torch.mean(-torch.log(g_positive + self.epsilon))
        )

        # Compute unlabeled risk: E_U[g(x)]
        g_unlabeled = g[unlabeled_mask]
        unlabeled_risk = torch.mean(g_unlabeled)

        # Total PUDRaSB loss (strictly non-negative and convex)
        loss = positive_risk + unlabeled_risk

        return loss
