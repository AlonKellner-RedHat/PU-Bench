"""Baseline PU loss functions for comparison.

Includes:
- PUDRa-naive: Pure PUDRa without prior or regularization
- VPU-NoMixUp: Pure VPU variance reduction without MixUp
"""

import torch
import torch.nn as nn
import math


class PUDRaNaiveLoss(nn.Module):
    """Pure PUDRa base loss without prior weighting or regularization.

    Implements: L = E_P[-log p + p] + E_U[p]
    """

    def __init__(self, epsilon: float = 1e-7):
        super().__init__()
        self.epsilon = float(epsilon)
        self.name = "pudra_naive"

    def forward(self, outputs: torch.Tensor, pu_labels: torch.Tensor, mode='pu') -> torch.Tensor:
        """Compute PUDRa-naive loss.

        Args:
            outputs: Model logits, shape [N]
            pu_labels: PU labels where 1=positive, -1=unlabeled, shape [N]
            mode: 'pu' (compatibility, ignored)

        Returns:
            Scalar loss tensor
        """
        outputs = outputs.view(-1)
        pu_labels = pu_labels.view(-1)

        # Apply sigmoid to get probabilities
        p = torch.sigmoid(outputs)

        # Separate positive and unlabeled samples
        positive_mask = pu_labels == 1
        unlabeled_mask = pu_labels == -1

        p_positive = p[positive_mask]
        p_unlabeled = p[unlabeled_mask]

        # Handle edge case: no positive samples
        if len(p_positive) == 0:
            return torch.tensor(0.0, device=p.device, requires_grad=True)

        # Positive risk: E_P[-log p + p]
        positive_risk = torch.mean(-torch.log(p_positive + self.epsilon) + p_positive)

        # Unlabeled risk: E_U[p]
        unlabeled_risk = torch.mean(p_unlabeled) if len(p_unlabeled) > 0 else 0.0

        # Total loss
        total_loss = positive_risk + unlabeled_risk

        return total_loss


class VPUNoMixUpLoss(nn.Module):
    """Pure VPU loss without MixUp regularization.

    Implements: L = log(E_all[φ(x)]) - E_P[log φ(x)]
    """

    def __init__(self, epsilon: float = 1e-7):
        super().__init__()
        self.epsilon = float(epsilon)
        self.name = "vpu_nomixup"

    def forward(self, outputs: torch.Tensor, pu_labels: torch.Tensor, mode='pu') -> torch.Tensor:
        """Compute pure VPU loss without MixUp.

        Args:
            outputs: Model logits, shape [N]
            pu_labels: PU labels where 1=positive, -1=unlabeled, shape [N]
            mode: 'pu' (compatibility, ignored)

        Returns:
            Scalar loss tensor
        """
        outputs = outputs.view(-1)
        pu_labels = pu_labels.view(-1)

        # Convert to log(sigmoid(outputs)) for numerical stability
        # log(sigmoid(x)) = -log(1 + exp(-x)) = -softplus(-x)
        log_phi_all = -nn.functional.softplus(-outputs)

        # Separate positive samples (pu_labels == 1)
        positive_mask = pu_labels == 1
        log_phi_p = log_phi_all[positive_mask]

        # Handle edge case: no positive samples
        if len(log_phi_p) == 0:
            return torch.tensor(0.0, device=log_phi_all.device, requires_grad=True)

        # Variance reduction loss: log(mean(φ(x))) - mean(log(φ_p(x)))
        # = logsumexp(log φ(x)) - log(N) - mean(log φ_p(x))
        var_loss = (
            torch.logsumexp(log_phi_all, dim=0)
            - math.log(len(log_phi_all))
            - torch.mean(log_phi_p)
        )

        return var_loss
