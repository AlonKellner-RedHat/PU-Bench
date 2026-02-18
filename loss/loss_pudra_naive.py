"""PUDRa-naive: Original PUDRa Loss Without Prior and Without Regularization

This is the pure base loss from PUDRa without:
    1. Prior weighting (π ignored)
    2. MixUp consistency regularization

Mathematical Formulation:
    L_base = E_P[-log p + p] + E_U[p]

    This is the original symmetric PUDRa form:
    - For positives: L(1, p) = -log p + p
    - For unlabeled: L(0, p) = p

Key Differences from Other Variants:
    - PUDRa: Uses π * E_P[-log p] + E_U[p] (WITH prior weighting)
    - VPUDRa-naive: Uses E_P[-log p + p] + E_U[p] + consistency (WITH regularization)
    - PUDRa-naive: Uses E_P[-log p + p] + E_U[p] (NO prior, NO regularization)

Why This Variant:
    - Tests the pure base loss without any enhancements
    - Isolates the effect of regularization (compare to VPUDRa-naive)
    - Simplest possible PU density ratio formulation
"""

import torch
from torch import nn


class PUDRaNaiveLoss(nn.Module):
    """Pure PUDRa base loss without prior weighting or regularization.

    Implements the symmetric PUDRa formulation without prior:
        L = E_P[-log p + p] + E_U[p]
    """

    def __init__(self, epsilon: float = 1e-7):
        super().__init__()
        self.epsilon = float(epsilon)
        self.name = "pudra_naive"

    def forward(self, outputs: torch.Tensor, pu_labels: torch.Tensor,
                weights: torch.Tensor = None) -> torch.Tensor:
        """Compute PUDRa-naive loss.

        Args:
            outputs: Model logits, shape [N]
            pu_labels: PU labels where 1=positive, -1=unlabeled, shape [N]
            weights: Optional sample weights, shape [N] (currently unused)

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

        # ===== Original PUDRa Base Loss (NO PRIOR, NO REGULARIZATION) =====
        # Positive risk: E_P[-log p + p] (symmetric form, NO π weighting!)
        positive_risk = torch.mean(-torch.log(p_positive + self.epsilon) + p_positive)

        # Unlabeled risk: E_U[p]
        unlabeled_risk = torch.mean(p_unlabeled) if len(p_unlabeled) > 0 else 0.0

        # Total loss
        total_loss = positive_risk + unlabeled_risk

        return total_loss
