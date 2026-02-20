"""PUDRa-Unified: Elementwise Loss Averaged Over All Samples

This variant computes the PUDRa loss elementwise for each sample, then
averages over ALL samples (not separately over positive and unlabeled).

Mathematical Formulation:
    L_unified = E_all[loss(x, t)]

    where loss(x, t) = {
        -log p + p    if t = 1 (positive)
        p             if t = -1 (unlabeled)
    }

Key Difference from PUDRa-naive:
    - PUDRa-naive: E_P[-log p + p] + E_U[p]
      → Averages over positive, averages over unlabeled, then sums
      → Two separate expectations

    - PUDRa-unified: E_all[loss(x, t)]
      → Computes loss per sample, then averages over all
      → Single expectation over entire batch

Why This Might Matter:
    - Implicit weighting: Batch composition affects loss scale
    - If batch has 100 positives and 1000 unlabeled:
      * PUDRa-naive: positive_risk (avg of 100) + unlabeled_risk (avg of 1000)
      * PUDRa-unified: (sum of all 1100 losses) / 1100
    - Could provide more stable gradients when positive/unlabeled ratio varies

Expected Properties:
    - More stable to batch composition variations
    - Natural weighting by sample count (no manual balancing)
    - Potentially better for imbalanced PU scenarios
"""

import torch
from torch import nn


class PUDRaUnifiedLoss(nn.Module):
    """PUDRa loss computed elementwise and averaged over all samples.

    Implements unified averaging:
        L = mean([loss(x_i, t_i) for all i])

    where loss(x_i, t_i) depends on the label:
        - Positive (t=1): -log p + p
        - Unlabeled (t=-1): p
    """

    def __init__(self, epsilon: float = 1e-7):
        super().__init__()
        self.epsilon = float(epsilon)
        self.name = "pudra_unified"

    def forward(self, outputs: torch.Tensor, pu_labels: torch.Tensor,
                weights: torch.Tensor = None) -> torch.Tensor:
        """Compute PUDRa-unified loss.

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

        # Create masks
        positive_mask = pu_labels == 1
        unlabeled_mask = pu_labels == -1

        # Handle edge case: no positive samples
        if not positive_mask.any():
            return torch.tensor(0.0, device=p.device, requires_grad=True)

        # ===== Elementwise Loss Computation =====
        # Initialize loss tensor (same shape as p)
        elementwise_loss = torch.zeros_like(p)

        # Positive samples: -log p + p
        elementwise_loss[positive_mask] = -torch.log(p[positive_mask] + self.epsilon) + p[positive_mask]

        # Unlabeled samples: p
        elementwise_loss[unlabeled_mask] = p[unlabeled_mask]

        # Average over ALL samples (not separately by group)
        total_loss = torch.mean(elementwise_loss)

        return total_loss
