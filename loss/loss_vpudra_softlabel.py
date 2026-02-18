"""VPUDRa with Point Process Soft Label Consistency

This variant uses PUDRa's Point Process loss structure for MixUp consistency
instead of VPU's log-MSE formulation.

Theoretical Motivation:
    For a mixed sample with expected label μ ∈ [0,1], the Point Process loss
    naturally extends via weighted combination:

    L(y=1, g) = -log g + g
    L(y=0, g) = g
    L(μ, g) = μ * L(1, g) + (1-μ) * L(0, g) = -μ log g + g

Key Differences from VPU:
    - No anchor assumption: Uses predicted probabilities, not assumed p=1
    - Point Process consistency: -μ log g + g instead of (log μ - log g)²
    - Asymmetric penalty: Inherits PUDRa's structure

Mathematical Formulation:
    sam_data = λ * x + (1-λ) * p_mix
    μ = λ * p(x) + (1-λ) * p(p_mix)  # NO anchor assumption
    L_consistency = -μ log p(sam_data) + p(sam_data)

    Total loss:
    L = π * E_P[-log p] + E_U[p] + λ_mixup * L_consistency
"""

import torch
from torch import nn


class VPUDRaSoftLabelLoss(nn.Module):
    """VPUDRa with Point Process soft label consistency (no anchor assumption).

    Uses PUDRa's Point Process loss structure for MixUp consistency instead of
    VPU's log-MSE, and doesn't assume p(positive) = 1.
    """

    def __init__(self, prior: float, mix_alpha: float = 0.3, epsilon: float = 1e-7):
        super().__init__()
        self.prior = float(prior)
        self.mix_alpha = float(mix_alpha)
        self.epsilon = float(epsilon)
        self.name = "vpudra_softlabel"

    def forward(self, p_all, pu_labels, p_mix, p_orig, p_pos_mix, lam):
        """Compute VPUDRa loss with Point Process soft label consistency.

        Args:
            p_all: Probabilities p = σ(f(x)) for all samples, shape [N]
            pu_labels: PU labels (1=positive, -1=unlabeled), shape [N]
            p_mix: Probabilities for mixed samples σ(f(λ*x + (1-λ)*p_mix)), shape [N]
            p_orig: Probabilities for original samples (detached), shape [N]
            p_pos_mix: Probabilities for positive samples used in mixing (detached), shape [N]
            lam: Mixing coefficient (scalar float)

        Returns:
            Scalar loss tensor
        """
        # Separate positive and unlabeled samples
        p_positive = p_all[pu_labels == 1]
        p_unlabeled = p_all[pu_labels == -1]

        # Handle edge case: no positive samples
        if len(p_positive) == 0:
            return torch.tensor(0.0, device=p_all.device, requires_grad=True)

        # ===== PUDRa Base Loss =====
        # Positive risk: π * E_P[-log p]
        positive_risk = self.prior * torch.mean(-torch.log(p_positive + self.epsilon))

        # Unlabeled risk: E_U[p]
        unlabeled_risk = torch.mean(p_unlabeled) if len(p_unlabeled) > 0 else 0.0

        # ===== Point Process Soft Label Consistency =====
        # Soft label (expected label under mixture, NO anchor assumption)
        mu = lam * p_orig + (1 - lam) * p_pos_mix  # both detached

        # Point Process loss on mixed samples: L(μ, p) = -μ log p + p
        # This is the natural extension of PUDRa's loss to soft labels!
        consistency_loss = torch.mean(-mu * torch.log(p_mix + self.epsilon) + p_mix)

        # Total loss (weight consistency by lam, like VPU)
        total_loss = positive_risk + unlabeled_risk + lam * consistency_loss

        return total_loss
