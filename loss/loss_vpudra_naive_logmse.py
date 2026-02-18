"""VPUDRa-naive-logmse: Original PUDRa Loss (No Prior) + VPU Log-MSE Consistency

This variant combines:
    1. Original PUDRa loss WITHOUT prior weighting (π ignored)
    2. VPU's anchor assumption for MixUp stability
    3. VPU's log-MSE consistency loss (NOT Point Process)

Key Differences:
    - VPUDRa-Fixed: π * E_P[-log p] + E_U[p] + log-MSE consistency (WITH prior)
    - VPUDRa-naive-logmse: E_P[-log p + p] + E_U[p] + log-MSE consistency (NO prior)
    - VPUDRa-naive: E_P[-log p + p] + E_U[p] + Point Process consistency

Mathematical Formulation:
    Base loss (NO PRIOR):
        L_base = E_P[-log p + p] + E_U[p]

        This is the original PUDRa form from the paper:
        - For positives: L(1, p) = -log p + p
        - For unlabeled: L(0, p) = p

    MixUp consistency (VPU's log-MSE):
        sam_data = λ * x + (1-λ) * p_mix
        μ_anchor = λ * p(x) + (1-λ) * 1.0  # ANCHOR ASSUMPTION

        # VPU's log-MSE consistency (symmetric):
        L_consistency = (log(μ_anchor) - log(p(sam_data)))²

    Total loss:
        L = L_base + λ_mixup * L_consistency

Why This Variant:
    - Tests whether VPU's log-MSE consistency works better than Point Process
      when no prior weighting is used
    - Completes the 2x2 design matrix (prior vs no-prior, log-MSE vs Point Process)
    - Most similar to VPU itself (same consistency), but uses original PUDRa base loss
"""

import torch
from torch import nn


class VPUDRaNaiveLogMSELoss(nn.Module):
    """VPUDRa with original PUDRa loss (no prior) and VPU's log-MSE consistency.

    Uses the original PUDRa formulation without prior weighting,
    combined with VPU's log-MSE consistency for regularization.
    """

    def __init__(self, mix_alpha: float = 0.3, epsilon: float = 1e-7):
        super().__init__()
        self.mix_alpha = float(mix_alpha)
        self.epsilon = float(epsilon)
        self.name = "vpudra_naive_logmse"

    def forward(self, p_all, pu_labels, p_mix, mu_anchor, lam):
        """Compute VPUDRa-naive-logmse loss.

        Args:
            p_all: Probabilities p = σ(f(x)) for all samples, shape [N]
            pu_labels: PU labels (1=positive, -1=unlabeled), shape [N]
            p_mix: Probabilities for mixed samples σ(f(λ*x + (1-λ)*p_mix)), shape [N]
            mu_anchor: Anchored MixUp targets λ*p(x) + (1-λ)*1.0, shape [N]
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

        # ===== Original PUDRa Loss (NO PRIOR) =====
        # Positive risk: E_P[-log p + p] (NO π weighting!)
        # This is the original L(1, p) = -log p + p from PUDRa
        positive_risk = torch.mean(-torch.log(p_positive + self.epsilon) + p_positive)

        # Unlabeled risk: E_U[p]
        # This is the original L(0, p) = p from PUDRa
        unlabeled_risk = torch.mean(p_unlabeled) if len(p_unlabeled) > 0 else 0.0

        # ===== VPU's Log-MSE Consistency with Anchor =====
        # μ_anchor = λ * p(x) + (1-λ) * 1.0  (computed in trainer, has anchor!)
        # VPU's symmetric log-MSE: (log μ - log p)²
        consistency_loss = torch.mean(
            (torch.log(mu_anchor + self.epsilon) - torch.log(p_mix + self.epsilon)) ** 2
        )

        # Total loss (weight consistency by lam)
        total_loss = positive_risk + unlabeled_risk + lam * consistency_loss

        return total_loss
