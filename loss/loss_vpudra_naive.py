"""VPUDRa-naive: Original PUDRa Loss (No Prior) + VPU MixUp Consistency

This variant combines:
    1. Original PUDRa loss WITHOUT prior weighting (π ignored)
    2. VPU's anchor assumption for MixUp stability
    3. Point Process consistency loss

Key Differences from Other Variants:
    - VPUDRa-Fixed/PP: Use π * E_P[-log p] + E_U[p] (prior-weighted)
    - VPUDRa-naive: Use E_P[-log p + p] + E_U[p] (NO prior weighting)

Mathematical Formulation:
    Base loss (NO PRIOR):
        L_base = E_P[-log p + p] + E_U[p]

        This is the original PUDRa form from the paper:
        - For positives: L(1, p) = -log p + p
        - For unlabeled: L(0, p) = p

    MixUp consistency (with anchor):
        sam_data = λ * x + (1-λ) * p_mix
        μ_anchor = λ * p(x) + (1-λ) * 1.0  # ANCHOR ASSUMPTION

        # Point Process soft label loss (PUDRa-esque):
        L_consistency = -μ_anchor * log(p(sam_data)) + p(sam_data)

    Total loss:
        L = L_base + λ_mixup * L_consistency

Why This Variant:
    - Tests whether ignoring the prior entirely works with MixUp stabilization
    - Uses the symmetric PUDRa formulation: L(1,p) = -log p + p
    - Maintains anchor for stability (learned from VPUDRa-SoftLabel failure)
    - No hyperparameter for prior (simpler than VPUDRa-Fixed)
"""

import torch
from torch import nn


class VPUDRaNaiveLoss(nn.Module):
    """VPUDRa with original PUDRa loss (no prior) and Point Process consistency.

    Uses the original PUDRa formulation without prior weighting,
    combined with VPU's MixUp consistency for regularization.
    """

    def __init__(self, mix_alpha: float = 0.3, epsilon: float = 1e-7):
        super().__init__()
        self.mix_alpha = float(mix_alpha)
        self.epsilon = float(epsilon)
        self.name = "vpudra_naive"

    def forward(self, p_all, pu_labels, p_mix, mu_anchor, lam):
        """Compute VPUDRa-naive loss with original PUDRa form (no prior).

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

        # ===== Point Process Consistency with Anchor =====
        # μ_anchor = λ * p(x) + (1-λ) * 1.0  (computed in trainer, has anchor!)
        # Point Process soft label loss: L(μ, p) = -μ log p + p
        consistency_loss = torch.mean(
            -mu_anchor * torch.log(p_mix + self.epsilon) + p_mix
        )

        # Total loss (weight consistency by lam)
        total_loss = positive_risk + unlabeled_risk + lam * consistency_loss

        return total_loss
