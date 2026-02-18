"""VPUDRa-naive-logmae: Original PUDRa Loss (No Prior) + VPU Log-MAE Consistency

This variant combines:
    1. Original PUDRa loss WITHOUT prior weighting (π ignored)
    2. VPU's anchor assumption for MixUp stability
    3. Log-MAE consistency loss (Mean Absolute Error in log-space, NOT MSE)

Key Differences:
    - VPUDRa-naive-logmse: E_P[-log p + p] + E_U[p] + (log μ - log p)²  (squared)
    - VPUDRa-naive-logmae: E_P[-log p + p] + E_U[p] + |log μ - log p|  (absolute)

Mathematical Formulation:
    Base loss (NO PRIOR):
        L_base = E_P[-log p + p] + E_U[p]

        This is the original PUDRa form from the paper:
        - For positives: L(1, p) = -log p + p
        - For unlabeled: L(0, p) = p

    MixUp consistency (Log-MAE):
        sam_data = λ * x + (1-λ) * p_mix
        μ_anchor = λ * p(x) + (1-λ) * 1.0  # ANCHOR ASSUMPTION

        # Log-MAE consistency (L1 penalty in log-space):
        L_consistency = |log(μ_anchor) - log(p(sam_data))|

    Total loss:
        L = L_base + λ_mixup * L_consistency

Why This Variant:
    - Tests whether squared penalty (MSE) is necessary in log-space
    - MAE is more robust to outliers than MSE
    - May provide smoother gradients (less penalty on small errors, more on large)
    - Completes exploration: Point Process vs Log-MSE vs Log-MAE
"""

import torch
from torch import nn


class VPUDRaNaiveLogMAELoss(nn.Module):
    """VPUDRa with original PUDRa loss (no prior) and Log-MAE consistency.

    Uses the original PUDRa formulation without prior weighting,
    combined with Log-MAE (L1 in log-space) consistency for regularization.
    """

    def __init__(self, mix_alpha: float = 0.3, epsilon: float = 1e-7):
        super().__init__()
        self.mix_alpha = float(mix_alpha)
        self.epsilon = float(epsilon)
        self.name = "vpudra_naive_logmae"

    def forward(self, p_all, pu_labels, p_mix, mu_anchor, lam):
        """Compute VPUDRa-naive-logmae loss.

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

        # ===== Log-MAE Consistency with Anchor =====
        # μ_anchor = λ * p(x) + (1-λ) * 1.0  (computed in trainer, has anchor!)
        # Log-MAE: |log μ - log p| (L1 in log-space, NOT L2)
        consistency_loss = torch.mean(
            torch.abs(
                torch.log(mu_anchor + self.epsilon) - torch.log(p_mix + self.epsilon)
            )
        )

        # Total loss (weight consistency by lam)
        total_loss = positive_risk + unlabeled_risk + lam * consistency_loss

        return total_loss
