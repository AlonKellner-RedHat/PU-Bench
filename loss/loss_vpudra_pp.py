"""VPUDRa with Point Process Consistency (Anchor Preserved)

This variant combines:
    1. PUDRa's Point Process loss structure for consistency
    2. VPU's anchor assumption for stability

Key Difference from VPUDRa-Fixed:
    - VPUDRa-Fixed uses VPU's log-MSE: (log(target) - log(pred))²
    - VPUDRa-PP uses Point Process: -target * log(pred) + pred

Key Difference from VPUDRa-SoftLabel:
    - VPUDRa-SoftLabel uses soft target: μ = λ*p(x) + (1-λ)*p(p_mix) (unstable!)
    - VPUDRa-PP uses anchored target: μ = λ*p(x) + (1-λ)*1.0 (stable!)

Mathematical Formulation:
    Base loss (same as all VPUDRa variants):
        L_base = π * E_P[-log p] + E_U[p]

    MixUp consistency (CHANGED):
        sam_data = λ * x + (1-λ) * p_mix
        μ_anchor = λ * p(x) + (1-λ) * 1.0  # ANCHOR ASSUMPTION

        # Point Process soft label loss (PUDRa-esque):
        L_consistency = -μ_anchor * log(p(sam_data)) + p(sam_data)

    Total loss:
        L = L_base + λ_mixup * L_consistency

Why This Should Work:
    - Anchor prevents collapse (learned from VPUDRa-SoftLabel failure)
    - Point Process aligns with PUDRa's theoretical framework
    - Asymmetric penalty may provide different regularization than log-MSE
"""

import torch
from torch import nn


class VPUDRaPointProcessLoss(nn.Module):
    """VPUDRa with Point Process consistency loss and anchor assumption.

    Uses PUDRa's Point Process structure for MixUp consistency while
    maintaining VPU's stabilizing anchor assumption.
    """

    def __init__(self, prior: float, mix_alpha: float = 0.3, epsilon: float = 1e-7):
        super().__init__()
        self.prior = float(prior)
        self.mix_alpha = float(mix_alpha)
        self.epsilon = float(epsilon)
        self.name = "vpudra_pp"

    def forward(self, p_all, pu_labels, p_mix, mu_anchor, lam):
        """Compute VPUDRa loss with Point Process consistency (anchored).

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

        # ===== PUDRa Base Loss =====
        # Positive risk: π * E_P[-log p]
        positive_risk = self.prior * torch.mean(-torch.log(p_positive + self.epsilon))

        # Unlabeled risk: E_U[p]
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
