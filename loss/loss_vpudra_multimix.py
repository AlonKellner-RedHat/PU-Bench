"""VPUDRa with Multi-Strategy Mixing

Unlike VPU which only mixes with positive samples, this variant uses three
mixing strategies aligned with PUDRa's treatment of positive and unlabeled data.

Theoretical Motivation:
    PUDRa treats positives and unlabeled differently. We extend this to MixUp:

    1. Positive-Positive mixing:
       - sam = λ * p1 + (1-λ) * p2  (both positive)
       - Expected label ≈ 1
       - Consistency: L(1, p(sam)) = -log p(sam) + p(sam)

    2. Positive-Unlabeled mixing:
       - sam = λ * p + (1-λ) * u
       - Expected label ≈ λ * 1 + (1-λ) * p(u)
       - Consistency: L(μ, p(sam)) = -μ log p(sam) + p(sam)

    3. Unlabeled-Unlabeled mixing:
       - sam = λ * u1 + (1-λ) * u2
       - Expected label ≈ λ * p(u1) + (1-λ) * p(u2)
       - Consistency: L(μ, p(sam)) = -μ log p(sam) + p(sam)

Key Differences from VPU:
    - Three mixing strategies instead of one (positive-only)
    - Different expected labels for each strategy
    - More thorough manifold smoothness enforcement

Mathematical Formulation:
    L_consistency = 1/3 * (L_pp + L_pu + L_uu)

    where each term uses Point Process soft label loss:
    L(μ, p) = -μ log p + p
"""

import torch
from torch import nn


class VPUDRaMultiMixLoss(nn.Module):
    """VPUDRa with multi-strategy mixing (P-P, P-U, U-U).

    Uses three different mixing strategies to enforce smoothness across
    different regions of the manifold, aligned with PUDRa's philosophy.
    """

    def __init__(self, prior: float, mix_alpha: float = 0.3, epsilon: float = 1e-7):
        super().__init__()
        self.prior = float(prior)
        self.mix_alpha = float(mix_alpha)
        self.epsilon = float(epsilon)
        self.name = "vpudra_multimix"

    def _point_process_soft(self, mu, p):
        """Point Process loss with soft label: L(μ, p) = -μ log p + p"""
        return -mu * torch.log(p + self.epsilon) + p

    def forward(
        self,
        p_all,
        pu_labels,
        p_pp_mix,
        p_pu_mix,
        p_uu_mix,
        mu_pp,
        mu_pu,
        mu_uu,
        lam,
    ):
        """Compute VPUDRa loss with multi-strategy mixing.

        Args:
            p_all: Probabilities for all samples, shape [N]
            pu_labels: PU labels (1=positive, -1=unlabeled), shape [N]
            p_pp_mix: Probabilities for P-P mixed samples, shape [n_pos]
            p_pu_mix: Probabilities for P-U mixed samples, shape [N]
            p_uu_mix: Probabilities for U-U mixed samples, shape [n_unlabeled]
            mu_pp: Soft labels for P-P mixing (≈1.0), shape [n_pos]
            mu_pu: Soft labels for P-U mixing, shape [N]
            mu_uu: Soft labels for U-U mixing, shape [n_unlabeled]
            lam: Mixing coefficient (scalar float)

        Returns:
            Scalar loss tensor
        """
        # Separate positive and unlabeled samples
        p_positive = p_all[pu_labels == 1]
        p_unlabeled = p_all[pu_labels == -1]

        if len(p_positive) == 0:
            return torch.tensor(0.0, device=p_all.device, requires_grad=True)

        # ===== PUDRa Base Loss =====
        positive_risk = self.prior * torch.mean(-torch.log(p_positive + self.epsilon))
        unlabeled_risk = torch.mean(p_unlabeled) if len(p_unlabeled) > 0 else 0.0

        # ===== Multi-Strategy Consistency =====
        # 1. Positive-Positive mixing (expect label ≈ 1)
        consistency_pp = torch.mean(self._point_process_soft(mu_pp, p_pp_mix))

        # 2. Positive-Unlabeled mixing (expect label intermediate)
        consistency_pu = torch.mean(self._point_process_soft(mu_pu, p_pu_mix))

        # 3. Unlabeled-Unlabeled mixing (expect label ≈ prior)
        consistency_uu = (
            torch.mean(self._point_process_soft(mu_uu, p_uu_mix))
            if len(p_unlabeled) > 0
            else 0.0
        )

        # Average consistency across strategies
        consistency_loss = (consistency_pp + consistency_pu + consistency_uu) / 3.0

        # Total loss (weight by lam)
        total_loss = positive_risk + unlabeled_risk + lam * consistency_loss

        return total_loss
