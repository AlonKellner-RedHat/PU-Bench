"""VPUDRa with Manifold Smoothness Regularization

This variant enforces smoothness by requiring that predictions on mixed samples
follow convex combinations of the original predictions, WITHOUT anchor assumptions.

Theoretical Motivation:
    Under the manifold hypothesis, if we mix two samples in feature space:
        sam_data = λ * x1 + (1-λ) * x2

    The prediction should approximately follow convex combination:
        p(sam_data) ≈ λ * p(x1) + (1-λ) * p(x2)

    This is a weaker assumption than VPU's anchor assumption and enforces
    smoothness via consistency rather than assuming p(positive) = 1.

Key Differences from VPU:
    - No anchor assumption: Uses actual predictions, not assumed p=1
    - Direct smoothness: Enforces convex combination property
    - Flexible: Can be in probability space or log-space

Mathematical Formulation:
    sam_data = λ * x + (1-λ) * x_mix
    p_target = λ * p(x) + (1-λ) * p(x_mix)  # detached predictions

    Smoothness loss (choose one):
    1. Probability space: |p(sam_data) - p_target|²
    2. Log space: |log p(sam_data) - log p_target|²
    3. KL divergence: KL(p_target || p(sam_data))

    Total loss:
    L = π * E_P[-log p] + E_U[p] + λ_mixup * L_smoothness
"""

import torch
from torch import nn


class VPUDRaManifoldLoss(nn.Module):
    """VPUDRa with manifold smoothness (convex combination consistency).

    Enforces that predictions on mixed samples follow convex combinations
    without assuming anchor points.
    """

    def __init__(
        self,
        prior: float,
        mix_alpha: float = 0.3,
        epsilon: float = 1e-7,
        smoothness_type: str = "log",
    ):
        """Initialize VPUDRa manifold smoothness loss.

        Args:
            prior: Class prior P(Y=1)
            mix_alpha: Beta distribution parameter for MixUp
            epsilon: Numerical stability constant
            smoothness_type: Type of smoothness penalty
                - "prob": |p(mix) - p_target|² in probability space
                - "log": |log p(mix) - log p_target|² in log space
                - "kl": KL(p_target || p(mix)) divergence
        """
        super().__init__()
        self.prior = float(prior)
        self.mix_alpha = float(mix_alpha)
        self.epsilon = float(epsilon)
        self.smoothness_type = smoothness_type
        self.name = "vpudra_manifold"

        if smoothness_type not in ["prob", "log", "kl"]:
            raise ValueError(
                f"smoothness_type must be 'prob', 'log', or 'kl', got '{smoothness_type}'"
            )

    def _smoothness_loss(self, p_pred, p_target):
        """Compute smoothness penalty based on type."""
        if self.smoothness_type == "prob":
            # MSE in probability space
            return torch.mean((p_pred - p_target) ** 2)
        elif self.smoothness_type == "log":
            # MSE in log space (KL-divergence flavor)
            return torch.mean(
                (
                    torch.log(p_pred + self.epsilon)
                    - torch.log(p_target + self.epsilon)
                )
                ** 2
            )
        else:  # kl
            # KL divergence: KL(p_target || p_pred)
            # For Bernoulli: p*log(p/q) + (1-p)*log((1-p)/(1-q))
            p_target_safe = torch.clamp(p_target, self.epsilon, 1 - self.epsilon)
            p_pred_safe = torch.clamp(p_pred, self.epsilon, 1 - self.epsilon)

            kl = p_target_safe * torch.log(p_target_safe / p_pred_safe) + (
                1 - p_target_safe
            ) * torch.log((1 - p_target_safe) / (1 - p_pred_safe))
            return torch.mean(kl)

    def forward(self, p_all, pu_labels, p_mix, p_x, p_mix_partner, lam):
        """Compute VPUDRa loss with manifold smoothness.

        Args:
            p_all: Probabilities for all samples, shape [N]
            pu_labels: PU labels (1=positive, -1=unlabeled), shape [N]
            p_mix: Probabilities for mixed samples, shape [N]
            p_x: Probabilities for x (detached), shape [N]
            p_mix_partner: Probabilities for mixing partner (detached), shape [N]
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

        # ===== Manifold Smoothness =====
        # Target: convex combination of original predictions (detached)
        p_target = lam * p_x + (1 - lam) * p_mix_partner

        # Smoothness penalty: enforce p(mix) ≈ p_target
        smoothness_loss = self._smoothness_loss(p_mix, p_target)

        # Total loss (weight by lam)
        total_loss = positive_risk + unlabeled_risk + lam * smoothness_loss

        return total_loss
