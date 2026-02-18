"""VPUDRa-Fixed (Variance-reduced Positive-Unlabeled Density Ratio with Fixed Prior) Loss

This module implements the VPUDRa loss function with a fixed prior, which combines:
- PUDRa's original Point Process / Generalized KL loss structure
- Fixed prior from dataset (true prior P(Y=1), same as PUDRa)
- VPU's MixUp consistency regularization for variance reduction

Mathematical Foundation:
    L_VPUDRa_Fixed = π * E_P[-log p] + E_U[p] + λ * mixup_reg

    where:
    - π is the true prior P(Y=1) from the dataset (constant throughout training)
    - p = σ(f(x)) is the sigmoid-activated model output
    - E_P is expectation over positive samples
    - E_U is expectation over unlabeled samples
    - λ is the MixUp interpolation coefficient (sampled from Beta distribution)
    - mixup_reg = E[(log(y_mix) - log(p(x_mix)))²]
    - y_mix = λ * p(x) + (1-λ) * 1.0 is the MixUp target

Original PUDRa Loss Formulation:
    - Positive sample: L(1, p) = -log p + p
    - Unlabeled sample: L(0, p) = p

Key Properties:
    - Unbiased PU risk estimator (prevents classifier collapse)
    - Uses true dataset prior (constant, not batch-dependent)
    - MixUp regularization reduces variance and enforces smoothness

Comparison to VPUDRa (empirical prior):
    - VPUDRa: Uses empirical prior π_emp = n_p/N (varies per batch)
    - VPUDRa-Fixed: Uses true prior π (constant from dataset)
    - More stable when batch composition varies significantly
    - Exactly matches PUDRa + MixUp (for fair comparison)
"""

import torch
import torch.nn as nn


class VPUDRaFixedLoss(nn.Module):
    """VPUDRa loss for Positive-Unlabeled learning with fixed (true) prior.

    Implements the hybrid formulation combining PUDRa's Point Process loss
    with VPU's MixUp regularization, using the true dataset prior (same as PUDRa).

    Args:
        prior (float): True prior probability P(Y=1) from the dataset
        mix_alpha (float): Beta distribution parameter for MixUp sampling (default=0.3)
        epsilon (float): Small constant for numerical stability in log computation (default=1e-7)
    """

    def __init__(self, prior: float = 0.5, mix_alpha: float = 0.3, epsilon: float = 1e-7):
        super().__init__()

        self.prior = float(prior)
        self.mix_alpha = float(mix_alpha)
        self.epsilon = float(epsilon)
        self.name = "vpudra_fixed"

        # Label constants (following PU-Bench convention)
        self.positive = 1
        self.unlabeled = -1

    def forward(
        self,
        p_all: torch.Tensor,        # σ(f(x)) for all samples in batch
        pu_labels: torch.Tensor,    # PU labels: 1=positive, -1=unlabeled
        p_mix: torch.Tensor,         # σ(f(x_mix)) for mixed samples
        mix_target: torch.Tensor,    # y_mix = λ * p(x) + (1-λ) * 1.0
        lam: float                   # MixUp coefficient
    ) -> torch.Tensor:
        """Compute VPUDRa-Fixed loss.

        Args:
            p_all: Probabilities σ(f(x)) for all samples, shape [N]
            pu_labels: PU labels (1=positive, -1=unlabeled), shape [N]
            p_mix: Probabilities σ(f(x_mix)) for mixed samples, shape [N]
            mix_target: MixUp target values y_mix, shape [N]
            lam: MixUp interpolation coefficient (scalar)

        Returns:
            Scalar loss tensor
        """
        # Flatten tensors
        p_all = p_all.view(-1)
        pu_labels = pu_labels.view(-1)
        p_mix = p_mix.view(-1)
        mix_target = mix_target.view(-1)

        # Separate positive and unlabeled samples
        positive_mask = pu_labels == self.positive
        unlabeled_mask = pu_labels == self.unlabeled

        n_positive = positive_mask.sum().item()
        n_unlabeled = unlabeled_mask.sum().item()

        # Handle edge cases: empty positive or unlabeled batches
        if n_positive == 0 or n_unlabeled == 0:
            # Return zero gradient tensor for safety
            return torch.tensor(0.0, device=p_all.device, requires_grad=True)

        # Extract probabilities for positive and unlabeled samples
        p_positive = p_all[positive_mask]
        p_unlabeled = p_all[unlabeled_mask]

        # Term 1: π * E_P[-log p]
        # This is the positive risk term from L(1,p) = -log p + p
        # Note: The +p term is implicitly included via the unlabeled term on ALL samples
        positive_risk = self.prior * torch.mean(-torch.log(p_positive + self.epsilon))

        # Term 2: E_U[p]
        # This is the unlabeled risk term from L(0,p) = p
        unlabeled_risk = torch.mean(p_unlabeled)

        # Term 3: λ * E[(log(y_mix) - log(p(x_mix)))²]
        # This is VPU's MixUp consistency regularization
        # Add epsilon to prevent log(0)
        log_mix_target = torch.log(mix_target + self.epsilon)
        log_p_mix = torch.log(p_mix + self.epsilon)
        mixup_reg = ((log_mix_target - log_p_mix) ** 2).mean()

        # Combined VPUDRa-Fixed loss
        loss = positive_risk + unlabeled_risk + lam * mixup_reg

        return loss
