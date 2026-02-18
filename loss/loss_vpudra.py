"""VPUDRa (Variance-reduced Positive-Unlabeled Density Ratio) Loss

This module implements the VPUDRa loss function, which combines:
- PUDRa's original Point Process / Generalized KL loss structure
- Empirical prior estimation (no manual hyperparameter tuning)
- VPU's MixUp consistency regularization for variance reduction

Mathematical Foundation:
    L_VPUDRa = π_emp * E_P[-log p] + E_U[p] + λ * mixup_reg

    where:
    - π_emp = n_p / N is the empirical prior (fraction of positives in batch)
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
    - Data-driven prior (no manual tuning needed)
    - Under SCAR assumption, π_emp is unbiased estimator of true prior
    - MixUp regularization reduces variance and enforces smoothness
"""

import torch
import torch.nn as nn


class VPUDRaLoss(nn.Module):
    """VPUDRa loss for Positive-Unlabeled learning with empirical prior.

    Implements the hybrid formulation combining PUDRa's Point Process loss
    with VPU's MixUp regularization, using empirical prior estimation.

    Args:
        mix_alpha (float): Beta distribution parameter for MixUp sampling (default=0.3)
        epsilon (float): Small constant for numerical stability in log computation (default=1e-7)
    """

    def __init__(self, mix_alpha: float = 0.3, epsilon: float = 1e-7):
        super().__init__()

        self.mix_alpha = float(mix_alpha)
        self.epsilon = float(epsilon)
        self.name = "vpudra"

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
        """Compute VPUDRa loss.

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
        N = len(pu_labels)

        # Handle edge cases: empty positive or unlabeled batches
        if n_positive == 0 or n_unlabeled == 0:
            # Return zero gradient tensor for safety
            return torch.tensor(0.0, device=p_all.device, requires_grad=True)

        # Compute empirical prior: π_emp = n_p / N
        pi_emp = n_positive / N

        # Extract probabilities for positive and unlabeled samples
        p_positive = p_all[positive_mask]
        p_unlabeled = p_all[unlabeled_mask]

        # Term 1: π_emp * E_P[-log p]
        # This is the positive risk term from L(1,p) = -log p + p
        # Note: The +p term is implicitly included via the unlabeled term on ALL samples
        positive_risk = pi_emp * torch.mean(-torch.log(p_positive + self.epsilon))

        # Term 2: E_U[p]
        # This is the unlabeled risk term from L(0,p) = p
        unlabeled_risk = torch.mean(p_unlabeled)

        # Term 3: λ * E[(log(y_mix) - log(p(x_mix)))²]
        # This is VPU's MixUp consistency regularization
        # Add epsilon to prevent log(0)
        log_mix_target = torch.log(mix_target + self.epsilon)
        log_p_mix = torch.log(p_mix + self.epsilon)
        mixup_reg = ((log_mix_target - log_p_mix) ** 2).mean()

        # Combined VPUDRa loss
        loss = positive_risk + unlabeled_risk + lam * mixup_reg

        return loss
