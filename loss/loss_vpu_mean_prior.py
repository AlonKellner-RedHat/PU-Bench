"""VPU-Mean-Prior Loss with MixUp regularization.

This module implements VPU-Mean with explicit prior weighting.

Mathematical Foundation:
    L_VPU_Mean_Prior = E_all[φ(x)] - π·E_P[log φ(x)] + λ·MixUp_regularization
                     = mean(φ(x)) - π·mean(log φ_p(x)) + λ·MixUp_regularization

    where:
    - φ(x) = σ(f(x)) is the sigmoid-activated model output
    - E_all is expectation over all samples (labeled + unlabeled)
    - E_P is expectation over positive samples only
    - π is the class prior (target_prevalence)
    - λ is the MixUp weight

Key Differences from VPU-Mean:
    - VPU-Mean: mean(φ(x)) - mean(log(φ_p(x)))
    - VPU-Mean-Prior: mean(φ(x)) - π·mean(log(φ_p(x)))
    - Adds explicit prior weighting to the positive term
    - Uses target_prevalence if available, otherwise defaults to π=1

Properties:
    - No log-of-mean variance reduction (uses mean directly)
    - Includes MixUp regularization
    - Prior-weighted positive term
    - Anchor assumption: ∃ points where P(y=1|x) = 1
"""

import torch
from torch import nn


class VPUMeanPriorLoss(nn.Module):
    """VPU-Mean loss with explicit prior weighting and MixUp regularization.

    Implements:
        L = mean(φ(x)) - π·mean(log(φ_p(x))) + λ·MixUp_regularization
    """

    def __init__(self, args, prior):
        """Initialize VPU-Mean-Prior loss.

        Args:
            args: Arguments containing mix_alpha
            prior: Class prior π from dataset statistics
        """
        super(VPUMeanPriorLoss, self).__init__()
        self.mix_alpha = args.mix_alpha
        self.prior = prior
        self.name = "vpu_mean_prior"

    def forward(self, output_phi_all, targets, out_log_phi_all, sam_target, lam):
        """Compute VPU-Mean-Prior loss.

        Args:
            output_phi_all: log(φ(x)) for all samples in batch
            targets: binary labels (1 for positive, 0 for unlabeled)
            out_log_phi_all: log(φ(x_mixed)) for mixed samples
            sam_target: mixed targets for mixup regularization
            lam: mixup weight

        Returns:
            Scalar loss tensor
        """
        log_phi_x = output_phi_all
        log_phi_p = output_phi_all[targets == 1]

        # Handle edge case: no positive samples in batch
        if len(log_phi_p) == 0:
            return torch.tensor(0.0, device=output_phi_all.device, requires_grad=True)

        # Convert log_phi to phi for the mean computation
        phi_all = torch.exp(log_phi_x)

        # Variance reduction term with prior: mean(φ(x)) - π·mean(log(φ_p(x)))
        var_loss = torch.mean(phi_all) - self.prior * torch.mean(log_phi_p)

        # MixUp regularization: E[(log(sam_target) - log(φ(x_mixed)))²]
        reg_mix_log = ((torch.log(sam_target) - out_log_phi_all) ** 2).mean()

        # Total loss
        phi_loss = var_loss + lam * reg_mix_log

        return phi_loss
