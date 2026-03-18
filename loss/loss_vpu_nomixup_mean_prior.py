"""VPU-NoMixUp-Mean-Prior Loss (VPU without MixUp, with prior weighting)

This module implements a variant of VPU-NoMixUp-Mean with explicit prior weighting.

Mathematical Foundation:
    L_VPU_NoMixUp_Mean_Prior = E_all[φ(x)] - π·E_P[log φ(x)]
                              = mean(φ(x)) - π·mean(log φ_p(x))

    where:
    - φ(x) = σ(f(x)) is the sigmoid-activated model output
    - E_all is expectation over all samples (labeled + unlabeled)
    - E_P is expectation over positive samples only
    - π is the class prior (from dataset statistics)

Key Differences:
    - VPU-NoMixUp-Mean: mean(φ(x)) - mean(log(φ_p(x)))
    - VPU-NoMixUp-Mean-Prior: mean(φ(x)) - π·mean(log(φ_p(x)))
    - Adds explicit prior weighting to the positive term
    - NO MixUp regularization

Properties:
    - No variance reduction via log-of-mean
    - Prior-weighted positive term
    - Still uses log for positive samples (to match VPU structure)
    - Anchor assumption: ∃ points where P(y=1|x) = 1
    - NO MixUp regularization
"""

import torch
from torch import nn


class VPUNoMixUpMeanPriorLoss(nn.Module):
    """VPU loss without MixUp, without log-of-mean, with prior weighting.

    Implements:
        L = mean(φ(x)) - π·mean(log(φ_p(x)))
    """

    def __init__(self, prior):
        """Initialize VPU-NoMixUp-Mean-Prior loss.

        Args:
            prior: Class prior π from dataset statistics
        """
        super().__init__()
        self.prior = prior
        self.name = "vpu_nomixup_mean_prior"

    def forward(self, log_phi_all: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute VPU-NoMixUp-Mean-Prior loss.

        Args:
            log_phi_all: log(σ(f(x))) for all samples, shape [N]
            targets: Binary targets (1=positive, 0=unlabeled), shape [N]

        Returns:
            Scalar loss tensor
        """
        # Separate positive samples
        log_phi_p = log_phi_all[targets == 1]

        # Handle edge case: no positive samples
        if len(log_phi_p) == 0:
            return torch.tensor(0.0, device=log_phi_all.device, requires_grad=True)

        # Convert log_phi to phi for the mean computation
        phi_all = torch.exp(log_phi_all)

        # Loss: mean(φ(x)) - π·mean(log(φ_p(x)))
        loss = torch.mean(phi_all) - self.prior * torch.mean(log_phi_p)

        return loss
