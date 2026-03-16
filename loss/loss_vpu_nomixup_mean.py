"""VPU-NoMixUp-Mean Loss (VPU without log-of-mean variance reduction)

This module implements a variant of VPU-NoMixUp that replaces log(mean) with mean.

Mathematical Foundation:
    L_VPU_NoMixUp_Mean = E_all[φ(x)] - E_P[log φ(x)]
                        = mean(φ(x)) - mean(log φ_p(x))

    where:
    - φ(x) = σ(f(x)) is the sigmoid-activated model output
    - E_all is expectation over all samples (labeled + unlabeled)
    - E_P is expectation over positive samples only

Key Differences from VPU-NoMixUp:
    - VPU-NoMixUp: log(mean(φ(x))) - mean(log(φ_p(x)))
    - VPU-NoMixUp-Mean: mean(φ(x)) - mean(log(φ_p(x)))
    - Removes the log-of-mean formulation for variance reduction
    - Tests whether the log transformation contributes to performance

Properties:
    - No variance reduction via log-of-mean
    - Still uses log for positive samples (to match VPU structure)
    - Anchor assumption: ∃ points where P(y=1|x) = 1
    - NO MixUp regularization
"""

import torch
from torch import nn


class VPUNoMixUpMeanLoss(nn.Module):
    """VPU loss without MixUp and without log-of-mean variance reduction.

    Implements:
        L = mean(φ(x)) - mean(log(φ_p(x)))
    """

    def __init__(self):
        super().__init__()
        self.name = "vpu_nomixup_mean"

    def forward(self, log_phi_all: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute VPU-NoMixUp-Mean loss.

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

        # Loss: mean(φ(x)) - mean(log(φ_p(x)))
        loss = torch.mean(phi_all) - torch.mean(log_phi_p)

        return loss
