"""VPU-NoMixUp Loss (Pure VPU variance reduction without MixUp regularization)

This module implements the pure VPU loss function without MixUp regularization.

Mathematical Foundation:
    L_VPU_NoMixUp = log(E_all[φ(x)]) - E_P[log φ(x)]
                  = logsumexp(log φ(x)) - log(N) - mean(log φ_p(x))

    where:
    - φ(x) = σ(f(x)) is the sigmoid-activated model output
    - E_all is expectation over all samples (labeled + unlabeled)
    - E_P is expectation over positive samples only
    - logsumexp provides numerical stability for log(mean(exp(...)))

Key Properties:
    - Variance reduction via log-of-mean formulation
    - Anchor assumption: ∃ points where P(y=1|x) = 1
    - NO MixUp regularization (pure VPU loss only)
    - Avoids need for explicit prior estimation

Comparison to full VPU:
    - VPU: var_loss + lam * mixup_regularization
    - VPU-NoMixUp: var_loss only (no MixUp term)
    - Simpler, faster training (no MixUp data augmentation)
    - Tests whether MixUp contributes to VPU's performance
"""

import math
import torch
from torch import nn


class VPUNoMixUpLoss(nn.Module):
    """Pure VPU loss without MixUp regularization.

    Implements only the variance reduction term:
        L = log(E_all[φ(x)]) - E_P[log φ(x)]
    """

    def __init__(self):
        super().__init__()
        self.name = "vpu_nomixup"

    def forward(self, log_phi_all: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute pure VPU loss without MixUp.

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

        # Variance reduction loss: log(mean(φ(x))) - mean(log(φ_p(x)))
        # = logsumexp(log φ(x)) - log(N) - mean(log φ_p(x))
        var_loss = (
            torch.logsumexp(log_phi_all, dim=0)
            - math.log(len(log_phi_all))
            - torch.mean(log_phi_p)
        )

        return var_loss
