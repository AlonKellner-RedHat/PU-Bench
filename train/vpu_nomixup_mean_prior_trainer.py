"""VPU-NoMixUp-Mean-Prior Trainer

Trainer for VPU-NoMixUp-Mean-Prior loss.
This variant removes MixUp, uses mean instead of log-of-mean, and weights the positive term by prior π.
"""

from __future__ import annotations

import torch
from loss.loss_vpu_nomixup_mean_prior import VPUNoMixUpMeanPriorLoss

from .base_trainer import BaseTrainer


class VPUNoMixUpMeanPriorTrainer(BaseTrainer):
    """VPU-NoMixUp-Mean-Prior learning trainer (no MixUp, with prior weighting)"""

    def create_criterion(self):
        return VPUNoMixUpMeanPriorLoss(self.prior)

    def train_one_epoch(self, epoch_idx: int):
        """Training loop for one epoch (VPU-NoMixUp-Mean-Prior)."""
        self.model.train()

        for x, t, _y_true, _idx, _ in self.train_loader:  # type: ignore
            x, t = x.to(self.device), t.to(self.device)

            # log σ(f(x))
            log_phi_all = torch.sigmoid(self.model(x)).log()

            # Convert PU labels {1, -1} to {1, 0} for targets
            targets = (t == 1).float()

            self.optimizer.zero_grad()
            loss = self.criterion(log_phi_all, targets)
            loss.backward()
            self.optimizer.step()
