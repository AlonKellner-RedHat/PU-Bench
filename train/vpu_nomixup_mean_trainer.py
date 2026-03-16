"""vpu_nomixup_mean_trainer.py

VPUNoMixUpMeanTrainer implements VPU without MixUp and without log-of-mean.

This variant tests whether the log-of-mean formulation contributes to VPU's
performance by using a simpler mean(φ(x)) instead of log(mean(φ(x))).
"""

from __future__ import annotations

import torch
from loss.loss_vpu_nomixup_mean import VPUNoMixUpMeanLoss

from .base_trainer import BaseTrainer


class VPUNoMixUpMeanTrainer(BaseTrainer):
    """VPU trainer without MixUp and without log-of-mean variance reduction."""

    def create_criterion(self):
        """Create VPU-NoMixUp-Mean loss function."""
        return VPUNoMixUpMeanLoss()

    def train_one_epoch(self, epoch_idx: int):
        """Training loop for one epoch (VPU with mean instead of log-of-mean)."""
        self.model.train()

        for x, t, _y_true, _idx, _ in self.train_loader:  # type: ignore
            x, t = x.to(self.device), t.to(self.device)

            # Create positive mask (1=positive, -1=unlabeled -> convert to boolean)
            p_mask = t == 1

            # Skip batches with no positive samples
            if p_mask.sum() == 0:
                continue

            # Forward pass: log(σ(f(x)))
            log_phi_all = torch.sigmoid(self.model(x)).log()

            # Convert PU labels to binary targets (1=positive, 0=unlabeled)
            # VPU expects 1/0, not 1/-1
            targets = p_mask.float()

            # Compute VPU-NoMixUp-Mean loss
            self.optimizer.zero_grad()
            loss = self.criterion(log_phi_all, targets)
            loss.backward()
            self.optimizer.step()
