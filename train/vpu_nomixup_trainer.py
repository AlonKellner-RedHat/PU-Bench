"""vpu_nomixup_trainer.py

VPUNoMixUpTrainer implements the pure VPU method without MixUp regularization.

This variant uses only the variance reduction loss (log-of-mean formulation)
without the MixUp consistency regularization term.
"""

from __future__ import annotations

import torch
from loss.loss_vpu_nomixup import VPUNoMixUpLoss

from .base_trainer import BaseTrainer


class VPUNoMixUpTrainer(BaseTrainer):
    """Pure VPU learning trainer without MixUp regularization."""

    def create_criterion(self):
        """Create pure VPU loss function (no MixUp)."""
        return VPUNoMixUpLoss()

    def train_one_epoch(self, epoch_idx: int):
        """Training loop for one epoch (pure VPU, no MixUp)."""
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

            # Compute pure VPU loss (variance reduction only, no MixUp)
            self.optimizer.zero_grad()
            loss = self.criterion(log_phi_all, targets)
            loss.backward()
            self.optimizer.step()
