"""pudra_unified_trainer.py

Trainer for PUDRa-Unified loss (elementwise averaging over all samples).
"""

from __future__ import annotations

import torch
from loss.loss_pudra_unified import PUDRaUnifiedLoss

from .base_trainer import BaseTrainer


class PUDRaUnifiedTrainer(BaseTrainer):
    """Trainer for PUDRa-Unified loss (no prior, unified averaging)."""

    def create_criterion(self):
        """Create PUDRa-unified loss function."""
        epsilon = self.params.get("epsilon", 1e-7)
        return PUDRaUnifiedLoss(epsilon=epsilon)

    def train_one_epoch(self, epoch_idx: int):
        """Training loop for one epoch (PUDRa-unified with unified averaging)."""
        self.model.train()

        for x, t, _y_true, _idx, _ in self.train_loader:  # type: ignore
            x, t = x.to(self.device), t.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(x).view(-1)

            # PUDRa-unified loss uses ±1 labels (1=positive, -1=unlabeled)
            # Unified averaging: E_all[loss(x, t)]
            loss = self.criterion(outputs, t)
            loss.backward()
            self.optimizer.step()
