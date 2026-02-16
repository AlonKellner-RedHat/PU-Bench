"""pudra_trainer.py

PUDRATrainer implements the PUDRa (Positive-Unlabeled Density Ratio) method.

PUDRa uses a strictly convex Point Process / Generalized KL loss that naturally
converges without requiring non-negative clipping constraints.
"""

from __future__ import annotations

import torch
from loss.loss_pudra import PUDRALoss

from .base_trainer import BaseTrainer


class PUDRATrainer(BaseTrainer):
    """PUDRa (Positive-Unlabeled Density Ratio) learning method trainer."""

    def create_criterion(self):
        """Create PUDRa loss function with configured hyperparameters."""
        activation = self.params.get("activation", "sigmoid")
        epsilon = self.params.get("epsilon", 1e-7)
        return PUDRALoss(self.prior, activation=activation, epsilon=epsilon)

    def train_one_epoch(self, epoch_idx: int):
        """Training loop for one epoch (PUDRa)."""
        self.model.train()

        for x, t, _y_true, _idx, _ in self.train_loader:  # type: ignore
            x, t = x.to(self.device), t.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(x).view(-1)

            # PUDRa loss uses ±1 labels (1=positive, -1=unlabeled)
            loss = self.criterion(outputs, t)
            loss.backward()
            self.optimizer.step()
