"""pudrasb_trainer.py

PUDRaSBTrainer implements PUDRaSB (PU Density Ratio with Selection Bias).

PUDRaSB combines:
- PUDRa's strictly convex Point Process / Generalized KL loss
- nnPUSB's scalar propensity weighting for selection bias handling

This provides PUDRa's theoretical elegance with readiness for SAR scenarios.
"""

from __future__ import annotations

import torch
from loss.loss_pudrasb import PUDRaSBLoss

from .base_trainer import BaseTrainer


class PUDRaSBTrainer(BaseTrainer):
    """PUDRaSB (PU Density Ratio with Selection Bias) learning method trainer."""

    def create_criterion(self):
        """Create PUDRaSB loss function with configured hyperparameters."""
        activation = self.params.get("activation", "sigmoid")
        weight = self.params.get("weight", 1.0)
        epsilon = self.params.get("epsilon", 1e-7)
        return PUDRaSBLoss(
            self.prior, activation=activation, weight=weight, epsilon=epsilon
        )

    def train_one_epoch(self, epoch_idx: int):
        """Training loop for one epoch (PUDRaSB)."""
        self.model.train()

        for x, t, _y_true, _idx, _ in self.train_loader:  # type: ignore
            x, t = x.to(self.device), t.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(x).view(-1)

            # PUDRaSB loss uses ±1 labels (1=positive, -1=unlabeled)
            loss = self.criterion(outputs, t)
            loss.backward()
            self.optimizer.step()
