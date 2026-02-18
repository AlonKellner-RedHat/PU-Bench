"""pudra_naive_trainer.py

PUDRaNaiveTrainer implements the pure PUDRa base loss without prior and without regularization.

Key Difference from PUDRa:
    - PUDRa: Uses π * E_P[-log p] + E_U[p] (WITH prior weighting)
    - PUDRa-naive: Uses E_P[-log p + p] + E_U[p] (NO prior weighting)

Key Difference from VPUDRa-naive:
    - VPUDRa-naive: Adds VPU's MixUp consistency for stability
    - PUDRa-naive: No MixUp regularization (pure base loss)

This tests the pure base loss to isolate the effect of regularization.
"""

from __future__ import annotations

import torch
from loss.loss_pudra_naive import PUDRaNaiveLoss

from .base_trainer import BaseTrainer


class PUDRaNaiveTrainer(BaseTrainer):
    """PUDRa-naive trainer with base loss only (no prior, no regularization)."""

    def create_criterion(self):
        """Create PUDRa-naive loss function with configured hyperparameters."""
        epsilon = self.params.get("epsilon", 1e-7)
        # NOTE: No prior, no mix_alpha - pure base loss only
        return PUDRaNaiveLoss(epsilon=epsilon)

    def train_one_epoch(self, epoch_idx: int):
        """Training loop for one epoch (PUDRa-naive with base loss only)."""
        self.model.train()

        for x, t, _y_true, _idx, _ in self.train_loader:  # type: ignore
            x, t = x.to(self.device), t.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(x).view(-1)

            # PUDRa-naive loss uses ±1 labels (1=positive, -1=unlabeled)
            # Base loss only: E_P[-log p + p] + E_U[p]
            loss = self.criterion(outputs, t)
            loss.backward()
            self.optimizer.step()
