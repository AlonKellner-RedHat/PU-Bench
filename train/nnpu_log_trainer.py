"""nnpu_log_trainer.py

NNPULogTrainer uses the nnPU loss with log loss function (λ(x) = -log(x))
instead of the default sigmoid loss.

This variant uses the "log" option from loss_nnpu.py that has been defined
but never used in the codebase.
"""

from __future__ import annotations

import torch
from loss.loss_nnpu import PULoss

from .base_trainer import BaseTrainer


class NNPULogTrainer(BaseTrainer):
    """nnPU learning method with log loss (λ(x) = -log(x))."""

    def create_criterion(self):
        """Create nnPU loss with log loss function."""
        gamma = self.params.get("gamma", 1.0)
        beta = self.params.get("beta", 0.0)
        # KEY CHANGE: loss="log" instead of loss="sigmoid"
        return PULoss(self.prior, loss="log", nnpu=True, gamma=gamma, beta=beta)

    def train_one_epoch(self, epoch_idx: int):
        """Training loop for one epoch (nnPU with log loss)."""
        self.model.train()

        for x, t, _y_true, _idx, _ in self.train_loader:  # type: ignore
            x, t = x.to(self.device), t.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(x).view(-1)

            # nnPU loss with log function uses ±1 labels
            loss = self.criterion(outputs, t)
            loss.backward()
            self.optimizer.step()
