"""oracle_bce_trainer.py

OracleBCETrainer implements fully supervised learning with Binary Cross-Entropy.
This serves as an upper-bound baseline since it has access to true labels.
"""

from __future__ import annotations

import torch
from loss.loss_oracle_bce import OracleBCELoss

from .base_trainer import BaseTrainer


class OracleBCETrainer(BaseTrainer):
    """Oracle trainer using true labels and standard BCE loss"""

    def create_criterion(self):
        return OracleBCELoss()

    def train_one_epoch(self, epoch_idx: int):
        self.model.train()

        for x, _t, y_true, _idx, _ in self.train_loader:  # type: ignore
            x, y_true = x.to(self.device), y_true.to(self.device)

            # Use true labels (both positives and negatives)
            # Convert to float for BCE
            targets = y_true.float()

            # Get raw logits
            logits = self.model(x)

            self.optimizer.zero_grad()
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()
