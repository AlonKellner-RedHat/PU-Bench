"""pn_naive_trainer.py

PN Naive: Treats all unlabeled examples as negatives.

This is a common naive baseline that shows what happens when you ignore
the PU learning problem and just apply standard supervised learning to PU data.

Expected behavior:
- Creates biased training (true positives in unlabeled set labeled as negative)
- Performance degrades with smaller label frequency c (more positives mislabeled)
- Demonstrates value of proper PU learning methods
"""

import torch
from tqdm import tqdm

from .base_trainer import BaseTrainer


class PNNaiveTrainer(BaseTrainer):
    """Naive PN baseline: treats unlabeled as negative.

    This trainer uses PU labels instead of true labels, converting:
    - Labeled positive (1) → 1.0
    - Unlabeled (-1) → 0.0 (INCORRECTLY treating positives in U as negative)
    """

    def create_criterion(self):
        """
        Creates the loss function for naive PN classification.
        BCEWithLogitsLoss is suitable for binary classification and is numerically
        more stable than using a Sigmoid layer followed by BCELoss.
        """
        return torch.nn.BCEWithLogitsLoss()

    def train_one_epoch(self, epoch_idx: int):
        """
        Executes training for one epoch using PU labels (naive approach).

        This method uses pu_labels instead of true_labels, incorrectly treating
        all unlabeled examples as negatives.

        Args:
            epoch_idx (int): The current epoch number.
        """
        self.model.train()
        total_loss = 0.0

        num_epochs = self.params.get("num_epochs", "N/A")
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch_idx}/{num_epochs} [PN Naive]",
            leave=False,
        )

        for batch in progress_bar:
            # PUDataset yields: features, pu_labels, true_labels, idx, pseudo
            # For PN Naive, we use PU labels instead of true labels (the naive mistake!)
            features, pu_labels, _, _, _ = batch

            features = features.to(self.device)

            # Convert PU labels to naive binary labels:
            # 1 (labeled positive) → 1.0
            # -1 (unlabeled) → 0.0 (incorrectly treating positives in U as negative)
            naive_labels = (pu_labels == 1).float().to(self.device)

            # Zero gradients, perform a forward pass, and calculate loss
            self.optimizer.zero_grad()
            outputs = self.model(features).squeeze()
            loss = self.criterion(outputs, naive_labels)

            # Perform a backward pass and update weights
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(self.train_loader)
        if avg_loss > 0:
            log_msg = f"Epoch {epoch_idx} - Average Training Loss: {avg_loss:.4f}"

            # Log to both consoles
            self.console.log(log_msg)
            if self.file_console:
                self.file_console.log(log_msg)
