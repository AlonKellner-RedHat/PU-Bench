import torch
from torch import nn
import torch.nn.functional as F


class OracleBCELoss(nn.Module):
    """Oracle Binary Cross-Entropy loss for fully supervised learning.

    This is a baseline that uses true labels (both positives and negatives).
    It's called "oracle" because it has access to ground truth labels that
    wouldn't be available in a true PU learning setting.
    """

    def __init__(self):
        super(OracleBCELoss, self).__init__()
        self.name = "oracle_bce"

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: raw model outputs (before sigmoid)
            targets: true binary labels (0 or 1)
        """
        return F.binary_cross_entropy_with_logits(logits, targets)
