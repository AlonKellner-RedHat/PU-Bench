"""Simple 3-parameter basis loss for PN learning.

Loss structure:
    L_PN = E_P[f(p)] + E_U[f(1-p)]

where f(x) = a₁ + a₂·x + a₃·log(x)

Optimal for BCE:
    a₁ = 0, a₂ = 0, a₃ = -1  →  f(x) = -log(x)
"""

import torch
import torch.nn as nn


class SimpleBasisLoss(nn.Module):
    """Learnable 3-parameter basis loss for PN learning.

    Parameters:
        a1: Constant term
        a2: Linear coefficient
        a3: Logarithmic coefficient

    Optimal (BCE-equivalent): a1=0, a2=0, a3=-1
    """

    def __init__(
        self,
        init_mode: str = 'random',
        init_scale: float = 0.01,
    ):
        """Initialize the basis loss.

        Args:
            init_mode: Initialization mode ('random', 'bce_equivalent', 'zeros')
            init_scale: Scale for random initialization
        """
        super().__init__()

        self.init_mode = init_mode
        self.init_scale = init_scale

        # Learnable parameters
        self.a1 = nn.Parameter(torch.zeros(1))
        self.a2 = nn.Parameter(torch.zeros(1))
        self.a3 = nn.Parameter(torch.zeros(1))

        # Initialize
        self._initialize_params()

    def _initialize_params(self):
        """Initialize parameters based on init_mode."""
        with torch.no_grad():
            if self.init_mode == 'random':
                self.a1.data = torch.randn(1) * self.init_scale
                self.a2.data = torch.randn(1) * self.init_scale
                self.a3.data = torch.randn(1) * self.init_scale
            elif self.init_mode == 'bce_equivalent':
                # BCE: -log(p) for positives, -log(1-p) for negatives
                self.a1.data = torch.zeros(1)
                self.a2.data = torch.zeros(1)
                self.a3.data = torch.tensor([-1.0])
            elif self.init_mode == 'zeros':
                # All zeros
                self.a1.data = torch.zeros(1)
                self.a2.data = torch.zeros(1)
                self.a3.data = torch.zeros(1)
            else:
                raise ValueError(f"Unknown init_mode: {self.init_mode}")

    def apply_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Apply basis function: f(x) = a₁ + a₂·x + a₃·log(x).

        Args:
            x: Input probabilities [batch_size] or [batch_size, 1]

        Returns:
            Transformed values [batch_size]
        """
        x = x.view(-1)  # Flatten to 1D

        # Clamp x for numerical stability
        eps = 1e-7
        x_safe = torch.clamp(x, min=eps, max=1.0 - eps)

        # f(x) = a₁ + a₂·x + a₃·log(x)
        result = self.a1 + self.a2 * x_safe + self.a3 * torch.log(x_safe)

        return result

    def forward(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        mode: str = 'pu',
    ) -> torch.Tensor:
        """Compute PN loss.

        PU mode (mode='pu'):
            L_PN = E_P[f(p)] + E_U[f(1-p)]
            where P = labeled positives (label=1), U = unlabeled (label=-1)

        PN mode (mode='pn'):
            L_PN = E_P[f(p)] + E_N[f(1-p)]
            where P = labeled positives (label=1), N = labeled negatives (label=0)
            Unlabeled samples (label=-1) are ignored in PN mode

        Args:
            outputs: Model logits [batch_size, 1] or [batch_size]
            labels: Labels [batch_size]
                - PU mode: 1 for labeled positive, -1 for unlabeled
                - PN mode: 1 for labeled positive, 0 for labeled negative, -1 for unlabeled (ignored)
            mode: 'pu' or 'pn'

        Returns:
            Scalar loss value
        """
        # Convert to probabilities
        p = torch.sigmoid(outputs.view(-1))
        labels = labels.view(-1)

        if mode == 'pu':
            # PU mode: labeled positives and unlabeled
            pos_mask = labels == 1
            unlabeled_mask = labels == -1

            p_pos = p[pos_mask]
            p_unlabeled = p[unlabeled_mask]

            # Handle edge cases
            if len(p_pos) == 0:
                return torch.tensor(0.0, device=p.device, requires_grad=True)
            if len(p_unlabeled) == 0:
                return self.apply_basis(p_pos).mean()

            # L_PN = E_P[f(p)] + E_U[f(1-p)]
            positive_term = self.apply_basis(p_pos).mean()
            unlabeled_term = self.apply_basis(1.0 - p_unlabeled).mean()

            return positive_term + unlabeled_term

        elif mode == 'pn':
            # PN mode: labeled positives and labeled negatives
            pos_mask = labels == 1
            neg_mask = labels == 0

            p_pos = p[pos_mask]
            p_neg = p[neg_mask]

            # Handle edge cases
            if len(p_pos) == 0 and len(p_neg) == 0:
                return torch.tensor(0.0, device=p.device, requires_grad=True)
            if len(p_pos) == 0:
                return self.apply_basis(1.0 - p_neg).mean()
            if len(p_neg) == 0:
                return self.apply_basis(p_pos).mean()

            # L_PN = E_P[f(p)] + E_N[f(1-p)]
            positive_term = self.apply_basis(p_pos).mean()
            negative_term = self.apply_basis(1.0 - p_neg).mean()

            return positive_term + negative_term

        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'pu' or 'pn'.")

    def get_parameters(self) -> torch.Tensor:
        """Get all parameters as a tensor [3]."""
        return torch.stack([self.a1.squeeze(), self.a2.squeeze(), self.a3.squeeze()])

    def get_num_parameters(self) -> int:
        """Get number of learnable parameters."""
        return 3

    def __repr__(self):
        """String representation."""
        params = self.get_parameters().detach().cpu().numpy()
        return (
            f"SimpleBasisLoss(a1={params[0]:.4f}, a2={params[1]:.4f}, "
            f"a3={params[2]:.4f})"
        )
