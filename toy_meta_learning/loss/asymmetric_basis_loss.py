"""Asymmetric 6-parameter basis loss for PN learning.

Loss structure:
    L_PN = E_P[f_p(p)] + E_N[f_n(1-p)]

where:
    f_p(x) = a1_p + a2_p·x + a3_p·log(x)  (for positives)
    f_n(x) = a1_n + a2_n·x + a3_n·log(x)  (for negatives)

If meta-learning finds f_p ≈ f_n, then symmetry is optimal.
If f_p ≠ f_n, then there's asymmetric structure to exploit!

Optimal for BCE (symmetric):
    a1_p = a1_n = 0
    a2_p = a2_n = 0
    a3_p = a3_n = -1
"""

import torch
import torch.nn as nn


class AsymmetricBasisLoss(nn.Module):
    """Learnable 6-parameter asymmetric basis loss for PN learning.

    Separate basis functions for positives and negatives.

    Parameters:
        Positives: a1_p, a2_p, a3_p
        Negatives: a1_n, a2_n, a3_n
    """

    def __init__(
        self,
        init_mode: str = 'random',
        init_scale: float = 0.01,
    ):
        """Initialize the asymmetric basis loss.

        Args:
            init_mode: Initialization mode ('random', 'bce_equivalent', 'zeros')
            init_scale: Scale for random initialization
        """
        super().__init__()

        self.init_mode = init_mode
        self.init_scale = init_scale

        # Learnable parameters for POSITIVES
        self.a1_p = nn.Parameter(torch.zeros(1))
        self.a2_p = nn.Parameter(torch.zeros(1))
        self.a3_p = nn.Parameter(torch.zeros(1))

        # Learnable parameters for NEGATIVES
        self.a1_n = nn.Parameter(torch.zeros(1))
        self.a2_n = nn.Parameter(torch.zeros(1))
        self.a3_n = nn.Parameter(torch.zeros(1))

        # Initialize
        self._initialize_params()

    def _initialize_params(self):
        """Initialize parameters based on init_mode."""
        with torch.no_grad():
            if self.init_mode == 'random':
                # Random initialization for all 6 parameters
                self.a1_p.data = torch.randn(1) * self.init_scale
                self.a2_p.data = torch.randn(1) * self.init_scale
                self.a3_p.data = torch.randn(1) * self.init_scale

                self.a1_n.data = torch.randn(1) * self.init_scale
                self.a2_n.data = torch.randn(1) * self.init_scale
                self.a3_n.data = torch.randn(1) * self.init_scale

            elif self.init_mode == 'bce_equivalent':
                # BCE: -log(p) for both positives and negatives
                self.a1_p.data = torch.zeros(1)
                self.a2_p.data = torch.zeros(1)
                self.a3_p.data = torch.tensor([-1.0])

                self.a1_n.data = torch.zeros(1)
                self.a2_n.data = torch.zeros(1)
                self.a3_n.data = torch.tensor([-1.0])

            elif self.init_mode == 'zeros':
                # All zeros
                self.a1_p.data = torch.zeros(1)
                self.a2_p.data = torch.zeros(1)
                self.a3_p.data = torch.zeros(1)

                self.a1_n.data = torch.zeros(1)
                self.a2_n.data = torch.zeros(1)
                self.a3_n.data = torch.zeros(1)

            else:
                raise ValueError(f"Unknown init_mode: {self.init_mode}")

    def apply_basis_positive(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positive basis function: f_p(x) = a1_p + a2_p·x + a3_p·log(x).

        Args:
            x: Input probabilities [batch_size] or [batch_size, 1]

        Returns:
            Transformed values [batch_size]
        """
        x = x.view(-1)

        # Clamp x for numerical stability
        eps = 1e-7
        x_safe = torch.clamp(x, min=eps, max=1.0 - eps)

        # f_p(x) = a1_p + a2_p·x + a3_p·log(x)
        result = self.a1_p + self.a2_p * x_safe + self.a3_p * torch.log(x_safe)

        return result

    def apply_basis_negative(self, x: torch.Tensor) -> torch.Tensor:
        """Apply negative basis function: f_n(x) = a1_n + a2_n·x + a3_n·log(x).

        Args:
            x: Input probabilities [batch_size] or [batch_size, 1]

        Returns:
            Transformed values [batch_size]
        """
        x = x.view(-1)

        # Clamp x for numerical stability
        eps = 1e-7
        x_safe = torch.clamp(x, min=eps, max=1.0 - eps)

        # f_n(x) = a1_n + a2_n·x + a3_n·log(x)
        result = self.a1_n + self.a2_n * x_safe + self.a3_n * torch.log(x_safe)

        return result

    def forward(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        mode: str = 'pu',
    ) -> torch.Tensor:
        """Compute PN loss with asymmetric basis functions.

        L_PN = E_P[f_p(p)] + E_N[f_n(1-p)]

        Args:
            outputs: Model logits [batch_size, 1] or [batch_size]
            labels: Labels [batch_size]
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
                return self.apply_basis_positive(p_pos).mean()

            # L_PN = E_P[f_p(p)] + E_U[f_n(1-p)]
            positive_term = self.apply_basis_positive(p_pos).mean()
            unlabeled_term = self.apply_basis_negative(1.0 - p_unlabeled).mean()

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
                return self.apply_basis_negative(1.0 - p_neg).mean()
            if len(p_neg) == 0:
                return self.apply_basis_positive(p_pos).mean()

            # L_PN = E_P[f_p(p)] + E_N[f_n(1-p)]
            positive_term = self.apply_basis_positive(p_pos).mean()
            negative_term = self.apply_basis_negative(1.0 - p_neg).mean()

            return positive_term + negative_term

        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'pu' or 'pn'.")

    def get_parameters(self) -> torch.Tensor:
        """Get all parameters as a tensor [6]."""
        return torch.stack([
            self.a1_p.squeeze(), self.a2_p.squeeze(), self.a3_p.squeeze(),
            self.a1_n.squeeze(), self.a2_n.squeeze(), self.a3_n.squeeze()
        ])

    def get_num_parameters(self) -> int:
        """Get number of learnable parameters."""
        return 6

    def get_symmetry_measure(self) -> float:
        """Measure how symmetric the learned functions are.

        Returns average absolute difference between positive and negative params.
        """
        with torch.no_grad():
            diff_a1 = abs(self.a1_p.item() - self.a1_n.item())
            diff_a2 = abs(self.a2_p.item() - self.a2_n.item())
            diff_a3 = abs(self.a3_p.item() - self.a3_n.item())
            return (diff_a1 + diff_a2 + diff_a3) / 3

    def __repr__(self):
        """String representation."""
        params = self.get_parameters().detach().cpu().numpy()
        return (
            f"AsymmetricBasisLoss(\n"
            f"  Positives: a1_p={params[0]:.4f}, a2_p={params[1]:.4f}, a3_p={params[2]:.4f}\n"
            f"  Negatives: a1_n={params[3]:.4f}, a2_n={params[4]:.4f}, a3_n={params[5]:.4f}\n"
            f"  Symmetry: {self.get_symmetry_measure():.4f}\n"
            f")"
        )
