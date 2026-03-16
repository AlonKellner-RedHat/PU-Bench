"""Asymmetric 6-parameter basis loss specifically for PU learning.

Loss structure:
    L_PU = E_P[f_p(p)] + E_U[f_u(1-p)]

where:
    f_p(x) = a1_p + a2_p·x + a3_p·log(x)  (for labeled POSITIVES)
    f_u(x) = a1_u + a2_u·x + a3_u·log(x)  (for UNLABELED)

This is more natural than symmetric because:
- Labeled positives are CLEAN labels
- Unlabeled is a MIXTURE (contains hidden positives + negatives)
- These groups have fundamentally different characteristics!

If meta-learning finds f_p ≈ f_u, then treating them the same is optimal.
If f_p ≠ f_u, then there's important asymmetric structure!

Optimal for symmetric BCE:
    a1_p = a1_u = 0
    a2_p = a2_u = 0
    a3_p = a3_u = -1
"""

import torch
import torch.nn as nn


class AsymmetricPULoss(nn.Module):
    """Learnable 6-parameter asymmetric basis loss for PU learning.

    Separate basis functions for labeled positives and unlabeled.

    Parameters:
        Labeled Positives: a1_p, a2_p, a3_p
        Unlabeled: a1_u, a2_u, a3_u
    """

    def __init__(
        self,
        init_mode: str = 'random',
        init_scale: float = 0.01,
    ):
        """Initialize the asymmetric PU loss.

        Args:
            init_mode: Initialization mode ('random', 'bce_equivalent', 'zeros')
            init_scale: Scale for random initialization
        """
        super().__init__()

        self.init_mode = init_mode
        self.init_scale = init_scale

        # Learnable parameters for LABELED POSITIVES
        self.a1_p = nn.Parameter(torch.zeros(1))
        self.a2_p = nn.Parameter(torch.zeros(1))
        self.a3_p = nn.Parameter(torch.zeros(1))

        # Learnable parameters for UNLABELED
        self.a1_u = nn.Parameter(torch.zeros(1))
        self.a2_u = nn.Parameter(torch.zeros(1))
        self.a3_u = nn.Parameter(torch.zeros(1))

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

                self.a1_u.data = torch.randn(1) * self.init_scale
                self.a2_u.data = torch.randn(1) * self.init_scale
                self.a3_u.data = torch.randn(1) * self.init_scale

            elif self.init_mode == 'bce_equivalent':
                # BCE: -log(p) for both
                self.a1_p.data = torch.zeros(1)
                self.a2_p.data = torch.zeros(1)
                self.a3_p.data = torch.tensor([-1.0])

                self.a1_u.data = torch.zeros(1)
                self.a2_u.data = torch.zeros(1)
                self.a3_u.data = torch.tensor([-1.0])

            elif self.init_mode == 'zeros':
                # All zeros
                self.a1_p.data = torch.zeros(1)
                self.a2_p.data = torch.zeros(1)
                self.a3_p.data = torch.zeros(1)

                self.a1_u.data = torch.zeros(1)
                self.a2_u.data = torch.zeros(1)
                self.a3_u.data = torch.zeros(1)

            else:
                raise ValueError(f"Unknown init_mode: {self.init_mode}")

    def apply_basis_positive(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positive basis: f_p(x) = a1_p + a2_p·x + a3_p·log(x).

        Args:
            x: Input probabilities

        Returns:
            Transformed values
        """
        x = x.view(-1)

        # Clamp for numerical stability
        eps = 1e-7
        x_safe = torch.clamp(x, min=eps, max=1.0 - eps)

        result = self.a1_p + self.a2_p * x_safe + self.a3_p * torch.log(x_safe)
        return result

    def apply_basis_unlabeled(self, x: torch.Tensor) -> torch.Tensor:
        """Apply unlabeled basis: f_u(x) = a1_u + a2_u·x + a3_u·log(x).

        Args:
            x: Input probabilities

        Returns:
            Transformed values
        """
        x = x.view(-1)

        # Clamp for numerical stability
        eps = 1e-7
        x_safe = torch.clamp(x, min=eps, max=1.0 - eps)

        result = self.a1_u + self.a2_u * x_safe + self.a3_u * torch.log(x_safe)
        return result

    def forward(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        mode: str = 'pu',
    ) -> torch.Tensor:
        """Compute PU loss with asymmetric basis functions.

        L_PU = E_P[f_p(p)] + E_U[f_u(1-p)]

        Args:
            outputs: Model logits
            labels: PU labels (1 for labeled positive, -1 for unlabeled)
            mode: Must be 'pu'

        Returns:
            Scalar loss value
        """
        if mode != 'pu':
            raise ValueError(f"AsymmetricPULoss only supports mode='pu', got '{mode}'")

        # Convert to probabilities
        p = torch.sigmoid(outputs.view(-1))
        labels = labels.view(-1)

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

        # L_PU = E_P[f_p(p)] + E_U[f_u(1-p)]
        positive_term = self.apply_basis_positive(p_pos).mean()
        unlabeled_term = self.apply_basis_unlabeled(1.0 - p_unlabeled).mean()

        return positive_term + unlabeled_term

    def get_parameters(self) -> torch.Tensor:
        """Get all parameters as a tensor [6]."""
        return torch.stack([
            self.a1_p.squeeze(), self.a2_p.squeeze(), self.a3_p.squeeze(),
            self.a1_u.squeeze(), self.a2_u.squeeze(), self.a3_u.squeeze()
        ])

    def get_num_parameters(self) -> int:
        """Get number of learnable parameters."""
        return 6

    def get_symmetry_measure(self) -> float:
        """Measure how symmetric the learned functions are.

        Returns average absolute difference between positive and unlabeled params.
        """
        with torch.no_grad():
            diff_a1 = abs(self.a1_p.item() - self.a1_u.item())
            diff_a2 = abs(self.a2_p.item() - self.a2_u.item())
            diff_a3 = abs(self.a3_p.item() - self.a3_u.item())
            return (diff_a1 + diff_a2 + diff_a3) / 3

    def __repr__(self):
        """String representation."""
        params = self.get_parameters().detach().cpu().numpy()
        return (
            f"AsymmetricPULoss(\n"
            f"  Labeled Pos: a1_p={params[0]:.4f}, a2_p={params[1]:.4f}, a3_p={params[2]:.4f}\n"
            f"  Unlabeled:   a1_u={params[3]:.4f}, a2_u={params[4]:.4f}, a3_u={params[5]:.4f}\n"
            f"  Symmetry: {self.get_symmetry_measure():.4f}\n"
            f")"
        )
