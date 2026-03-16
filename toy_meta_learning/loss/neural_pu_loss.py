#!/usr/bin/env python3
"""Neural network-based learnable PU loss.

This loss uses a learnable neural network to process batch-level statistics
and produce a scalar loss value. Designed to be more stable and expressive
than basis function approaches.

Architecture:
    Input: 13 feature vectors (masks, probabilities, logs) of size B
    → Stack to [B, 13]
    → Linear(13, hidden_dim) → [B, hidden_dim] (T)
    → Split T → X1, X2 (each [B, hidden_dim/2])
    → Multiply: P = X1 * X2 (element-wise)
    → Sum over batch → [hidden_dim/2] (A)
    → Split A → A1, A2 (each hidden_dim/4)
    → Learned params W1, W2 (each hidden_dim/4)
    → O1 = W1*A1, O2 = W2*log(|A2|)
    → Loss = O1.sum() + O2.sum()

Parameters (hidden_dim=128):
    - Linear layer: 13 × 128 + 128 = 1,792
    - W1: 32, W2: 32
    - Total: 1,856 parameters

Regularization:
    L1 norm on weights (not intermediate activations)
    Encourages sparse, interpretable solutions
"""

import torch
import torch.nn as nn
import torch.nn.init as init


class NeuralPULoss(nn.Module):
    """Neural network-based learnable PU loss.

    Parameters:
        hidden_dim: Hidden layer dimension (must be divisible by 4)
        eps: Numerical stability epsilon
        l1_lambda: L1 regularization strength
        init_mode: Weight initialization strategy
        init_scale: Scaling factor for initialization
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        eps: float = 1e-7,
        l1_lambda: float = 0.0,
        l05_lambda: float = 0.0,
        init_mode: str = 'xavier_uniform',
        init_scale: float = 1.0,
        max_weight_norm: float = 10.0,
    ):
        super().__init__()

        # Validate hidden_dim
        if hidden_dim % 4 != 0:
            raise ValueError(
                f"hidden_dim must be divisible by 4 (for splitting), got {hidden_dim}"
            )
        if hidden_dim < 4:
            raise ValueError(f"hidden_dim must be at least 4, got {hidden_dim}")

        self.hidden_dim = hidden_dim
        self.eps = eps
        self.l1_lambda = l1_lambda
        self.l05_lambda = l05_lambda
        self.init_mode = init_mode
        self.init_scale = init_scale
        self.max_weight_norm = max_weight_norm

        # Linear layer: 13 input features → hidden_dim
        self.linear = nn.Linear(13, hidden_dim)

        # Learned weights for final aggregation (hidden_dim/4 each)
        # W1 weights the linear terms (A1), W2 weights the log terms (A2)
        quarter = hidden_dim // 4
        self.W1 = nn.Parameter(torch.ones(quarter))  # Linear term weights
        self.W2 = nn.Parameter(torch.ones(quarter))  # Log term weights

        # Initialize weights
        self._initialize_weights()

        # Storage for intermediate activations (for regularization)
        self._last_T = None
        self._last_P = None
        self._last_A = None

    def _initialize_weights(self):
        """Initialize linear layer weights based on init_mode."""
        with torch.no_grad():
            if self.init_mode == 'xavier_uniform':
                init.xavier_uniform_(self.linear.weight, gain=self.init_scale)
                init.zeros_(self.linear.bias)

            elif self.init_mode == 'kaiming_normal':
                init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='linear')
                self.linear.weight.data *= self.init_scale
                init.zeros_(self.linear.bias)

            elif self.init_mode == 'bce_equivalent':
                # Initialize to approximate BCE: -E_p[log(p)] - E_u[log(1-p)]
                self.linear.weight.zero_()
                self.linear.bias.zero_()

                # Index 8 = log_V_p, Index 12 = log_1mV_u  (updated for 13 features)
                # Set negative weights to approximate BCE
                self.linear.weight[:, 8] = -1.0 / self.hidden_dim
                self.linear.weight[:, 12] = -1.0 / self.hidden_dim

            elif self.init_mode == 'random_normal':
                self.linear.weight.data.normal_(0, self.init_scale)
                self.linear.bias.data.zero_()

            else:
                raise ValueError(f"Unknown init_mode: {self.init_mode}")

    def safe_log(self, x: torch.Tensor) -> torch.Tensor:
        """Safe logarithm that handles zeros and overflow gracefully.

        Returns log(clamp(x, eps, 1e6)) to prevent -inf and +inf gradients.
        Differentiable and numerically stable.

        Args:
            x: Input tensor

        Returns:
            Log of clamped input
        """
        return torch.log(torch.clamp(x, min=self.eps, max=1e6))

    def forward(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        mode: str = 'pu'
    ) -> torch.Tensor:
        """Compute neural PU loss.

        Args:
            outputs: Model logits [batch_size] or [batch_size, 1]
            labels: PU labels [batch_size] where 1=positive, -1=unlabeled
            mode: 'pu' for PU learning (only mode supported)

        Returns:
            Scalar loss tensor with gradients
        """
        if mode != 'pu':
            raise ValueError(f"NeuralPULoss only supports mode='pu', got '{mode}'")

        # Step 0: Clip weights to prevent explosion
        if self.max_weight_norm > 0:
            with torch.no_grad():
                weight_norm = self.linear.weight.norm()
                if weight_norm > self.max_weight_norm:
                    self.linear.weight *= (self.max_weight_norm / weight_norm)

        # Step 1: Convert logits to probabilities
        p = torch.sigmoid(outputs.view(-1))  # [B]
        labels = labels.view(-1)  # [B]

        # Step 2: Create masks
        M_p = (labels == 1).float()  # [B]
        M_u = (labels == -1).float()  # [B]

        # Step 3: Edge case - no positives
        if M_p.sum() == 0:
            return torch.tensor(0.0, device=p.device, requires_grad=True)

        # Step 4: Edge case - no unlabeled (pad with dummy zero)
        if M_u.sum() == 0:
            p = torch.cat([p, torch.zeros(1, device=p.device)])
            M_p = torch.cat([M_p, torch.zeros(1, device=p.device)])
            M_u = torch.cat([M_u, torch.ones(1, device=p.device)])

        # Step 4.5: Compute batch size and normalized masks
        B = float(len(M_p))
        M_p_norm = M_p / B  # [B] - fraction of positives
        M_u_norm = M_u / B  # [B] - fraction of unlabeled

        # Step 5: Compute derived vectors
        V = p
        V_p = V * M_p
        V_u = V * M_u

        # Step 6: Compute log vectors
        log_V = self.safe_log(V)
        log_V_p = self.safe_log(V_p)
        log_V_u = self.safe_log(V_u)
        log_1mV = self.safe_log(1 - V)
        log_1mV_p = self.safe_log(1 - V_p)
        log_1mV_u = self.safe_log(1 - V_u)

        # Step 7: Stack into [B, 13] tensor
        input_tensor = torch.stack([
            M_p, M_u, M_p_norm, M_u_norm, V, V_p, V_u,
            log_V, log_V_p, log_V_u,
            log_1mV, log_1mV_p, log_1mV_u
        ], dim=1)  # [B, 13]

        # Step 8: Linear transformation
        T = self.linear(input_tensor)  # [B, hidden_dim]
        self._last_T = T  # Store for regularization

        # Step 9: Split and multiply
        half = self.hidden_dim // 2
        X1 = T[:, :half]  # [B, hidden_dim/2]
        X2 = T[:, half:]  # [B, hidden_dim/2]
        P = X1 * X2  # [B, hidden_dim/2] - element-wise multiplication
        self._last_P = P  # Store for regularization

        # Step 10: Aggregate over batch
        A = P.sum(dim=0)  # [hidden_dim/2]
        self._last_A = A  # Store for regularization

        # Step 11: Split A and compute final loss with learned weights
        quarter = half // 2
        A1 = A[:quarter]  # [hidden_dim/4]
        A2 = A[quarter:]  # [hidden_dim/4]

        # Apply learned weights to each term
        O1 = self.W1 * A1  # [hidden_dim/4] - weighted linear terms
        O2 = self.W2 * self.safe_log(torch.abs(A2))  # [hidden_dim/4] - weighted log terms

        loss = O1.sum() + O2.sum()

        # Step 12: Add regularization
        if self.l1_lambda > 0 or self.l05_lambda > 0:
            loss = loss + self.compute_regularization()

        return loss

    def compute_regularization(self) -> torch.Tensor:
        """Compute L1 and L0.5 regularization.

        L1 encourages sparsity: sum(|w|)
        L0.5 encourages stronger sparsity: sum((|w| + eps)^0.5)

        Note: L0.5 uses eps to prevent gradient explosion at w=0,
        since d/dw(|w|^0.5) = 0.5 * |w|^(-0.5) → ∞ as w → 0

        Applied to:
        - All linear layer weights and biases ONLY
        - NOT to intermediate activations (can cause numerical instability with large networks)

        Returns:
            Combined regularization penalty
        """
        if self.l1_lambda == 0.0 and self.l05_lambda == 0.0:
            return torch.tensor(0.0, device=self.linear.weight.device)

        reg = torch.tensor(0.0, device=self.linear.weight.device)

        # Regularize linear layer weights and biases
        for param in self.linear.parameters():
            if self.l1_lambda > 0:
                reg = reg + self.l1_lambda * torch.sum(torch.abs(param))
            if self.l05_lambda > 0:
                # Add eps to prevent gradient explosion near zero
                reg = reg + self.l05_lambda * torch.sum((torch.abs(param) + self.eps) ** 0.5)

        # Regularize W1 and W2 (final aggregation weights)
        if self.l1_lambda > 0:
            reg = reg + self.l1_lambda * (torch.sum(torch.abs(self.W1)) + torch.sum(torch.abs(self.W2)))
        if self.l05_lambda > 0:
            reg = reg + self.l05_lambda * (
                torch.sum((torch.abs(self.W1) + self.eps) ** 0.5) +
                torch.sum((torch.abs(self.W2) + self.eps) ** 0.5)
            )

        # DO NOT regularize intermediate activations (disabled for numerical stability)
        # With larger networks (hidden_dim=128), regularizing ~14k activation values
        # can cause NaN gradients during backpropagation

        return reg

    def get_num_parameters(self) -> int:
        """Get total number of learnable parameters.

        Returns:
            Parameter count
        """
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        """String representation showing configuration."""
        num_params = self.get_num_parameters()
        return (
            f"NeuralPULoss(\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  num_parameters={num_params},\n"
            f"  eps={self.eps},\n"
            f"  l1_lambda={self.l1_lambda},\n"
            f"  l05_lambda={self.l05_lambda},\n"
            f"  init_mode='{self.init_mode}',\n"
            f"  init_scale={self.init_scale},\n"
            f"  max_weight_norm={self.max_weight_norm}\n"
            f")"
        )
