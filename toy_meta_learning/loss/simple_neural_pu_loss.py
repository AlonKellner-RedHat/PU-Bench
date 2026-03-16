#!/usr/bin/env python3
"""Simple Neural PU Loss with 40 features and direct aggregation.

This is a simplified version of NeuralPULoss that:
- Uses 40 input features (expanded from 13)
- Removes the split-multiply step for simplicity
- Directly sums over batch dimension
- Applies learned weights W1 and W2 to linear and log terms

Architecture:
    Input: 40 feature vectors of size B
        Base features (8):
        - 1, 1/B, p, p/B, log(p), log(p)/B, log(1-p), log(1-p)/B

        M_p masked (8):
        - M_p*1, M_p*1/B, M_p*p, M_p*p/B, M_p*log(p), M_p*log(p)/B, M_p*log(1-p), M_p*log(1-p)/B

        M_u masked (8):
        - M_u*1, M_u*1/B, M_u*p, M_u*p/B, M_u*log(p), M_u*log(p)/B, M_u*log(1-p), M_u*log(1-p)/B

        M_p normalized (8):
        - (M_p/M_p.sum())*1, ..., (M_p/M_p.sum())*log(1-p)/B

        M_u normalized (8):
        - (M_u/M_u.sum())*1, ..., (M_u/M_u.sum())*log(1-p)/B

    → Stack to [B, 40]
    → Linear(40, hidden_dim) → [B, hidden_dim]
    → Sum over batch → [hidden_dim]
    → Split → A1, A2 (each hidden_dim/2)
    → Learned params W1, W2 (each hidden_dim/2)
    → O1 = W1*A1, O2 = W2*log(|A2|)
    → Loss = O1.sum() + O2.sum()

Parameters (hidden_dim=128):
    - Linear layer: 40 × 128 + 128 = 5,248
    - W1: 64, W2: 64
    - Total: 5,376 parameters
"""

import torch
import torch.nn as nn
import torch.nn.init as init


class SimpleNeuralPULoss(nn.Module):
    """Simple neural PU loss with 40 features and direct aggregation.

    Parameters:
        hidden_dim: Hidden layer dimension (must be divisible by 2)
        eps: Numerical stability epsilon
        l1_lambda: L1 regularization strength
        l05_lambda: L0.5 regularization strength
        init_mode: Weight initialization strategy
        init_scale: Scaling factor for initialization
        max_weight_norm: Maximum weight norm (for clipping)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        eps: float = 1e-7,
        l1_lambda: float = 0.0,
        l05_lambda: float = 0.0,
        init_mode: str = 'xavier_uniform',
        init_scale: float = 1.0,
        max_weight_norm: float = 10.0,
    ):
        super().__init__()

        # Validate hidden_dim
        if hidden_dim % 2 != 0:
            raise ValueError(
                f"hidden_dim must be divisible by 2 (for splitting), got {hidden_dim}"
            )
        if hidden_dim < 2:
            raise ValueError(f"hidden_dim must be at least 2, got {hidden_dim}")

        self.hidden_dim = hidden_dim
        self.eps = eps
        self.l1_lambda = l1_lambda
        self.l05_lambda = l05_lambda
        self.init_mode = init_mode
        self.init_scale = init_scale
        self.max_weight_norm = max_weight_norm

        # Linear layer: 40 input features → hidden_dim
        self.linear = nn.Linear(40, hidden_dim)

        # Learned weights for final aggregation (hidden_dim/2 each)
        # W1 weights the linear terms (A1), W2 weights the log terms (A2)
        half = hidden_dim // 2
        self.W1 = nn.Parameter(torch.ones(half))  # Linear term weights
        self.W2 = nn.Parameter(torch.ones(half))  # Log term weights

        # Initialize weights
        self._initialize_weights()

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

            elif self.init_mode == 'random_normal':
                init.normal_(self.linear.weight, mean=0.0, std=self.init_scale)
                init.zeros_(self.linear.bias)

            else:
                raise ValueError(f"Unknown init_mode: {self.init_mode}")

    def safe_log(self, x: torch.Tensor) -> torch.Tensor:
        """Compute safe logarithm with numerical stability.

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
        """Compute simple neural PU loss.

        Args:
            outputs: Model logits [batch_size] or [batch_size, 1]
            labels: PU labels [batch_size] where 1=positive, -1=unlabeled
            mode: 'pu' for PU learning (only mode supported)

        Returns:
            Scalar loss tensor with gradients
        """
        if mode != 'pu':
            raise ValueError(f"SimpleNeuralPULoss only supports mode='pu', got '{mode}'")

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

        # Step 5: Compute batch size and normalizations
        B = float(len(M_p))
        B_inv = 1.0 / B

        # Normalize masks by their sums (for expectation-like features)
        M_p_sum = M_p.sum()
        M_u_sum = M_u.sum()
        M_p_norm = M_p / (M_p_sum + self.eps)  # Normalized positive mask
        M_u_norm = M_u / (M_u_sum + self.eps)  # Normalized unlabeled mask

        # Step 6: Compute log features
        log_p = self.safe_log(p)
        log_1mp = self.safe_log(1 - p)

        # Step 7: Create 40 features (5 groups of 8)
        # Group 1: Base features (8)
        ones = torch.ones_like(p)
        base_features = torch.stack([
            ones,           # 1
            ones * B_inv,   # 1/B
            p,              # p
            p * B_inv,      # p/B
            log_p,          # log(p)
            log_p * B_inv,  # log(p)/B
            log_1mp,        # log(1-p)
            log_1mp * B_inv # log(1-p)/B
        ], dim=1)  # [B, 8]

        # Group 2: M_p masked features (8)
        M_p_features = torch.stack([
            M_p,               # M_p * 1
            M_p * B_inv,       # M_p * 1/B
            M_p * p,           # M_p * p
            M_p * p * B_inv,   # M_p * p/B
            M_p * log_p,       # M_p * log(p)
            M_p * log_p * B_inv,   # M_p * log(p)/B
            M_p * log_1mp,     # M_p * log(1-p)
            M_p * log_1mp * B_inv  # M_p * log(1-p)/B
        ], dim=1)  # [B, 8]

        # Group 3: M_u masked features (8)
        M_u_features = torch.stack([
            M_u,
            M_u * B_inv,
            M_u * p,
            M_u * p * B_inv,
            M_u * log_p,
            M_u * log_p * B_inv,
            M_u * log_1mp,
            M_u * log_1mp * B_inv
        ], dim=1)  # [B, 8]

        # Group 4: M_p normalized features (8)
        M_p_norm_features = torch.stack([
            M_p_norm,
            M_p_norm * B_inv,
            M_p_norm * p,
            M_p_norm * p * B_inv,
            M_p_norm * log_p,
            M_p_norm * log_p * B_inv,
            M_p_norm * log_1mp,
            M_p_norm * log_1mp * B_inv
        ], dim=1)  # [B, 8]

        # Group 5: M_u normalized features (8)
        M_u_norm_features = torch.stack([
            M_u_norm,
            M_u_norm * B_inv,
            M_u_norm * p,
            M_u_norm * p * B_inv,
            M_u_norm * log_p,
            M_u_norm * log_p * B_inv,
            M_u_norm * log_1mp,
            M_u_norm * log_1mp * B_inv
        ], dim=1)  # [B, 8]

        # Step 8: Concatenate all features
        input_tensor = torch.cat([
            base_features,
            M_p_features,
            M_u_features,
            M_p_norm_features,
            M_u_norm_features
        ], dim=1)  # [B, 40]

        # Step 9: Linear transformation
        T = self.linear(input_tensor)  # [B, hidden_dim]

        # Step 10: Sum over batch (instead of split-multiply)
        A = T.sum(dim=0)  # [hidden_dim]

        # Step 11: Split A and compute final loss with learned weights
        half = self.hidden_dim // 2
        A1 = A[:half]  # [hidden_dim/2]
        A2 = A[half:]  # [hidden_dim/2]

        # Apply learned weights to each term
        O1 = self.W1 * A1  # [hidden_dim/2] - weighted linear terms
        O2 = self.W2 * self.safe_log(torch.abs(A2))  # [hidden_dim/2] - weighted log terms

        loss = O1.sum() + O2.sum()

        # Step 12: Add regularization
        if self.l1_lambda > 0 or self.l05_lambda > 0:
            loss = loss + self.compute_regularization()

        return loss

    def compute_regularization(self) -> torch.Tensor:
        """Compute L1 and L0.5 regularization.

        L1 encourages sparsity: sum(|w|)
        L0.5 encourages stronger sparsity: sum((|w| + eps)^0.5)

        Applied to all learnable parameters: linear layer weights/biases and W1/W2.

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
            f"SimpleNeuralPULoss(\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  num_parameters={num_params},\n"
            f"  num_features=40,\n"
            f"  eps={self.eps},\n"
            f"  l1_lambda={self.l1_lambda},\n"
            f"  l05_lambda={self.l05_lambda},\n"
            f"  init_mode='{self.init_mode}',\n"
            f"  init_scale={self.init_scale},\n"
            f"  max_weight_norm={self.max_weight_norm}\n"
            f")"
        )
