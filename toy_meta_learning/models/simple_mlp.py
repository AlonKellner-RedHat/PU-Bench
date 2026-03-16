"""Simple MLP model for binary classification."""

import torch
import torch.nn as nn
from typing import List


class SimpleMLP(nn.Module):
    """Simple multi-layer perceptron for binary classification.

    Architecture:
        Input(D) → Linear(h1) → ReLU → ... → Linear(hn) → ReLU → Linear(1)

    Output is logits (not probabilities). Use sigmoid for probabilities.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [32, 32],
        activation: str = 'relu',
    ):
        """Initialize MLP.

        Args:
            input_dim: Input feature dimensionality
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ('relu', 'tanh', 'sigmoid')
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Select activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            prev_dim = hidden_dim

        # Output layer (logits, no activation)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Logits [batch_size, 1]
        """
        return self.network(x)

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
