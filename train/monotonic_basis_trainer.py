"""Trainer for Learnable Monotonic Basis Loss.

This trainer handles training with the monotonic basis loss, supporting both
oracle (PN) and PU modes.

Key features:
- Standard training optimizes only model parameters (not loss parameters)
- For future meta-learning: would optimize both model and loss parameters
- Supports both oracle mode (binary labels) and PU mode (±1 labels)
- Compatible with prior-conditioned loss parameters
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.base_trainer import BaseTrainer
from loss.loss_monotonic_basis import MonotonicBasisLoss


class MonotonicBasisTrainer(BaseTrainer):
    """Trainer for learnable monotonic basis loss.

    Supports both oracle (PN) and PU modes via oracle_mode parameter in config.

    Configuration parameters (from config/methods/monotonic_basis.yaml):
        - num_repetitions: Number of repetition blocks (default 3)
        - num_fourier: Number of Fourier coefficients (default 5)
        - use_prior: Whether to condition on prior (default True)
        - oracle_mode: If True, uses binary labels; if False, PU labels (default False)
        - init_scale: Initialization scale for parameters (default 0.01)

    Examples:
        >>> from train.monotonic_basis_trainer import MonotonicBasisTrainer
        >>> # Standard PU training
        >>> trainer = MonotonicBasisTrainer(params, device, train_loader, test_loader)
        >>> trainer.run()

        >>> # Oracle mode training (would need oracle_mode=true in config)
        >>> # trainer_oracle = MonotonicBasisTrainer(params_oracle, ...)
        >>> # trainer_oracle.run()
    """

    def __init__(
        self, params: dict, device, train_loader, test_loader, val_loader=None
    ):
        """Initialize trainer.

        Args:
            params: Configuration dictionary from YAML file
            device: PyTorch device (cuda/mps/cpu)
            train_loader: Training data loader
            test_loader: Test data loader
            val_loader: Optional validation data loader
        """
        super().__init__(params, device, train_loader, test_loader, val_loader)

    def create_criterion(self):
        """Create monotonic basis loss with parameters from config.

        Reads hyperparameters from self.params and creates the loss module.

        Returns:
            MonotonicBasisLoss instance configured according to params

        Note:
            The loss is created with self.prior, which is computed from
            the dataset in BaseTrainer._build_model().
        """
        num_repetitions = self.params.get("num_repetitions", 3)
        num_fourier = self.params.get("num_fourier", 5)
        use_prior = self.params.get("use_prior", True)
        oracle_mode = self.params.get("oracle_mode", False)
        init_scale = self.params.get("init_scale", 0.01)

        loss = MonotonicBasisLoss(
            num_repetitions=num_repetitions,
            num_fourier=num_fourier,
            use_prior=use_prior,
            prior=self.prior,
            oracle_mode=oracle_mode,
            init_scale=init_scale,
        ).to(self.device)

        # Print configuration summary
        print(f"\nMonotonic Basis Loss Configuration:")
        print(f"  Num repetitions: {num_repetitions}")
        print(f"  Num Fourier terms: {num_fourier}")
        print(f"  Use prior: {use_prior}")
        print(f"  Oracle mode: {oracle_mode}")
        print(f"  Prior: {self.prior:.4f}")
        print(f"  Init scale: {init_scale}")
        summary = loss.get_parameter_summary()
        print(f"  Parameter summary:")
        print(f"    Total params: {summary['total_params']}")
        print(f"    Baseline params: {summary['baseline_params']}")
        print(f"    Fourier params: {summary['fourier_params']}")
        print(f"    Num basis functions: {summary['num_basis_functions']}")

        return loss

    def train_one_epoch(self, epoch_idx: int):
        """Train for one epoch.

        Standard training loop that optimizes only model parameters.

        For meta-learning (future extension), would also optimize loss parameters
        by modifying the optimizer in _build_model() to include self.criterion.parameters().

        Args:
            epoch_idx: Current epoch index (0-based)

        Note:
            This follows the standard PU-Bench training pattern:
            - Only model parameters are optimized
            - Loss parameters remain fixed (initialized randomly)
            - For meta-learning, would optimize both
        """
        self.model.train()

        for x, t, _y_true, _idx, _ in self.train_loader:
            x, t = x.to(self.device), t.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(x).view(-1)

            # Compute loss
            loss = self.criterion(outputs, t)

            # Backward pass
            loss.backward()

            # Update only model parameters (not loss parameters)
            # For meta-learning, you would include loss parameters in optimizer
            self.optimizer.step()
