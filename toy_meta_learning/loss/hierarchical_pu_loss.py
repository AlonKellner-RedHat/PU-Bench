"""Hierarchical 27-parameter PU loss with nested functions.

Loss structure:
    L_PU = f_p1(E_P[f_p2(f_p3(p))]) + f_u1(E_U[f_u2(f_u3(p))]) + f_a1(E_A[f_a2(f_a3(p))]) + λ·L1_reg

where:
    - Each f is the simple basis: f(x) = a1 + a2·x + a3·log(x)
    - E_P = mean over labeled positives
    - E_U = mean over unlabeled samples
    - E_A = mean over ALL samples (both groups combined)
    - Functions are nested: innermost (f_3) → middle (f_2) → mean → outermost (f_1)
    - L1 regularization encourages sparsity across all 27 parameters

This generalizes multiple PU methods:
- BCE: Set innermost to linear/reverse, middle to -log, outer to identity
- VPU, PUDRa, uPU: Can be represented with specific parameter settings
- Novel structures: Meta-learning can discover new useful transformations

Parameters:
    Labeled Positives (9 params): a1_p1, a2_p1, a3_p1, a1_p2, a2_p2, a3_p2, a1_p3, a2_p3, a3_p3
    Unlabeled (9 params): a1_u1, a2_u1, a3_u1, a1_u2, a2_u2, a3_u2, a1_u3, a2_u3, a3_u3
    All Samples (9 params): a1_a1, a2_a1, a3_a1, a1_a2, a2_a2, a3_a2, a1_a3, a2_a3, a3_a3
"""

import torch
import torch.nn as nn


class HierarchicalPULoss(nn.Module):
    """Learnable 27-parameter hierarchical PU loss with nested transformations.

    Three sample groups (labeled positives, unlabeled, all samples), each with
    three nested basis functions (innermost, middle, outermost).

    Parameters:
        Labeled Positives: f_p1, f_p2, f_p3 (9 parameters)
        Unlabeled: f_u1, f_u2, f_u3 (9 parameters)
        All Samples: f_a1, f_a2, f_a3 (9 parameters)
    """

    def __init__(
        self,
        init_mode: str = 'random',
        init_scale: float = 0.01,
        l1_lambda: float = 0.0,
        eps: float = 1e-7,
    ):
        """Initialize the hierarchical PU loss.

        Args:
            init_mode: Initialization mode ('random', 'zeros', 'bce_equivalent',
                      'identity_chain', 'diverse_init', 'pudra_inspired', 'vpu_inspired')
            init_scale: Scale for random initialization
            l1_lambda: L1 regularization coefficient (encourages sparsity)
            eps: Numerical stability epsilon for log operations
        """
        super().__init__()

        self.init_mode = init_mode
        self.init_scale = init_scale
        self.l1_lambda = l1_lambda
        self.eps = eps

        # === LABELED POSITIVES GROUP (9 parameters) ===
        # f_p1: outermost (applied to mean)
        self.a1_p1 = nn.Parameter(torch.zeros(1))
        self.a2_p1 = nn.Parameter(torch.zeros(1))
        self.a3_p1 = nn.Parameter(torch.zeros(1))

        # f_p2: middle
        self.a1_p2 = nn.Parameter(torch.zeros(1))
        self.a2_p2 = nn.Parameter(torch.zeros(1))
        self.a3_p2 = nn.Parameter(torch.zeros(1))

        # f_p3: innermost (applied to each sample)
        self.a1_p3 = nn.Parameter(torch.zeros(1))
        self.a2_p3 = nn.Parameter(torch.zeros(1))
        self.a3_p3 = nn.Parameter(torch.zeros(1))

        # === UNLABELED GROUP (9 parameters) ===
        # f_u1: outermost
        self.a1_u1 = nn.Parameter(torch.zeros(1))
        self.a2_u1 = nn.Parameter(torch.zeros(1))
        self.a3_u1 = nn.Parameter(torch.zeros(1))

        # f_u2: middle
        self.a1_u2 = nn.Parameter(torch.zeros(1))
        self.a2_u2 = nn.Parameter(torch.zeros(1))
        self.a3_u2 = nn.Parameter(torch.zeros(1))

        # f_u3: innermost
        self.a1_u3 = nn.Parameter(torch.zeros(1))
        self.a2_u3 = nn.Parameter(torch.zeros(1))
        self.a3_u3 = nn.Parameter(torch.zeros(1))

        # === ALL SAMPLES GROUP (9 parameters) ===
        # f_a1: outermost
        self.a1_a1 = nn.Parameter(torch.zeros(1))
        self.a2_a1 = nn.Parameter(torch.zeros(1))
        self.a3_a1 = nn.Parameter(torch.zeros(1))

        # f_a2: middle
        self.a1_a2 = nn.Parameter(torch.zeros(1))
        self.a2_a2 = nn.Parameter(torch.zeros(1))
        self.a3_a2 = nn.Parameter(torch.zeros(1))

        # f_a3: innermost
        self.a1_a3 = nn.Parameter(torch.zeros(1))
        self.a2_a3 = nn.Parameter(torch.zeros(1))
        self.a3_a3 = nn.Parameter(torch.zeros(1))

        # Initialize
        self._initialize_params()

    def _initialize_params(self):
        """Initialize parameters based on init_mode."""
        with torch.no_grad():
            if self.init_mode == 'random':
                # Random initialization for all 27 parameters
                for param in self.parameters():
                    param.data = torch.randn(1) * self.init_scale

            elif self.init_mode == 'zeros':
                # All zeros (already initialized)
                pass

            elif self.init_mode == 'bce_equivalent':
                # Approximate BCE: -E_P[log(p)] - E_U[log(1-p)]

                # Positive group: identity → -log → identity
                self.a1_p3.data = torch.tensor([0.0])  # f_p3(p) = p
                self.a2_p3.data = torch.tensor([1.0])
                self.a3_p3.data = torch.tensor([0.0])

                self.a1_p2.data = torch.tensor([0.0])  # f_p2(x) = -log(x)
                self.a2_p2.data = torch.tensor([0.0])
                self.a3_p2.data = torch.tensor([-1.0])

                self.a1_p1.data = torch.tensor([0.0])  # f_p1(x) = x
                self.a2_p1.data = torch.tensor([1.0])
                self.a3_p1.data = torch.tensor([0.0])

                # Unlabeled group: reverse → -log → identity
                self.a1_u3.data = torch.tensor([1.0])  # f_u3(p) = 1-p
                self.a2_u3.data = torch.tensor([-1.0])
                self.a3_u3.data = torch.tensor([0.0])

                self.a1_u2.data = torch.tensor([0.0])  # f_u2(x) = -log(x)
                self.a2_u2.data = torch.tensor([0.0])
                self.a3_u2.data = torch.tensor([-1.0])

                self.a1_u1.data = torch.tensor([0.0])  # f_u1(x) = x
                self.a2_u1.data = torch.tensor([1.0])
                self.a3_u1.data = torch.tensor([0.0])

                # All samples group: zeros (no contribution)
                # Already initialized to zero

            elif self.init_mode == 'identity_chain':
                # All functions = identity (a1=0, a2=1, a3=0)
                for name, param in self.named_parameters():
                    if 'a1' in name:
                        param.data = torch.tensor([0.0])
                    elif 'a2' in name:
                        param.data = torch.tensor([1.0])
                    elif 'a3' in name:
                        param.data = torch.tensor([0.0])

            elif self.init_mode == 'diverse_init':
                # Positive group: BCE-like
                self.a1_p3.data = torch.tensor([0.0])
                self.a2_p3.data = torch.tensor([1.0])
                self.a3_p3.data = torch.tensor([0.0])

                self.a1_p2.data = torch.tensor([0.0])
                self.a2_p2.data = torch.tensor([0.0])
                self.a3_p2.data = torch.tensor([-1.0])

                self.a1_p1.data = torch.tensor([0.0])
                self.a2_p1.data = torch.tensor([1.0])
                self.a3_p1.data = torch.tensor([0.0])

                # Unlabeled group: identity chain
                for name, param in self.named_parameters():
                    if 'u' in name and 'a1' in name:
                        param.data = torch.tensor([0.0])
                    elif 'u' in name and 'a2' in name:
                        param.data = torch.tensor([1.0])
                    elif 'u' in name and 'a3' in name:
                        param.data = torch.tensor([0.0])

                # All group: small random
                for name, param in self.named_parameters():
                    if 'a' in name and name.startswith('a'):  # All group parameters
                        param.data = torch.randn(1) * self.init_scale

            elif self.init_mode == 'pudra_inspired':
                # Exact PUDRa-naive: L = E_P[-log(p) + p] + E_U[p]

                # Positive group: E_P[-log(p) + p]
                # f_p3: identity → p
                self.a1_p3.data = torch.tensor([0.0])
                self.a2_p3.data = torch.tensor([1.0])
                self.a3_p3.data = torch.tensor([0.0])

                # f_p2: -log(p) + p
                self.a1_p2.data = torch.tensor([0.0])
                self.a2_p2.data = torch.tensor([1.0])  # linear term: p
                self.a3_p2.data = torch.tensor([-1.0])  # log term: -log(p)

                # f_p1: identity (preserve mean)
                self.a1_p1.data = torch.tensor([0.0])
                self.a2_p1.data = torch.tensor([1.0])
                self.a3_p1.data = torch.tensor([0.0])

                # Unlabeled group: E_U[p]
                # All three functions are identity
                self.a1_u3.data = torch.tensor([0.0])
                self.a2_u3.data = torch.tensor([1.0])
                self.a3_u3.data = torch.tensor([0.0])

                self.a1_u2.data = torch.tensor([0.0])
                self.a2_u2.data = torch.tensor([1.0])
                self.a3_u2.data = torch.tensor([0.0])

                self.a1_u1.data = torch.tensor([0.0])
                self.a2_u1.data = torch.tensor([1.0])
                self.a3_u1.data = torch.tensor([0.0])

                # All samples group: zeros (no contribution)
                # Already initialized to zero

            elif self.init_mode == 'vpu_inspired':
                # VPU-NoMixUp: L = log(E_all[p]) - E_P[log(p)]

                # Positive group: E_P[-log(p)]
                # f_p3: identity → p
                self.a1_p3.data = torch.tensor([0.0])
                self.a2_p3.data = torch.tensor([1.0])
                self.a3_p3.data = torch.tensor([0.0])

                # f_p2: -log(p)
                self.a1_p2.data = torch.tensor([0.0])
                self.a2_p2.data = torch.tensor([0.0])
                self.a3_p2.data = torch.tensor([-1.0])

                # f_p1: identity (preserve mean)
                self.a1_p1.data = torch.tensor([0.0])
                self.a2_p1.data = torch.tensor([1.0])
                self.a3_p1.data = torch.tensor([0.0])

                # Unlabeled group: zeros (no contribution in VPU)
                # Already initialized to zero

                # All samples group: log(E_A[p])
                # f_a3: identity → p
                self.a1_a3.data = torch.tensor([0.0])
                self.a2_a3.data = torch.tensor([1.0])
                self.a3_a3.data = torch.tensor([0.0])

                # f_a2: identity → preserve p
                self.a1_a2.data = torch.tensor([0.0])
                self.a2_a2.data = torch.tensor([1.0])
                self.a3_a2.data = torch.tensor([0.0])

                # f_a1: log(mean(p))
                self.a1_a1.data = torch.tensor([0.0])
                self.a2_a1.data = torch.tensor([0.0])
                self.a3_a1.data = torch.tensor([1.0])

            else:
                raise ValueError(f"Unknown init_mode: {self.init_mode}")

    def apply_basis(self, x: torch.Tensor, a1, a2, a3, is_probability: bool = True) -> torch.Tensor:
        """Apply basis function f(x) = a1 + a2·x + a3·log(x).

        Args:
            x: Input tensor
            a1, a2, a3: Basis coefficients
            is_probability: If True, clamp to [eps, 1-eps] for probabilities;
                          if False, allow wider range for intermediate values

        Returns:
            Transformed values with clamping for numerical stability
        """
        x = x.view(-1)

        if is_probability:
            # Innermost functions on probabilities: tight clamping
            x_safe = torch.clamp(x, min=self.eps, max=1.0 - self.eps)
            max_output = 100.0
        else:
            # Middle/outer functions on intermediate values: wider range
            x_safe = torch.clamp(x, min=self.eps, max=1e6)
            max_output = 1e3

        # Apply basis: f(x) = a1 + a2·x + a3·log(x + eps)
        # QUICK WIN 1: Epsilon INSIDE log for extra numerical stability
        # This is especially important when x is a mean value that could be very small
        result = a1 + a2 * x_safe + a3 * torch.log(x_safe + self.eps)

        # Clamp output to prevent overflow in subsequent layers
        result = torch.clamp(result, min=-max_output, max=max_output)

        return result

    def apply_nested_group(
        self,
        p_group: torch.Tensor,
        params_inner: tuple,
        params_middle: tuple,
        params_outer: tuple,
    ) -> torch.Tensor:
        """Apply three nested functions with mean in between.

        Computes: f_1(mean(f_2(f_3(p))))

        Args:
            p_group: Probabilities for this group [N]
            params_inner: (a1, a2, a3) for innermost function
            params_middle: (a1, a2, a3) for middle function
            params_outer: (a1, a2, a3) for outermost function

        Returns:
            Scalar result
        """
        # Innermost: apply to each sample (probabilities)
        z = self.apply_basis(p_group, *params_inner, is_probability=True)

        # Middle: apply to each transformed sample (intermediate values)
        y = self.apply_basis(z, *params_middle, is_probability=False)

        # Take mean
        mean_val = y.mean()

        # Outermost: apply to scalar mean
        result = self.apply_basis(mean_val.view(1), *params_outer, is_probability=False)

        return result.squeeze()

    def forward(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        mode: str = 'pu',
    ) -> torch.Tensor:
        """Compute hierarchical PU loss.

        L_PU = f_p1(E_P[f_p2(f_p3(p))]) + f_u1(E_U[f_u2(f_u3(p))]) + f_a1(E_A[f_a2(f_a3(p))]) + λ·L1_reg

        Args:
            outputs: Model logits [batch_size, 1] or [batch_size]
            labels: PU labels [batch_size] (1 = labeled positive, -1 = unlabeled)
            mode: 'pu' (only mode supported)

        Returns:
            Scalar loss value
        """
        if mode != 'pu':
            raise ValueError(f"HierarchicalPULoss only supports mode='pu', got '{mode}'")

        # Convert to probabilities
        p = torch.sigmoid(outputs.view(-1))
        labels = labels.view(-1)

        # Separate into groups
        pos_mask = labels == 1
        unlabeled_mask = labels == -1

        p_pos = p[pos_mask]
        p_unlabeled = p[unlabeled_mask]
        p_all = p

        # Handle edge cases
        if len(p_pos) == 0:
            return torch.tensor(0.0, device=p.device, requires_grad=True)

        # === Labeled Positives Branch ===
        result_p = self.apply_nested_group(
            p_pos,
            (self.a1_p3, self.a2_p3, self.a3_p3),  # innermost
            (self.a1_p2, self.a2_p2, self.a3_p2),  # middle
            (self.a1_p1, self.a2_p1, self.a3_p1),  # outermost
        )

        # === Unlabeled Branch ===
        if len(p_unlabeled) > 0:
            result_u = self.apply_nested_group(
                p_unlabeled,
                (self.a1_u3, self.a2_u3, self.a3_u3),
                (self.a1_u2, self.a2_u2, self.a3_u2),
                (self.a1_u1, self.a2_u1, self.a3_u1),
            )
        else:
            result_u = torch.tensor(0.0, device=p.device)

        # === All Samples Branch ===
        result_a = self.apply_nested_group(
            p_all,
            (self.a1_a3, self.a2_a3, self.a3_a3),
            (self.a1_a2, self.a2_a2, self.a3_a2),
            (self.a1_a1, self.a2_a1, self.a3_a1),
        )

        # Sum all three terms
        loss = result_p + result_u + result_a

        # Add L1 regularization
        if self.l1_lambda > 0:
            loss = loss + self.compute_l1_regularization()

        return loss

    def compute_l1_regularization(self) -> torch.Tensor:
        """Compute L1 penalty on all 27 parameters.

        Returns:
            L1 penalty = l1_lambda * sum(|parameters|)
        """
        params = self.get_parameters()
        return self.l1_lambda * torch.abs(params).sum()

    def get_parameters(self) -> torch.Tensor:
        """Get all 27 parameters as a single tensor.

        Returns:
            Tensor of shape [27] containing all parameters in order:
            [p1_params, p2_params, p3_params, u1_params, u2_params, u3_params, a1_params, a2_params, a3_params]
        """
        return torch.stack([
            # Positive group (9 params)
            self.a1_p1.squeeze(), self.a2_p1.squeeze(), self.a3_p1.squeeze(),
            self.a1_p2.squeeze(), self.a2_p2.squeeze(), self.a3_p2.squeeze(),
            self.a1_p3.squeeze(), self.a2_p3.squeeze(), self.a3_p3.squeeze(),
            # Unlabeled group (9 params)
            self.a1_u1.squeeze(), self.a2_u1.squeeze(), self.a3_u1.squeeze(),
            self.a1_u2.squeeze(), self.a2_u2.squeeze(), self.a3_u2.squeeze(),
            self.a1_u3.squeeze(), self.a2_u3.squeeze(), self.a3_u3.squeeze(),
            # All samples group (9 params)
            self.a1_a1.squeeze(), self.a2_a1.squeeze(), self.a3_a1.squeeze(),
            self.a1_a2.squeeze(), self.a2_a2.squeeze(), self.a3_a2.squeeze(),
            self.a1_a3.squeeze(), self.a2_a3.squeeze(), self.a3_a3.squeeze(),
        ])

    def get_num_parameters(self) -> int:
        """Get number of learnable parameters."""
        return 27

    def __repr__(self):
        """String representation showing all 27 parameters."""
        params = self.get_parameters().detach().cpu().numpy()

        return (
            f"HierarchicalPULoss(\n"
            f"  Positive Group:\n"
            f"    f_p1 (outer): a1={params[0]:.4f}, a2={params[1]:.4f}, a3={params[2]:.4f}\n"
            f"    f_p2 (mid):   a1={params[3]:.4f}, a2={params[4]:.4f}, a3={params[5]:.4f}\n"
            f"    f_p3 (inner): a1={params[6]:.4f}, a2={params[7]:.4f}, a3={params[8]:.4f}\n"
            f"  Unlabeled Group:\n"
            f"    f_u1 (outer): a1={params[9]:.4f}, a2={params[10]:.4f}, a3={params[11]:.4f}\n"
            f"    f_u2 (mid):   a1={params[12]:.4f}, a2={params[13]:.4f}, a3={params[14]:.4f}\n"
            f"    f_u3 (inner): a1={params[15]:.4f}, a2={params[16]:.4f}, a3={params[17]:.4f}\n"
            f"  All Samples Group:\n"
            f"    f_a1 (outer): a1={params[18]:.4f}, a2={params[19]:.4f}, a3={params[20]:.4f}\n"
            f"    f_a2 (mid):   a1={params[21]:.4f}, a2={params[22]:.4f}, a3={params[23]:.4f}\n"
            f"    f_a3 (inner): a1={params[24]:.4f}, a2={params[25]:.4f}, a3={params[26]:.4f}\n"
            f"  L1_lambda: {self.l1_lambda}\n"
            f")"
        )
