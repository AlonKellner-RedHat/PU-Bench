"""vpudra_naive_logmse_trainer.py

VPUDRaNaiveLogMSETrainer implements VPUDRa with original PUDRa loss (no prior)
and VPU's log-MSE consistency.

Key Differences:
    - VPUDRa-Fixed: π * E_P[-log p] + E_U[p] + log-MSE (WITH prior)
    - VPUDRa-naive-logmse: E_P[-log p + p] + E_U[p] + log-MSE (NO prior)
    - VPUDRa-naive: E_P[-log p + p] + E_U[p] + Point Process
    - VPUDRa-PP: π * E_P[-log p] + E_U[p] + Point Process

This completes the 2x2 design matrix:
    |               | Log-MSE consistency | Point Process consistency |
    |---------------|---------------------|---------------------------|
    | With prior    | VPUDRa-Fixed       | VPUDRa-PP                |
    | Without prior | VPUDRa-naive-logmse| VPUDRa-naive             |

This tests whether VPU's log-MSE consistency works better than Point Process
when no prior weighting is used in the base loss.
"""

from __future__ import annotations

import torch
from loss.loss_vpudra_naive_logmse import VPUDRaNaiveLogMSELoss

from .base_trainer import BaseTrainer


class VPUDRaNaiveLogMSETrainer(BaseTrainer):
    """VPUDRa trainer with original PUDRa loss (no prior) and VPU's log-MSE consistency."""

    def create_criterion(self):
        """Create VPUDRa-naive-logmse loss function with configured hyperparameters."""
        mix_alpha = self.params.get("mix_alpha", 0.3)
        epsilon = self.params.get("epsilon", 1e-7)
        # NOTE: No prior parameter - VPUDRa-naive-logmse ignores the prior
        return VPUDRaNaiveLogMSELoss(mix_alpha=mix_alpha, epsilon=epsilon)

    def train_one_epoch(self, epoch_idx: int):
        """Training loop for one epoch (VPUDRa-naive-logmse with MixUp)."""
        self.model.train()

        for x, t, _y_true, _idx, _ in self.train_loader:  # type: ignore
            x, t = x.to(self.device), t.to(self.device)

            # Create positive mask (1=positive, -1=unlabeled -> convert to boolean)
            p_mask = t == 1
            p_features = x[p_mask]

            # Skip batches with no positive samples
            if len(p_features) == 0:
                continue

            # Forward pass on original batch
            # p_all = σ(f(x)) ∈ [0,1] for all samples
            p_all = torch.sigmoid(self.model(x))

            # MixUp: sample positive features for interpolation
            # Same strategy as VPU: sample with replacement if needed
            if p_features.size(0) >= x.size(0):
                # Enough positives: sample without replacement
                rand_perm = torch.randperm(p_features.size(0), device=self.device)[
                    : x.size(0)
                ]
                p_mix_features = p_features[rand_perm]
            else:
                # Not enough positives: sample with replacement
                idx = torch.randint(
                    0, p_features.size(0), (x.size(0),), device=self.device
                )
                p_mix_features = p_features[idx]

            # Sample MixUp coefficient from Beta(mix_alpha, mix_alpha)
            m = torch.distributions.beta.Beta(
                self.criterion.mix_alpha, self.criterion.mix_alpha
            )
            lam = m.sample().to(self.device)
            lam_float = float(lam)

            # Create mixed samples: x_mix = lam * x + (1-lam) * p_mix_features
            sam_data = lam * x + (1 - lam) * p_mix_features

            # Create anchored MixUp target (ANCHOR ASSUMPTION for stability)
            # μ = λ * p(x) + (1-λ) * 1.0
            # Note: detach p(x) to avoid double backprop
            pos_prob = torch.sigmoid(self.model(x)).detach()
            mu_anchor = lam_float * pos_prob + (1 - lam_float) * torch.ones_like(
                pos_prob
            )

            # Forward pass on mixed samples
            # p_mix = σ(f(x_mix)) ∈ [0,1]
            p_mix_output = torch.sigmoid(self.model(sam_data))

            # Compute VPUDRa-naive-logmse loss
            # Uses original PUDRa form WITHOUT prior: E_P[-log p + p] + E_U[p]
            # Uses VPU's log-MSE consistency: (log μ - log p)²
            self.optimizer.zero_grad()
            loss = self.criterion(
                p_all,          # σ(f(x)) for all samples
                t,              # PU labels (1=positive, -1=unlabeled)
                p_mix_output,   # σ(f(x_mix)) for mixed samples
                mu_anchor,      # λ*p(x) + (1-λ)*1.0 (anchored target)
                lam_float       # MixUp coefficient
            )
            loss.backward()
            self.optimizer.step()
