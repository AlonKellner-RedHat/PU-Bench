"""vpudra_pp_trainer.py

VPUDRaPointProcessTrainer implements VPUDRa with Point Process consistency loss.

Key Difference from VPUDRa-Fixed:
    - VPUDRa-Fixed: Uses VPU's log-MSE consistency: (log(target) - log(pred))²
    - VPUDRa-PP: Uses Point Process consistency: -target * log(pred) + pred

Both use the same anchored target: μ = λ * p(x) + (1-λ) * 1.0

This tests whether PUDRa's asymmetric Point Process loss provides better
consistency regularization than VPU's symmetric log-MSE.
"""

from __future__ import annotations

import torch
from loss.loss_vpudra_pp import VPUDRaPointProcessLoss

from .base_trainer import BaseTrainer


class VPUDRaPointProcessTrainer(BaseTrainer):
    """VPUDRa trainer with Point Process consistency (anchored)."""

    def create_criterion(self):
        """Create VPUDRa-PP loss function with configured hyperparameters."""
        mix_alpha = self.params.get("mix_alpha", 0.3)
        epsilon = self.params.get("epsilon", 1e-7)
        return VPUDRaPointProcessLoss(
            prior=self.prior, mix_alpha=mix_alpha, epsilon=epsilon
        )

    def train_one_epoch(self, epoch_idx: int):
        """Training loop for one epoch (VPUDRa-PP with MixUp)."""
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

            # Compute VPUDRa-PP loss
            # Uses Point Process consistency: -μ log p + p (instead of log-MSE)
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
