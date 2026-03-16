import torch
from torch import nn


class VPUMeanLoss(nn.Module):
    """VPU loss with mean (no log-of-mean) and MixUp regularization.

    This variant replaces log(mean(φ(x))) with mean(φ(x)) while keeping the MixUp regularization.
    """

    def __init__(self, args):
        super(VPUMeanLoss, self).__init__()
        self.mix_alpha = args.mix_alpha
        self.name = "vpu_mean"

    def forward(self, output_phi_all, targets, out_log_phi_all, sam_target, lam):
        """
        Args:
            output_phi_all: log(φ(x)) for all samples in batch
            targets: binary labels (1 for positive, 0 for unlabeled)
            out_log_phi_all: log(φ(x_mixed)) for mixed samples
            sam_target: mixed targets for mixup regularization
            lam: mixup weight
        """
        log_phi_x = output_phi_all
        log_phi_p = output_phi_all[targets == 1]

        # Convert log_phi to phi for the mean computation
        phi_all = torch.exp(log_phi_x)

        # Variance reduction term: mean(φ(x)) - mean(log(φ_p(x)))
        # This removes the log-of-mean compared to original VPU
        var_loss = torch.mean(phi_all) - torch.mean(log_phi_p)

        # MixUp regularization: E[(log(sam_target) - log(φ(x_mixed)))²]
        reg_mix_log = ((torch.log(sam_target) - out_log_phi_all) ** 2).mean()

        # Total loss
        phi_loss = var_loss + lam * reg_mix_log

        return phi_loss
