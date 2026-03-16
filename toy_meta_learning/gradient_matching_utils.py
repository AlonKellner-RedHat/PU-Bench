#!/usr/bin/env python3
"""Gradient Matching Utilities.

Functions for computing gradient MSE between learned PU loss and oracle BCE,
and for advancing checkpoints through training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from typing import Dict, Tuple, Any
import numpy as np

from models.simple_mlp import SimpleMLP
from tasks.gaussian_task import GaussianBlobTask
from loss.baseline_losses import PUDRaNaiveLoss, VPUNoMixUpLoss


def compute_gradient_mse(
    model: SimpleMLP,
    params: Dict[str, torch.Tensor],
    x_batch: torch.Tensor,
    y_pu_batch: torch.Tensor,
    y_true_batch: torch.Tensor,
    learned_loss: nn.Module,
    device: str,
    eps: float = 1e-7,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute log(MSE) objective for gradient matching.

    Meta-objective: log(MSE(pu_grad, bce_grad)) → minimize difference between gradients

    ALWAYS matches learned PU loss gradients to oracle BCE gradients, regardless of
    checkpoint's training objective.

    Args:
        model: SimpleMLP instance
        params: Dictionary of model parameters (requires_grad=True)
        x_batch: Input features [batch_size, input_dim]
        y_pu_batch: PU labels [batch_size] (1=positive, -1=unlabeled)
        y_true_batch: True binary labels [batch_size] (0 or 1)
        learned_loss: NeuralPULoss module
        device: Device ('cpu', 'cuda', 'mps')
        eps: Epsilon for numerical stability

    Returns:
        meta_loss: Scalar loss for meta-optimization (WITH computational graph)
        diagnostics: Dictionary with gradient norms, cosine similarity, etc.
    """

    # 1. Compute PU loss gradients (WITH computational graph for meta-learning)
    def pu_loss_fn(param_dict):
        outputs = functional_call(model, param_dict, x_batch).squeeze(-1)
        return learned_loss(outputs, y_pu_batch, mode='pu')

    pu_grads = torch.func.grad(pu_loss_fn)(params)

    # 2. Compute oracle BCE gradients (WITHOUT computational graph - fixed targets)
    # ALWAYS use oracle BCE as target, regardless of checkpoint's training objective
    with torch.no_grad():
        def bce_loss_fn(param_dict):
            outputs = functional_call(model, param_dict, x_batch).squeeze(-1)
            return nn.BCEWithLogitsLoss()(outputs, y_true_batch)

        bce_grads_dict = torch.func.grad(bce_loss_fn)(params)

        # Detach oracle gradients to treat as fixed targets
        bce_grads = {k: v.detach() for k, v in bce_grads_dict.items()}

    # 3. Flatten gradients to vectors
    pu_grad_vec = torch.cat([g.flatten() for g in pu_grads.values()])
    bce_grad_vec = torch.cat([g.flatten() for g in bce_grads.values()])

    # 4. Compute cosine similarity (for diagnostics)
    cos_sim = F.cosine_similarity(pu_grad_vec.unsqueeze(0), bce_grad_vec.unsqueeze(0))
    cosine_loss = 1.0 - cos_sim.squeeze()

    # 5. Compute magnitude matching loss
    gradient_mse = torch.mean((pu_grad_vec - bce_grad_vec) ** 2)
    magnitude_loss = torch.sqrt(gradient_mse + eps)

    # 6. Meta-objective: cosine similarity (minimize 1 - cos_sim)
    meta_loss = cosine_loss

    # 7. Compute diagnostics (for logging)
    with torch.no_grad():
        pu_norm = torch.norm(pu_grad_vec)
        bce_norm = torch.norm(bce_grad_vec)

        # Relative MSE (normalized by BCE gradient norm)
        relative_mse = gradient_mse / (bce_norm ** 2 + eps)

        diagnostics = {
            'meta_loss': meta_loss.item(),
            'cosine_loss': cosine_loss.item(),
            'magnitude_loss': magnitude_loss.item(),
            'pu_grad_norm': pu_norm.item(),
            'bce_grad_norm': bce_norm.item(),
            'cosine_similarity': cos_sim.item(),
            'relative_mse': relative_mse.item(),
            'gradient_mse': gradient_mse.item(),
        }

    return meta_loss, diagnostics


def advance_checkpoint_one_step(
    checkpoint: Dict[str, Any],
    learned_loss: nn.Module,
    device: str,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """Take ONE training step and return updated checkpoint.

    Uses the checkpoint's assigned objective for training.

    Args:
        checkpoint: Checkpoint dictionary with task_config, model_state, optimizer_state, objective, etc.
        learned_loss: NeuralPULoss module
        device: Device ('cpu', 'cuda', 'mps')
        batch_size: Batch size for training step

    Returns:
        Updated checkpoint dictionary with new model_state, optimizer_state, step_count
    """

    # 1. Load checkpoint state
    model = SimpleMLP(
        input_dim=checkpoint['task_config']['num_dimensions'],
        hidden_dims=[32, 32],  # TODO: Make configurable if needed
    ).to(device)

    model.load_state_dict({
        k: v.to(device) for k, v in checkpoint['model_state'].items()
    })

    # Reconstruct optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,  # TODO: Make configurable if needed
        momentum=0.9,
    )

    # Load optimizer state
    optimizer_state_device = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in checkpoint['optimizer_state'].items()
    }
    optimizer.load_state_dict(optimizer_state_device)

    # 2. Get batch from checkpoint's task
    task = GaussianBlobTask(**checkpoint['task_config'])

    # Use step_count as seed offset for deterministic batching
    batch_seed = checkpoint['task_config']['seed'] + checkpoint['step_count']
    torch.manual_seed(batch_seed)
    np.random.seed(batch_seed)

    dataloaders = task.get_dataloaders(
        batch_size=batch_size,
        num_train=1000,
        num_val=500,
        num_test=500,
    )

    train_batch = next(iter(dataloaders['train']))
    x = train_batch[0].to(device)
    y_true = train_batch[1].to(device)
    y_pu = train_batch[2].to(device)

    # 3. Take one training step using checkpoint's assigned objective
    optimizer.zero_grad()
    outputs = model(x).squeeze(-1)

    objective = checkpoint['objective']

    if objective == 'oracle_bce':
        loss = nn.BCEWithLogitsLoss()(outputs, y_true)

    elif objective == 'pudra':
        pudra_loss = PUDRaNaiveLoss()
        loss = pudra_loss(outputs, y_pu, mode='pu')

    elif objective == 'vpu':
        vpu_loss = VPUNoMixUpLoss()
        loss = vpu_loss(outputs, y_pu, mode='pu')

    elif objective == 'naive':
        # Naive BCE: treat unlabeled (-1) as negative (0)
        y_naive = torch.where(y_pu == 1.0, torch.ones_like(y_pu), torch.zeros_like(y_pu))
        loss = nn.BCEWithLogitsLoss()(outputs, y_naive)

    else:
        raise ValueError(f"Unknown objective: {objective}")

    loss.backward()
    optimizer.step()

    # 4. Return updated checkpoint
    return {
        'task_id': checkpoint['task_id'],
        'task_config': checkpoint['task_config'],
        'objective': checkpoint['objective'],  # Preserve objective
        'model_state': {k: v.cpu().clone() for k, v in model.state_dict().items()},
        'optimizer_state': {
            k: v.cpu().clone() if torch.is_tensor(v) else v
            for k, v in optimizer.state_dict().items()
        },
        'step_count': checkpoint['step_count'] + 1,
        'training_history': checkpoint['training_history'] + [loss.item()],
        'last_updated_iteration': checkpoint.get('last_updated_iteration', -1),
    }


def sample_batch_deterministic(
    task: GaussianBlobTask,
    step_count: int,
    batch_size: int = 64,
    device: str = 'cpu',
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a deterministic batch from task using step_count as seed offset.

    Args:
        task: GaussianBlobTask instance
        step_count: Current training step (used as seed offset)
        batch_size: Number of samples
        device: Device to move tensors to

    Returns:
        Tuple of (x, y_true, y_pu) tensors
    """
    # Use step_count as seed offset for reproducibility
    batch_seed = task.seed + step_count

    # Set random seed for deterministic sampling
    torch.manual_seed(batch_seed)
    np.random.seed(batch_seed)

    # Create dataloader with deterministic seed
    dataloaders = task.get_dataloaders(
        batch_size=batch_size,
        num_train=1000,
        num_val=500,
        num_test=500,
    )

    # Get batch
    batch = next(iter(dataloaders['train']))
    x = batch[0].to(device)
    y_true = batch[1].to(device)
    y_pu = batch[2].to(device)

    return x, y_true, y_pu


def create_model_from_checkpoint(
    checkpoint: Dict[str, Any],
    device: str,
) -> SimpleMLP:
    """Create and load model from checkpoint.

    Args:
        checkpoint: Checkpoint dictionary
        device: Device to load model on

    Returns:
        SimpleMLP with loaded state
    """
    model = SimpleMLP(
        input_dim=checkpoint['task_config']['num_dimensions'],
        hidden_dims=[32, 32],
    ).to(device)

    model.load_state_dict({
        k: v.to(device) for k, v in checkpoint['model_state'].items()
    })

    return model
