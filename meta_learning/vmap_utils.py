"""Utility functions for vmap-based meta-training.

This module provides helper functions for parallelizing operations across
multiple model checkpoints using torch.func.vmap.
"""

from typing import List, Dict, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.func import functional_call, vmap


def stack_state_dicts(state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack multiple state_dicts into batched tensors.

    Args:
        state_dicts: List of state_dict OrderedDicts

    Returns:
        Dictionary mapping parameter name -> stacked tensor [num_ckpts, ...shape]

    Raises:
        AssertionError: If state_dicts have different keys or parameter shapes
    """
    if not state_dicts:
        raise ValueError("Cannot stack empty list of state_dicts")

    # Verify all state_dicts have identical keys
    keys = set(state_dicts[0].keys())
    for i, sd in enumerate(state_dicts[1:], 1):
        sd_keys = set(sd.keys())
        if sd_keys != keys:
            missing = keys - sd_keys
            extra = sd_keys - keys
            raise ValueError(
                f"State dict {i} has different keys:\n"
                f"  Missing: {missing}\n"
                f"  Extra: {extra}"
            )

    # Stack each parameter
    stacked = {}
    for key in keys:
        params = [sd[key] for sd in state_dicts]

        # Verify shapes match
        shape = params[0].shape
        for i, p in enumerate(params[1:], 1):
            if p.shape != shape:
                raise ValueError(
                    f"Parameter '{key}' has inconsistent shapes:\n"
                    f"  state_dict[0]: {shape}\n"
                    f"  state_dict[{i}]: {p.shape}"
                )

        # Stack along new dimension 0: [num_ckpts, ...param_shape]
        stacked[key] = torch.stack(params, dim=0)

    return stacked


def unstack_state_dict(
    stacked: Dict[str, torch.Tensor],
    index: int
) -> Dict[str, torch.Tensor]:
    """Extract single state_dict from stacked batch.

    Args:
        stacked: Stacked state_dict from stack_state_dicts
        index: Index of checkpoint to extract

    Returns:
        State dict for checkpoint at index
    """
    unstacked = {}
    for key, stacked_param in stacked.items():
        unstacked[key] = stacked_param[index]

    return unstacked


def check_vmap_compatibility(device: torch.device) -> bool:
    """Test if vmap works correctly on this device.

    Args:
        device: Device to test

    Returns:
        True if vmap is compatible, False otherwise
    """
    try:
        # Create simple test function
        def test_fn(x):
            return x * 2

        # Test vmapping
        x = torch.randn(5, 3, device=device)
        result = vmap(test_fn)(x)

        # Verify shape
        expected_shape = (5, 3)
        if result.shape != expected_shape:
            return False

        # Verify values
        expected = x * 2
        if not torch.allclose(result, expected):
            return False

        return True

    except Exception as e:
        print(f"Vmap compatibility test failed: {e}")
        return False


def safe_vmap(
    func,
    in_dims: Tuple,
    *args,
    device: torch.device,
    fallback_sequential: bool = True
):
    """Vmap with fallback to sequential processing on failure.

    Args:
        func: Function to vmap
        in_dims: Input dimensions for vmap
        *args: Arguments to pass to vmapped function
        device: Device to use
        fallback_sequential: If True, fall back to sequential on error

    Returns:
        Vmapped results or sequential results if vmap fails
    """
    try:
        # Try vmap
        vmapped_func = vmap(func, in_dims=in_dims)
        result = vmapped_func(*args)
        return result

    except Exception as e:
        if not fallback_sequential:
            raise

        print(f"Warning: Vmap failed ({e}), falling back to sequential processing")

        # Fall back to sequential
        # Assume first dimension is batch
        batch_size = args[0].shape[0] if len(args) > 0 and hasattr(args[0], 'shape') else 1

        results = []
        for i in range(batch_size):
            # Extract elements at index i
            args_i = []
            for arg, dim in zip(args, in_dims):
                if dim == 0:
                    args_i.append(arg[i])
                else:
                    args_i.append(arg)

            result_i = func(*args_i)
            results.append(result_i)

        # Stack results
        return torch.stack(results, dim=0)


class VmapContext:
    """Context manager for vmap operations with error handling.

    Usage:
        with VmapContext(device='mps') as ctx:
            if ctx.vmap_available:
                result = vmap(func)(args)
            else:
                result = sequential_fallback(args)
    """

    def __init__(self, device: torch.device):
        """Initialize context.

        Args:
            device: Device to check vmap compatibility
        """
        self.device = device
        self.vmap_available = False

    def __enter__(self):
        """Enter context and check vmap compatibility."""
        self.vmap_available = check_vmap_compatibility(self.device)
        if not self.vmap_available:
            print(f"Warning: Vmap not available on {self.device}, will use sequential processing")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        return False  # Don't suppress exceptions


def get_device() -> torch.device:
    """Get device with MPS priority (MPS -> CUDA -> CPU).

    Returns:
        torch.device: Best available device
    """
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def prepare_batched_data(
    checkpoints: List[Dict],
    device: torch.device,
    batch_key_train: str = 'train_loader',
    batch_key_val: str = 'val_loader'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample and stack data batches from checkpoint loaders.

    Args:
        checkpoints: List of checkpoint dicts with DataLoaders
        device: Device to move tensors to
        batch_key_train: Key for training DataLoader in checkpoint dict
        batch_key_val: Key for validation DataLoader in checkpoint dict

    Returns:
        Tuple of (x_train_batched, t_train_batched, x_val_batched, y_val_batched)
        Each with shape [num_ckpts, batch_size, ...]
    """
    x_train_list, t_train_list = [], []
    x_val_list, y_val_list = [], []

    for cp in checkpoints:
        # Sample batch from train loader (PU data)
        train_loader = cp.get(batch_key_train)
        if train_loader is None:
            raise ValueError(f"Checkpoint missing '{batch_key_train}' key")

        x_train, t_train = next(iter(train_loader))
        x_train_list.append(x_train)
        t_train_list.append(t_train)

        # Sample batch from val loader (PN data)
        val_loader = cp.get(batch_key_val)
        if val_loader is None:
            raise ValueError(f"Checkpoint missing '{batch_key_val}' key")

        x_val, y_val = next(iter(val_loader))
        x_val_list.append(x_val)
        y_val_list.append(y_val)

    # Stack into [num_ckpts, batch_size, ...feature_shape]
    x_train_batched = torch.stack(x_train_list).to(device)
    t_train_batched = torch.stack(t_train_list).to(device)
    x_val_batched = torch.stack(x_val_list).to(device)
    y_val_batched = torch.stack(y_val_list).to(device)

    return x_train_batched, t_train_batched, x_val_batched, y_val_batched


def vmapped_forward_pass(
    model: nn.Module,
    stacked_params: Dict[str, torch.Tensor],
    inputs: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """Parallel forward passes using vmap.

    Args:
        model: Model template (architecture only, params will be replaced)
        stacked_params: Stacked parameters [num_ckpts, ...param_shape]
        inputs: Batched inputs [num_ckpts, batch_size, ...]
        device: Device to use

    Returns:
        Batched outputs [num_ckpts, batch_size, ...]
    """
    def single_forward(params, x):
        """Single forward pass with given params."""
        return functional_call(model, params, (x,))

    # Vmap across checkpoints (dim 0)
    outputs = vmap(single_forward, in_dims=(0, 0))(stacked_params, inputs)

    return outputs
