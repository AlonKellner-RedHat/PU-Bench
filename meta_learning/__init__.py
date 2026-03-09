"""Meta-learning module for checkpoint-based loss optimization.

This module implements multi-task checkpoint-based meta-learning for the
MonotonicBasisLoss. Key components:

- CheckpointPool: Manages checkpoint pool with curriculum refresh strategy
- MetaTrainer: Implements "models as data" meta-learning paradigm
- Utilities: Task profiling, split creation, pool initialization

The approach treats pre-trained model checkpoints as data samples, enabling
memory-efficient meta-learning without higher-order gradients.
"""

from .checkpoint_pool import CheckpointPool, load_task_split, format_task_id
from .meta_trainer import MetaTrainer

__all__ = [
    'CheckpointPool',
    'MetaTrainer',
    'load_task_split',
    'format_task_id'
]
