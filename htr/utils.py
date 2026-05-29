"""Small utility functions for HTR experiments."""

from __future__ import annotations

from typing import Dict, Tuple

from torch import nn


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return total and trainable parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def print_shape_trace(shapes: Dict[str, Tuple[int, ...]]) -> None:
    """Print a readable CNN shape trace."""
    for name, shape in shapes.items():
        print(f"{name:>20}: {shape}")
