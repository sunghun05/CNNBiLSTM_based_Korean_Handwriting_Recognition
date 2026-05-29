"""CTC helpers for sequence targets."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
from torch import Tensor


def encode_token_sequences(
    token_sequences: Sequence[Sequence[str]],
    token_to_id: Dict[str, int],
) -> Tuple[Tensor, Tensor]:
    """Encode a batch of token sequences for nn.CTCLoss.

    nn.CTCLoss expects a flattened 1D target tensor and a target_lengths tensor:
        targets:        [sum(target_lengths)]
        target_lengths: [B]
    """
    target_lengths = torch.tensor([len(sequence) for sequence in token_sequences], dtype=torch.long)
    flat_targets = [
        token_to_id[token]
        for sequence in token_sequences
        for token in sequence
    ]
    targets = torch.tensor(flat_targets, dtype=torch.long)
    return targets, target_lengths
