"""Decoding utilities for CTC model outputs."""

from __future__ import annotations

from typing import Dict, List

from torch import Tensor


def token_to_display_text(token: str) -> str:
    """Convert a namespaced token into a simple display character."""
    if token == "SPACE":
        return " "
    if "_" in token:
        return token.split("_", maxsplit=1)[1]
    return token


def greedy_ctc_decode(
    logits: Tensor,
    id_to_token: Dict[int, str],
    blank_id: int = 0,
) -> List[str]:
    """Greedily decode CTC logits into display strings.

    Args:
        logits: Raw CTC logits [T, B, vocab_size].
        id_to_token: Mapping from class id to token string.
        blank_id: CTC blank token index.

    Returns:
        List of decoded strings. Repeated token ids and blanks are collapsed.

    TODO: Replace simple jamo concatenation with proper jamo composition into
    precomposed Hangul syllables.
    """
    best_paths = logits.argmax(dim=-1).detach().cpu()  # [T, B]
    decoded: List[str] = []

    for batch_index in range(best_paths.size(1)):
        previous_id: int | None = None
        pieces: List[str] = []

        for time_index in range(best_paths.size(0)):
            token_id = int(best_paths[time_index, batch_index])
            if token_id == blank_id:
                previous_id = token_id
                continue
            if token_id == previous_id:
                continue

            token = id_to_token[token_id]
            pieces.append(token_to_display_text(token))
            previous_id = token_id

        decoded.append("".join(pieces))

    return decoded
