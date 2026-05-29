"""Vocabulary definitions for Korean jamo-based HTR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Vocabulary:
    """Container for token-index mappings used by CTC."""

    tokens: List[str]
    token_to_id: Dict[str, int]
    id_to_token: Dict[int, str]
    blank_id: int


def build_jamo_vocabulary() -> Vocabulary:
    """Build an example jamo-based vocabulary for Korean HTR.

    CTC reserves token 0 for blank. For modeling convenience, initial,
    medial, and final jamo are namespaced separately:
        CHO_ㄱ, JUNG_ㅏ, JONG_ㄱ

    This lets the model distinguish the role of the same visual jamo token.
    A later post-processing step can compose these jamo tokens into complete
    Hangul syllables.

    TODO: Implement jamo-to-precomposed-Hangul syllable composition after
    decoding, once the target dataset annotation format is finalized.
    """
    blank = "<blank>"
    specials = ["SPACE"]

    choseong = [
        "ㄱ",
        "ㄲ",
        "ㄴ",
        "ㄷ",
        "ㄸ",
        "ㄹ",
        "ㅁ",
        "ㅂ",
        "ㅃ",
        "ㅅ",
        "ㅆ",
        "ㅇ",
        "ㅈ",
        "ㅉ",
        "ㅊ",
        "ㅋ",
        "ㅌ",
        "ㅍ",
        "ㅎ",
    ]
    jungseong = [
        "ㅏ",
        "ㅐ",
        "ㅑ",
        "ㅒ",
        "ㅓ",
        "ㅔ",
        "ㅕ",
        "ㅖ",
        "ㅗ",
        "ㅘ",
        "ㅙ",
        "ㅚ",
        "ㅛ",
        "ㅜ",
        "ㅝ",
        "ㅞ",
        "ㅟ",
        "ㅠ",
        "ㅡ",
        "ㅢ",
        "ㅣ",
    ]
    jongseong = [
        "ㄱ",
        "ㄲ",
        "ㄳ",
        "ㄴ",
        "ㄵ",
        "ㄶ",
        "ㄷ",
        "ㄹ",
        "ㄺ",
        "ㄻ",
        "ㄼ",
        "ㄽ",
        "ㄾ",
        "ㄿ",
        "ㅀ",
        "ㅁ",
        "ㅂ",
        "ㅄ",
        "ㅅ",
        "ㅆ",
        "ㅇ",
        "ㅈ",
        "ㅊ",
        "ㅋ",
        "ㅌ",
        "ㅍ",
        "ㅎ",
    ]

    tokens = (
        [blank]
        + specials
        + [f"CHO_{token}" for token in choseong]
        + [f"JUNG_{token}" for token in jungseong]
        + [f"JONG_{token}" for token in jongseong]
    )
    token_to_id = {token: index for index, token in enumerate(tokens)}
    id_to_token = {index: token for token, index in token_to_id.items()}
    return Vocabulary(tokens=tokens, token_to_id=token_to_id, id_to_token=id_to_token, blank_id=0)
