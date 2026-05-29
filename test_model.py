"""Runnable CRNN HTR smoke test.

The model, vocabulary, decoding, and utility code live under the htr package.
Run this file to verify tensor shapes, CTC loss usage, and greedy decoding:

    python htr_crnn.py
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from htr import (
    CRNNHTRModel,
    build_jamo_vocabulary,
    count_parameters,
    encode_token_sequences,
    greedy_ctc_decode,
    print_shape_trace,
)


def run_dummy_test() -> None:
    """Run a small end-to-end smoke test with CTC loss and greedy decoding."""
    torch.manual_seed(7)

    vocabulary = build_jamo_vocabulary()
    model = CRNNHTRModel(vocab_size=len(vocabulary.tokens))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    batch_size = 2
    image_height = 64
    padded_width = 256
    original_widths = torch.tensor([256, 224], dtype=torch.long)

    images = torch.randn(batch_size, 1, image_height, padded_width, device=device)

    print("Vocabulary size:", len(vocabulary.tokens))
    print("Blank id:", vocabulary.blank_id)
    total_params, trainable_params = count_parameters(model)
    print(f"Parameters: total={total_params:,}, trainable={trainable_params:,}")

    with torch.no_grad():
        cnn_features, shape_trace = model.cnn(images, return_shape_trace=True)
        logits, returned_features = model(images, return_features=True)

    print("\nCNN shape trace:")
    print_shape_trace(shape_trace)
    print("\nModel outputs:")
    print("cnn_features:", tuple(cnn_features.shape))       # [B, C, 1, T]
    print("returned_features:", tuple(returned_features.shape))
    print("logits:", tuple(logits.shape))                   # [T, B, vocab_size]

    input_lengths = model.sequence_lengths(original_widths).to(device)
    print("original_widths:", original_widths.tolist())
    print("ctc_input_lengths:", input_lengths.detach().cpu().tolist())

    target_sequences = [
        ["CHO_ㅎ", "JUNG_ㅏ", "JONG_ㄴ", "CHO_ㄱ", "JUNG_ㅡ", "JONG_ㄹ"],
        [
            "CHO_ㅅ",
            "JUNG_ㅗ",
            "JONG_ㄴ",
            "SPACE",
            "CHO_ㄱ",
            "JUNG_ㅡ",
            "JONG_ㄹ",
            "CHO_ㅆ",
            "JUNG_ㅣ",
        ],
    ]
    targets, target_lengths = encode_token_sequences(target_sequences, vocabulary.token_to_id)
    targets = targets.to(device)
    target_lengths = target_lengths.to(device)

    print("target_lengths:", target_lengths.detach().cpu().tolist())
    if torch.any(target_lengths > input_lengths):
        raise ValueError("Every target length must be <= its CTC input length.")

    ctc_loss = nn.CTCLoss(blank=vocabulary.blank_id, reduction="mean", zero_infinity=True)
    log_probs = F.log_softmax(logits, dim=-1)  # [T, B, vocab_size]
    loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
    print("dummy_ctc_loss:", float(loss.detach().cpu()))

    decoded = greedy_ctc_decode(logits, vocabulary.id_to_token, blank_id=vocabulary.blank_id)
    print("greedy_decoded:", decoded)


if __name__ == "__main__":
    run_dummy_test()
