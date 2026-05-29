"""CRNN model for Korean handwritten text recognition."""

from __future__ import annotations

from typing import Tuple

from torch import Tensor, nn

from htr.models.backbone import KoreanHTRCNNBackbone


class CRNNHTRModel(nn.Module):
    """CRNN model for handwritten word or text-line recognition.

    Pipeline:
        grayscale image [B, 1, 64, W]
        -> custom CNN [B, C, 1, T]
        -> width-wise sequence [T, B, C]
        -> BiLSTM [T, B, 2H]
        -> Linear classifier [T, B, vocab_size]

    The output logits are intended for nn.CTCLoss.
    """

    def __init__(
        self,
        vocab_size: int,
        in_channels: int = 1,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cnn = KoreanHTRCNNBackbone(in_channels=in_channels)
        self.sequence_model = nn.LSTM(
            input_size=self.cnn.output_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=False,
        )
        self.classifier = nn.Linear(lstm_hidden_size * 2, vocab_size)

    def sequence_lengths(self, input_widths: Tensor) -> Tensor:
        """Return output time lengths for given original image widths."""
        return self.cnn.sequence_lengths(input_widths)

    def forward(
        self,
        images: Tensor,
        return_features: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        """Compute CTC logits from input images.

        Args:
            images: Input tensor [B, 1, 64, W].
            return_features: If true, also return CNN features [B, C, 1, T].

        Returns:
            logits: [T, B, vocab_size], suitable for nn.CTCLoss after log_softmax.
        """
        # CNN feature map: [B, 1, 64, W] -> [B, C, 1, T]
        features = self.cnn(images)

        # Remove height dimension: [B, C, 1, T] -> [B, C, T]
        sequence = features.squeeze(2)

        # Read feature columns as time steps: [B, C, T] -> [T, B, C]
        sequence = sequence.permute(2, 0, 1).contiguous()

        # BiLSTM sequence modeling: [T, B, C] -> [T, B, 2 * hidden_size]
        recurrent_output, _ = self.sequence_model(sequence)

        # Per-time-step classifier: [T, B, 2H] -> [T, B, vocab_size]
        logits = self.classifier(recurrent_output)

        if return_features:
            return logits, features
        return logits
