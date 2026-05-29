"""Custom CNN backbone for Korean handwritten text recognition."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor, nn


class ConvBNReLU(nn.Module):
    """A small Conv-BatchNorm-ReLU block used by the CNN backbone."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int] = 3,
        stride: int | Tuple[int, int] = 1,
        padding: int | Tuple[int, int] = 1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply Conv-BN-ReLU to a BCHW tensor."""
        return self.block(x)


class KoreanHTRCNNBackbone(nn.Module):
    """Custom CNN backbone for Korean handwritten text recognition.

    The design assumes grayscale images with height 64 by default:
        input:  [B, 1, 64, W]
        output: [B, C, 1, T]

    Width is downsampled only twice by 2x2 pooling, so T is about W / 4.
    Later pooling uses 2x1 kernels to reduce height while preserving the
    left-to-right sequence resolution needed by HTR.
    """

    output_channels: int = 512

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(
            ConvBNReLU(in_channels, 64),
            ConvBNReLU(64, 64),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        self.stage2 = nn.Sequential(
            ConvBNReLU(64, 128),
            ConvBNReLU(128, 128),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        self.stage3 = nn.Sequential(
            ConvBNReLU(128, 256),
            ConvBNReLU(256, 256),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )
        self.stage4 = nn.Sequential(
            ConvBNReLU(256, 384),
            ConvBNReLU(384, 384),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )
        self.stage5 = nn.Sequential(
            ConvBNReLU(384, 512),
            ConvBNReLU(512, 512),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )
        self.stage6 = nn.Sequential(
            # For input height 64, stage5 produces height 2.
            # This convolution collapses height 2 -> 1 and preserves width.
            ConvBNReLU(512, self.output_channels, kernel_size=(2, 3), padding=(0, 1)),
        )

    @staticmethod
    def sequence_lengths(input_widths: Tensor) -> Tensor:
        """Return CNN output sequence lengths for original image widths.

        Only the first two 2x2 pooling layers reduce width, so:
            T = floor(floor(W / 2) / 2) = floor(W / 4)
        """
        return torch.div(input_widths, 4, rounding_mode="floor").clamp_min(1)

    def forward(
        self,
        x: Tensor,
        return_shape_trace: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Tuple[int, ...]]]:
        """Run the CNN and optionally return intermediate tensor shapes.

        Args:
            x: Grayscale input image batch, [B, 1, 64, W].
            return_shape_trace: If true, also return shapes after each stage.

        Returns:
            Feature map [B, C, 1, T], or a pair of feature map and shape trace.
        """
        shapes: Dict[str, Tuple[int, ...]] = {"input": tuple(x.shape)}

        # [B, 1, 64, W] -> [B, 64, 32, W/2]
        x = self.stage1(x)
        shapes["stage1_2x2_pool"] = tuple(x.shape)

        # [B, 64, 32, W/2] -> [B, 128, 16, W/4]
        x = self.stage2(x)
        shapes["stage2_2x2_pool"] = tuple(x.shape)

        # [B, 128, 16, W/4] -> [B, 256, 8, W/4]
        x = self.stage3(x)
        shapes["stage3_2x1_pool"] = tuple(x.shape)

        # [B, 256, 8, W/4] -> [B, 384, 4, W/4]
        x = self.stage4(x)
        shapes["stage4_2x1_pool"] = tuple(x.shape)

        # [B, 384, 4, W/4] -> [B, 512, 2, W/4]
        x = self.stage5(x)
        shapes["stage5_2x1_pool"] = tuple(x.shape)

        # [B, 512, 2, W/4] -> [B, 512, 1, W/4]
        x = self.stage6(x)
        shapes["stage6_height_to_1"] = tuple(x.shape)

        if x.size(2) != 1:
            raise RuntimeError(
                "CNN output height must be 1. This baseline expects input "
                f"height 64, but got final feature shape {tuple(x.shape)}."
            )

        if return_shape_trace:
            return x, shapes
        return x
