"""MIT implementations compatible with two externally distributed checkpoints.

These architectures were implemented from the published model descriptions and the
checkpoint tensor schemas. No source code from either upstream implementation is
included in AttackBench.
"""

import math

import torch
from torch import nn


class _CCATResidualBlock(nn.Module):
    """Post-activation CIFAR residual block used by the Stutz 2020 checkpoint."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs if self.downsample is None else self.downsample(inputs)
        outputs = self.relu(self.norm1(self.conv1(inputs)))
        outputs = self.norm2(self.conv2(outputs))
        return self.relu(outputs + residual)


class Stutz2020CCAT(nn.Module):
    """CIFAR-10 ResNet-20 compatible with the published CCAT checkpoint."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.block0 = self._stage(64, 64, blocks=3, stride=1)
        self.block1 = self._stage(64, 128, blocks=3, stride=2)
        self.block2 = self._stage(128, 256, blocks=3, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.logits = nn.Linear(256, 10)

    @staticmethod
    def _stage(
        in_channels: int, out_channels: int, blocks: int, stride: int
    ) -> nn.Sequential:
        layers = [_CCATResidualBlock(in_channels, out_channels, stride)]
        layers.extend(
            _CCATResidualBlock(out_channels, out_channels) for _ in range(blocks - 1)
        )
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # The released model was trained and evaluated on image values in [0, 1].
        outputs = inputs.clamp(0, 1)
        outputs = self.relu1(self.norm1(self.conv1(outputs)))
        outputs = self.block0(outputs)
        outputs = self.block1(outputs)
        outputs = self.block2(outputs)
        outputs = self.avgpool(outputs)
        return self.logits(torch.flatten(outputs, 1))


class KWTA2d(nn.Module):
    """Keep the largest fraction of each sample's complete feature volume.

    This is equation (1) of Xiao, Zhong and Zheng (2019), applied to the flattened
    C * H * W feature volume as specified in section 2.1 of the paper.
    """

    def __init__(self, fraction: float):
        super().__init__()
        if not 0 < fraction <= 1:
            raise ValueError("k-WTA fraction must be in (0, 1]")
        self.fraction = fraction

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        flattened = torch.flatten(inputs, 1)
        winners = max(1, math.floor(self.fraction * flattened.shape[1]))
        values, indices = torch.topk(flattened, winners, dim=1, sorted=False)
        sparse = torch.zeros_like(flattened).scatter(1, indices, values)
        return sparse.reshape_as(inputs)


class _KWTABasicBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stride: int, fraction: float
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.sparse1 = KWTA2d(fraction)
        self.sparse2 = KWTA2d(fraction)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.sparse1(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs)) + self.shortcut(inputs)
        return self.sparse2(outputs)


class Xiao2020KWTA(nn.Module):
    """CIFAR-10 ResNet-18 using volume-wise 10% k-WTA activations."""

    def __init__(self, fraction: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.layer1 = self._stage(64, 64, blocks=2, stride=1, fraction=fraction)
        self.layer2 = self._stage(64, 128, blocks=2, stride=2, fraction=fraction)
        self.layer3 = self._stage(128, 256, blocks=2, stride=2, fraction=fraction)
        self.layer4 = self._stage(256, 512, blocks=2, stride=2, fraction=fraction)
        self.linear = nn.Linear(512, 10)

    @staticmethod
    def _stage(
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int,
        fraction: float,
    ) -> nn.Sequential:
        layers = [_KWTABasicBlock(in_channels, out_channels, stride, fraction)]
        layers.extend(
            _KWTABasicBlock(out_channels, out_channels, 1, fraction)
            for _ in range(blocks - 1)
        )
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)
        outputs = torch.nn.functional.avg_pool2d(outputs, 4)
        return self.linear(torch.flatten(outputs, 1))
