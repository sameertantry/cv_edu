import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.pooling = nn.AvgPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = F.leaky_relu(y)

        z = self.pooling(y)
        z = self.norm(z)

        return z


class LeNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.conv = self._build_conv_layers_from_config()
        self.classification = nn.Linear(
            in_features=self.config.channels[-1] * 7 * 7,
            out_features=self.config.num_classes,
        )

    def forward(self, x) -> torch.Tensor:
        y = self.conv(x)
        z = torch.flatten(y)
        logits = self.classification(z)

        return logits

    def _build_conv_layers_from_config(self) -> nn.Sequential:
        layers = []
        prev_n = 3
        for n in self.config.channels:
            layers.append(ConvBlock(in_channels=prev_n, out_channels=n))
            prev_n = n

        return nn.Sequential(*layers)
