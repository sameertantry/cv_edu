import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


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


class Lenet(L.LightningModule):
    def __init__(self, config: DictConfig, num_classes: int):
        super().__init__()

        self.config = config.model

        self.conv = self._build_conv_layers()
        self.loss = nn.CrossEntropyLoss()

        n_h = config.data.image_height // 2 ** (len(self.config.channels) - 1)
        n_w = config.data.image_width // 2 ** (len(self.config.channels) - 1)
        self.flatten = nn.Flatten()
        self.classification = nn.Linear(
            in_features=self.config.channels[-1] * n_h * n_w,
            out_features=num_classes,
        )

    def forward(self, x) -> torch.Tensor:
        y = self.conv(x)
        z = self.flatten(y)
        logits = self.classification(z)

        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss

    def _build_conv_layers(self) -> nn.Sequential:
        layers = []
        for i in range(len(self.config.channels) - 1):
            layers.append(
                ConvBlock(
                    in_channels=self.config.channels[i],
                    out_channels=self.config.channels[i + 1],
                )
            )

        return nn.Sequential(*layers)

    def configure_optimizers(self):
        if self.config.optimizer.name == "adam":
            return torch.optim.Adam(self.parameters(), **self.config.optimizer.params)
        else:
            pass


def build_classifier_from_config(config: DictConfig) -> L.LightningModule:
    if config.model.name not in config.task.allowed_models:
        raise ValueError(
            f"Model {config.model.name} is not allowed. Expected: \
                {config.task.allowed_models}"
        )
    if config.model.name == "lenet":
        return Lenet(config=config, num_classes=config.task.num_classes)
    else:
        pass
