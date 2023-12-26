import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torchmetrics import F1Score, Precision, Recall


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
    def __init__(self, config: DictConfig):
        super().__init__()

        self.config = config.model

        self.conv = self._build_conv_layers()
        self.loss = nn.CrossEntropyLoss()
        self.metrics = {
            "f1": F1Score(task="multiclass", num_classes=self.config.num_classes),
            "recall": Recall(task="multiclass", num_classes=self.config.num_classes),
            "precision": Precision(
                task="multiclass", num_classes=self.config.num_classes
            ),
        }

        n_h = config.data.image_height // 2 ** (len(self.config.channels) - 1)
        n_w = config.data.image_width // 2 ** (len(self.config.channels) - 1)
        self.flatten = nn.Flatten()
        self.classification = nn.Linear(
            in_features=self.config.channels[-1] * n_h * n_w,
            out_features=self.config.num_classes,
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

    def _eval_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        metrics = {"loss": loss}
        for metric_name, metric in self.metrics.items():
            metrics[metric_name] = metric(y_hat, y)

        return metrics, y_hat

    def validation_step(self, batch, batch_idx):
        metrics, _ = self._eval_step(batch)
        self.log_dict({"val_" + name: value for name, value in metrics.items()})

    def test_step(self, batch, batch_idx):
        metrics, yhat = self._eval_step(batch)
        self.log_dict({"test_" + name: value for name, value in metrics.items()})

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        return self(x).argmax(-1)

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


def build_classifier_from_config(
    config: DictConfig, checkpoint_path: str
) -> L.LightningModule:
    if config.model.name == "lenet":
        if checkpoint_path is None:
            return Lenet(config=config)
        else:
            return Lenet.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                config=config,
            )
    else:
        pass
