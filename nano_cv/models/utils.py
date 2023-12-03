import lightning as L
import torch
from cv_models.models.classifiers import build_classifier_from_config
from omegaconf import DictConfig
from torch.utils.data import DataLoader


def build_model_from_config(
    config: DictConfig, from_checkpoint: bool = False
) -> L.LightningModule:
    if config.task.name == "clf":
        return build_classifier_from_config(config, from_checkpoint=from_checkpoint)
    else:
        pass


@torch.no_grad()
def collect_predictions(model: L.LightningModule, dataloader: DataLoader):
    model.eval()
    predictions = []
    for x, _ in dataloader:
        yhat = model(x)
        predictions += [x.item() for x in yhat.argmax(-1)]

    return predictions
