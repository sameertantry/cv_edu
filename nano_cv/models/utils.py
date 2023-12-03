import lightning as L
from nano_cv.models.classifiers import build_classifier_from_config
from omegaconf import DictConfig


def build_model_from_config(
    config: DictConfig, from_checkpoint: bool = False
) -> L.LightningModule:
    if config.task.name == "clf":
        return build_classifier_from_config(config, from_checkpoint=from_checkpoint)
    else:
        pass
