import lightning as L
from cv_models.models.classifiers import build_classifier_from_config
from omegaconf import DictConfig


def build_model_from_config(config: DictConfig) -> L.LightningModule:
    if config.task.name == "clf":
        return build_classifier_from_config(config)
    else:
        pass
