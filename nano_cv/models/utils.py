import lightning as L
from nano_cv.models.classifiers import build_classifier_from_config
from omegaconf import DictConfig


def build_model_from_config(
    config: DictConfig, checkpoint_path: str = None
) -> L.LightningModule:
    if config.model.name in ("lenet",):
        return build_classifier_from_config(config, checkpoint_path=checkpoint_path)
    else:
        pass
