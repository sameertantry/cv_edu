import hydra
import lightning as L
from cv_models.dataset.utils import build_train_dataloader_from_config
from cv_models.models.utils import build_model_from_config
from cv_models.tools.configs import (
    ClassificationConfig,
    FlowersDataset,
    Lenet,
    TrainConfig,
)
from hydra.core.config_store import ConfigStore


cs = ConfigStore.instance()

cs.store(group="data", name="base_flowers", node=FlowersDataset)

cs.store(group="task", name="base_clf", node=ClassificationConfig)

cs.store(group="model", name="base_lenet", node=Lenet)


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def main(cfg: TrainConfig) -> None:
    model = build_model_from_config(cfg)
    dataloader = build_train_dataloader_from_config(cfg)
    trainer = L.Trainer(**cfg.trainer)
    trainer.fit(model, train_dataloaders=dataloader)
    trainer.save_checkpoint(cfg.checkpoint_path)


if __name__ == "__main__":
    main()
