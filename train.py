import hydra
import lightning as L
from hydra.core.config_store import ConfigStore
from lightning.pytorch.loggers import MLFlowLogger
from nano_cv.dataset.utils import build_train_dataloader_from_config
from nano_cv.models.utils import build_model_from_config
from nano_cv.tools.configs import ClassificationConfig, FlowersDataset, Lenet, TrainConfig
from omegaconf import OmegaConf


cs = ConfigStore.instance()

cs.store(group="data", name="base_flowers", node=FlowersDataset)

cs.store(group="task", name="base_clf", node=ClassificationConfig)

cs.store(group="model", name="base_lenet", node=Lenet)


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def main(cfg: TrainConfig) -> None:
    model = build_model_from_config(cfg)
    dataloader = build_train_dataloader_from_config(cfg)

    logger = MLFlowLogger(**cfg.logger)
    logger.log_hyperparams(OmegaConf.to_container(cfg))
    trainer = L.Trainer(
        **cfg.trainer,
        logger=[
            logger,
        ],
    )
    trainer.fit(model, train_dataloaders=dataloader)
    trainer.save_checkpoint(cfg.checkpoint_path)


if __name__ == "__main__":
    main()
