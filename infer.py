import hydra
import lightning as L
import pandas as pd
from hydra.core.config_store import ConfigStore
from nano_cv.dataset.utils import build_eval_dataloader_from_config
from nano_cv.models.utils import build_model_from_config, predict
from nano_cv.tools.configs import (
    ClassificationConfig,
    FlowersDataset,
    InferenceConfig,
    Lenet,
)


cs = ConfigStore.instance()

cs.store(group="data", name="base_flowers", node=FlowersDataset)

cs.store(group="task", name="base_clf", node=ClassificationConfig)

cs.store(group="model", name="base_lenet", node=Lenet)


@hydra.main(config_path="configs", config_name="infer", version_base="1.3")
def main(cfg: InferenceConfig) -> None:
    model = build_model_from_config(cfg, from_checkpoint=True)
    dataloader = build_eval_dataloader_from_config(cfg)
    trainer = L.Trainer(**cfg.trainer)
    trainer.test(model, dataloaders=dataloader)
    if cfg.task.name == "clf":
        preds = predict(model, dataloader)
        preds = pd.DataFrame(preds, columns=["target"])
        preds.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()
