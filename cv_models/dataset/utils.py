import os

from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from .flowers_dataset import build_flowers_dataset_from_config


def build_dataset_from_config(config: DictConfig, stage: str) -> Dataset:
    if config.data.name == "flowers":
        os.system("dvc pull data/flowers.dvc")
        return build_flowers_dataset_from_config(config=config, stage=stage)
    else:
        pass


def build_train_dataloader_from_config(config: DictConfig) -> DataLoader:
    dataset = build_dataset_from_config(config=config, stage="train")
    return DataLoader(
        dataset=dataset,
        shuffle=True,
        **config.dataloader,
    )


def build_eval_dataloader_from_config(
    config: DictConfig, stage: str = "test"
) -> DataLoader:
    dataset = build_dataset_from_config(config=config, stage=stage)
    return DataLoader(
        dataset=dataset,
        shuffle=False,
        **config.dataloader,
    )
