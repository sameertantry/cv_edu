from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from .flowers_dataset import build_flowers_dataset_from_config


def build_dataset_from_config(config: DictConfig, stage: str) -> Dataset:
    if config.data.name == "flowers":
        return build_flowers_dataset_from_config(config=config, stage=stage)
    else:
        pass


def build_train_dataloader_from_config(config: DictConfig) -> DataLoader:
    dataset = build_dataset_from_config(config=config, stage="train")
    return DataLoader(
        dataset=dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
    )


def build_eval_dataloader_from_config(config: DictConfig) -> DataLoader:
    dataset = build_dataset_from_config(config=config, stage="test")
    return DataLoader(
        dataset=dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
    )
