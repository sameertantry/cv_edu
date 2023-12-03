from typing import Any

import albumentations as A
import cv2
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from omegaconf import DictConfig
from torch.utils.data import Dataset


class FlowersDatasetBase(Dataset):
    def __init__(self, config: DictConfig, annotations_path: str, stage: str):
        super().__init__()
        self.config = config
        self.annotations = pd.read_csv(annotations_path)
        self.stage = stage
        self.is_train = stage == "train"
        self.transform = self._get_transform()

    def __len__(self):
        return len(self.annotations)

    def _get_train_transform(self):
        raise NotImplementedError("Implement '_get_train_transform' method")

    def _get_validation_transform(self):
        raise NotImplementedError("Implement '_get_validation_transform' method")

    def _get_transform(self):
        if self.is_train:
            return self._get_train_transform()
        else:
            return self._get_validation_transform()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Any]:
        raise NotImplementedError("Implement '__getitem__' method")


class FlowersClassificationDataset(FlowersDatasetBase):
    def _get_train_transform(self):
        return A.Compose(
            [
                A.Affine(p=0.1, interpolation=cv2.INTER_LINEAR),
                A.RandomResizedCrop(
                    scale=(self.config.crop_scale, 1),
                    height=self.config.image_height,
                    width=self.config.image_height,
                    p=1,
                    interpolation=cv2.INTER_LINEAR,
                ),
                A.Flip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Downscale(p=0.5, interpolation=cv2.INTER_LINEAR),
                A.GaussianBlur(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def _get_validation_transform(self):
        return A.Compose(
            [
                A.Resize(
                    height=self.config.image_height,
                    width=self.config.image_width,
                    interpolation=cv2.INTER_LINEAR,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Any]:
        image_path = (
            self.config.path + self.stage + "/" + self.annotations.iloc[idx]["img_path"]
        )

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"]
        image = TF.to_tensor(image)

        label = (
            int(self.annotations.iloc[idx]["target"][-1]) - 1
        )  # data/flowers/Ax/... -> x - 1

        return image, label


def build_flowers_dataset_from_config(config: DictConfig, stage: str) -> Dataset:
    annotations_path = config.data.path + stage + "/annotations.csv"
    if config.task.name == "clf":
        return FlowersClassificationDataset(config.data, annotations_path, stage=stage)
    else:
        pass
