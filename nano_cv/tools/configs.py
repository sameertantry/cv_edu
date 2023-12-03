from dataclasses import dataclass
from typing import Any


@dataclass
class FlowersDataset:
    name: str
    path: str
    dvc_path: str
    image_height: int
    image_width: int
    crop_scale: float


@dataclass
class ClassificationConfig:
    name: str
    num_classes: int
    allowed_models: Any


@dataclass
class Lenet:
    name: str
    channels: Any


@dataclass
class TrainConfig:
    dataloader: Any
    trainer: Any
    data: Any
    task: Any
    model: Any


@dataclass
class InferenceConfig:
    dataloader: Any
    trainer: Any
    data: Any
    task: Any
    model: Any
