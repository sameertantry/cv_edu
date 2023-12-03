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
class TrainParams:
    batch_size: int
    max_epochs: int
    optimizer: str
    lr: float
    weight_decay: float
    num_workers: int


@dataclass
class TrainConfig:
    params: TrainParams
    data: Any
    task: Any
    model: Any


@dataclass
class InferConfig:
    pass
