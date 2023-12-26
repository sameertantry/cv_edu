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
class LenetConfig:
    name: str
    channels: Any
    num_classes: int


@dataclass
class TrainConfig:
    dataloader: Any
    trainer: Any
    data: Any
    model: Any
    logger: Any


@dataclass
class InferenceConfig:
    dataloader: Any
    trainer: Any
    data: Any
    model: Any
    logger: Any


@dataclass
class ExportConfig:
    data: Any
    model: Any
    model_name: str
    model_dir: str
