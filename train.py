import torch
from omegaconf import DictConfig, OmegaConf
from src.models.lenet import LeNet
from src.utils import load_splits


splits = load_splits("data/flowers/tiny_splits.csv")
config = OmegaConf.load("configs/classification_config.yaml")

model = LeNet(config.model)


torch.save(
    {"model_state_dict": model.state_dict()},
    "lenet.pth",
)


def train(config: DictConfig):
    pass


if __name__ == "__main__":
    config = OmegaConf.load("configs/classification_config.yaml")
