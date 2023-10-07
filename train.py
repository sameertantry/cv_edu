import torch
from omegaconf import OmegaConf
from src.models.lenet import LeNet
from src.utils import load_splits


splits = load_splits("data/flowers/tiny_splits.csv")
config = OmegaConf.load("configs/classification_config.yaml")

model = LeNet(config.model)

# train loop makes no sense since we have tiny dataset
torch.save(
    {"model_state_dict": model.state_dict()},
    "lenet.pth",
)
