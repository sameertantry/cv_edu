import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from src.dataset.flowers_dataset import FlowersClassificationDataset
from src.models.lenet import LeNet
from src.utils import load_splits
from torch.utils.data import DataLoader


splits = load_splits("data/flowers/tiny_splits.csv")
config = OmegaConf.load("configs/classification_config.yaml")

test_dataset = FlowersClassificationDataset(
    config.dataset, splits["test"], is_train=False
)
test_loader = DataLoader(test_dataset, batch_size=1)

model = LeNet(config.model)
checkpoint = torch.load("lenet.pth")
model.load_state_dict(checkpoint["model_state_dict"])

loss_fn = nn.CrossEntropyLoss(reduction="sum")
preds = []
loss = 0

with torch.no_grad():
    for batch in test_loader:
        x, labels = batch
        logits = model(x)
        probas = F.softmax(logits, dim=-1)
        preds.append(probas.argmax(dim=-1).item())
        loss += loss_fn(probas.unsqueeze(0), labels)

print(loss / len(test_dataset))

pd.DataFrame({"predictions": preds}).to_csv("predictions.csv")
