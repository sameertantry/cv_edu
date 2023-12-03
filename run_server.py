import albumentations as A
import cv2
import hydra
import mlflow.pyfunc
import onnx
import torch
import torchvision.transforms.functional as TF
from hydra.core.config_store import ConfigStore
from mlflow.models import infer_signature
from nano_cv.models.utils import build_model_from_config
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
def main(cfg: InferenceConfig):
    model = build_model_from_config(cfg, from_checkpoint=True)
    x = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(model, x, "model.onnx")
    onnx_model = onnx.load_model("model.onnx")

    with mlflow.start_run():
        mlflow.set_tracking_uri(cfg.logger.tracking_uri)
        signature = infer_signature(x.numpy(), model(x).detach().numpy())
        model_info = mlflow.onnx.log_model(onnx_model, "model", signature=signature)

    onnx_pyfunc = mlflow.pyfunc.load_model(model_info.model_uri)

    transform = A.Compose(
        [
            A.Resize(
                height=cfg.data.image_height,
                width=cfg.data.image_width,
                interpolation=cv2.INTER_LINEAR,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    image = cv2.imread("images/plant001_rgb.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image=image)["image"]
    image = TF.to_tensor(image)

    preds = onnx_pyfunc.predict(image.unsqueeze(0).numpy())
    print(preds)


if __name__ == "__main__":
    main()
