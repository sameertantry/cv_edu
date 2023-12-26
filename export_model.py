import hydra
import numpy as np
import onnxruntime as ort
import torch
from hydra.core.config_store import ConfigStore
from nano_cv.models.utils import build_model_from_config
from nano_cv.tools.configs import ExportConfig, FlowersDataset, LenetConfig


cs = ConfigStore.instance()

cs.store(group="data", name="base_flowers", node=FlowersDataset)
cs.store(group="model", name="base_lenet", node=LenetConfig)


@hydra.main(config_path="configs", config_name="export", version_base="1.3")
def main(cfg: ExportConfig) -> None:
    model_path = cfg.model_dir + cfg.model_name

    model = build_model_from_config(cfg, checkpoint_path=model_path + ".pth")
    model.eval()

    dummy_input = torch.randn(1, 3, cfg.data.image_height, cfg.data.image_width)
    torch.onnx.export(
        model,
        dummy_input,
        model_path + ".onnx",
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=["IMAGES"],
        output_names=["CLASS_PROBS"],
        dynamic_axes={"IMAGES": {0: "BATCH_SIZE"}, "CLASS_PROBS": {0: "BATCH_SIZE"}},
    )

    reference_output = model(dummy_input).detach().numpy()
    ort_inputs = {"IMAGES": dummy_input.numpy()}
    ort_session = ort.InferenceSession(model_path + ".onnx")
    ort_output = ort_session.run(None, ort_inputs)[0]

    assert np.allclose(reference_output, ort_output)


if __name__ == "__main__":
    main()
