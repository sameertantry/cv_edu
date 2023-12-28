from functools import lru_cache

import albumentations as A
import cv2
import numpy as np
import torchvision.transforms.functional as TF
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
from tritonclient.utils import np_to_triton_dtype


@lru_cache
def get_client():
    return InferenceServerClient(url="0.0.0.0:8500")


def call_triton_classification_ensembele(image: np.ndarray):
    triton_client = get_client()

    input_image = InferInput(
        name="INPUT_IMAGES", shape=image.shape, datatype=np_to_triton_dtype(image.dtype)
    )
    input_image.set_data_from_numpy(image, binary_data=False)

    infer_output = InferRequestedOutput("CLASS_PROBS", binary_data=False)
    query_response = triton_client.infer(
        "ensemble-onnx", [input_image], outputs=[infer_output]
    )
    logits = query_response.as_numpy("CLASS_PROBS")[0]
    return logits


def call_triton_transform(image: np.ndarray):
    triton_client = get_client()

    input_image = InferInput(
        name="INPUT_IMAGES", shape=image.shape, datatype=np_to_triton_dtype(image.dtype)
    )
    input_image.set_data_from_numpy(image, binary_data=False)

    query_response = triton_client.infer(
        "python-transform",
        [input_image],
        outputs=[
            InferRequestedOutput("IMAGES", binary_data=False),
        ],
    )
    image = query_response.as_numpy("IMAGES")[0]
    return image


def main():
    image = cv2.imread("images/plant001_rgb.png")

    transform = A.Compose(
        [
            A.Resize(
                height=224,
                width=224,
                interpolation=cv2.INTER_LINEAR,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    reference_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(reference_image)
    reference_image = TF.to_tensor(reference_image)

    transformed_image = call_triton_transform(image)
    assert np.allclose(reference_image, transformed_image)

    logits = call_triton_classification_ensembele(image)

    print(logits)


if __name__ == "__main__":
    main()
