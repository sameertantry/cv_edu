import albumentations as A
import cv2
import numpy as np
import torchvision.transforms.functional as TF
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        # self.config = OmegaConf.load("/assets/lenet-transform/config.yaml")
        self.transform = A.Compose(
            [
                A.Resize(
                    height=224,
                    width=224,
                    interpolation=cv2.INTER_LINEAR,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def convert(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        image = TF.to_tensor(image)

        return np.array(image, dtype=np.float32)

    def execute(self, requests):
        responses = []
        for request in requests:
            images = pb_utils.get_input_tensor_by_name(request, "INPUT_IMAGES").as_numpy()

            transformed_images = [self.convert(image) for image in images]
            output_tensor = pb_utils.Tensor("IMAGES", transformed_images)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    output_tensor,
                ]
            )
            responses.append(inference_response)
        return responses
