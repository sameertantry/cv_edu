import albumentations as A
import cv2
import torchvision.transforms.functional as TF
import triton_python_backend_utils as pb_utils
from omegaconf import OmegaConf


class TritonPythonModel:
    def initialize(self, args):
        self.config = OmegaConf.load("/asserts/lenet-transform/config.yaml")
        self.transform = A.Compose(
            [
                A.Resize(
                    height=self.config.image_height,
                    width=self.config.image_height,
                    interpolation=cv2.INTER_LINEAR,
                ),
                A.Normalize(mean=self.config.mean, std=self.config.std),
            ]
        )

    def convert(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        image = TF.to_tensor(image)

        return image.numpy()

    def execute(self, requests):
        responses = []
        for request in requests:
            images = pb_utils.get_input_tensor_by_name(request, "IMAGES").as_numpy()

            transformed_images = [self.convert(image) for image in images]
            output_tensor = pb_utils.Tensor("IMAGES", transformed_images)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    output_tensor,
                ]
            )
            responses.append(inference_response)
        return responses
