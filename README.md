# Computer vision small models

### The primary goal of this repository is to solve different computer vision problems, such as:

- image classification
- object detection (TBD)
- segmentation (TBD)
- generation (TBD)

User have an opportunity to download pretrained weights from dvc or train his
own model on provided datasets.

## Example: Flowers classification

<img src="images/plant001_rgb.png" alt="Flower" width="224"/>

```bash
poetry install
poetry shell

dvc pull

python infer.py +data=flowers +model=lenet
```
