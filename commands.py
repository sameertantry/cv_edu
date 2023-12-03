from pathlib import Path

import fire
from omegaconf import OmegaConf


class Experiment:
    def __init__(self, type: str, config_path: Path):
        self.type = type
        self.config = OmegaConf.load(config_path)

    def train(self, dataset_path: Path):
        pass

    def infer(self, dataset_path: Path):
        pass


if __name__ == "__main__":
    experiment = Experiment()
    fire.Fire(Experiment)
