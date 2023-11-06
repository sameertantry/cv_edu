from pathlib import Path

import pandas as pd


def load_splits(path: Path | str) -> dict[str, pd.DataFrame]:
    dataset = pd.read_csv(path)

    splits = {
        "train": dataset.loc[dataset["split"] == "train"],
        "validation": dataset.loc[dataset["split"] == "dev"],
        "test": dataset.loc[dataset["split"] == "test"],
    }

    return splits
