import json
from logging import Logger
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_data(test_size: float = 0.2, seed=1234, logger: Optional[Logger] = None):

    logger_info = logger.info if logger is not None else print

    with open("data/features.json") as f:
        feature_metadata = json.load(f)

    features = ["era"] + feature_metadata["feature_sets"]["medium"]

    logger_info("Loading data")

    train_df = pd.read_parquet("data/train.parquet", columns=features + ["target"])
    train_df["era"] = train_df["era"].astype(int)
    x = train_df[features]
    y = train_df[["target"]]

    logger_info("Split data into train and test")

    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=test_size, random_state=seed
    )

    return train_x, test_x, train_y, test_y, features
