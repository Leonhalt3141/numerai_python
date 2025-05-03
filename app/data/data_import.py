import json

import pandas as pd


def read_features() -> dict:
    with open("data/features.json") as f:
        feature_metadata = json.load(f)
    return feature_metadata


def open_train_data() -> pd.DataFrame:
    train_df = pd.read_parquet("data/train.parquet")
    return train_df
