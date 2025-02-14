import numpy as np
import pandas as pd
from numerai_tools.scoring import neutralize
from sklearn.base import BaseEstimator, TransformerMixin


def neutralize_function(
    df: pd.DataFrame, prediction_col: str, features: list[str], proportion: float
) -> pd.DataFrame:
    neutralized = (
        neutralize(df[[prediction_col]], df[features], proportion=proportion)
        .reset_index()
        .set_index("id")
    )
    return neutralized


class CustomNeutralizer(BaseEstimator, TransformerMixin):
    def __init__(self, prediction_col: str, features: list[str], proportion: float):
        self.prediction_col = prediction_col
        self.features = features
        self.proportion = proportion

    def fit(self, x, y=None):
        return self

    def transform(self, x: pd.DataFrame):
        return x.groupby("era", group_keys=True).apply(
            lambda d: neutralize_function(
                d, self.prediction_col, self.features, self.proportion
            )
        )
