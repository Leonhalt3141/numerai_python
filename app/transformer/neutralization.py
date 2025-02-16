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


class FeatureNeutralizer(BaseEstimator, TransformerMixin):
    """
    Transformer for feature neutralization in scikit-learn pipelines.
    """

    def __init__(self, features: list[str], proportion=1.0):
        self.features = features
        self.proportion = proportion

    def neutralize(self, predictions, features):
        """
        Apply feature neutralization to predictions.
        """
        scores = predictions.reshape(-1, 1)
        exposures = np.linalg.pinv(features.T @ features) @ (features.T @ scores)
        neutralized = scores - self.proportion * (features @ exposures)
        return neutralized.flatten()

    def fit(self, x, y=None):
        # No fitting needed, returns self
        return self

    def transform(self, x):
        """
        x is expected to be a DataFrame where:
        - The last column is the predictions (from a model).
        - The remaining columns are the features.
        """
        neutralized_preds = neutralize_function(
            x, "prediction", self.features, self.proportion
        )

        # 予測値を更新して DataFrame に戻す
        x.iloc[:, -1] = neutralized_preds
        return x
