from typing import Any

import pandas as pd
from app.transformer.neutralization import FeatureNeutralizer
from sklearn.pipeline import Pipeline


class CustomPipeline(Pipeline):
    def __init__(
        self,
        steps: list[Any],
        proportion: float,
        features: list[str],
        neutralize_flag: bool,
    ):
        super().__init__(steps)
        self.proportion = proportion
        self.features = features
        self.neutralize_flag = neutralize_flag

    def neutralize(self, x: pd.DataFrame):
        neutralizer = FeatureNeutralizer(
            proportion=self.proportion, features=self.features
        )
        neutralized_preds = neutralizer.transform(x)
        return neutralized_preds

    def predict(self, x: pd.DataFrame, **kwargs):
        """
        Makes a prediction on the provided dataset and applies neutralization to
        the predictions. This method first utilizes the base model's prediction
        functionality and then adjusts the predictions by applying a neutralization
        process.

        :param x: The input dataset for making predictions. Should be provided
            as a pandas DataFrame.
        :param kwargs: Additional arguments that may be passed to the base
            prediction model.
        :return: The neutralized predictions as a pandas DataFrame.
        """
        # Step 1: Standard ML prediction
        predictions = super().predict(x)

        # Step 2: Apply neutralization
        x["prediction"] = predictions

        if self.neutralize_flag:
            neutralized_predictions = self.neutralize(x)
            return neutralized_predictions

        return predictions


def build_pipeline(
    features: list[str],
    estimator_cls,
    estimator_params: dict,
    proportion: float,
    neutralize_flag: bool,
):
    estimator = estimator_cls(**estimator_params)
    return CustomPipeline(
        [
            ("regressor", estimator),
        ],
        proportion=proportion,
        features=features,
        neutralize_flag=neutralize_flag,
    )
