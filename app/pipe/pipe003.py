from typing import Any

import pandas as pd
from app.transformer.neutralization import FeatureNeutralizer
from sklearn.pipeline import Pipeline


class SubModelPipeline(Pipeline):
    def __init__(
        self,
        steps: list[tuple[str, Any]],
        proportion: float,
        neutralize_flag: bool,
        features: list[str],
    ):
        super().__init__(steps)
        self.steps = steps
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
        # Step 1: Standard ML prediction
        predictions = super().predict(x)

        # Step 2: Apply neutralization

        if self.neutralize_flag:
            x["prediction"] = predictions
            neutralized_predictions = self.neutralize(x)
            return neutralized_predictions

        x["prediction"] = predictions
        return x


class MetaModelPipeline(Pipeline):
    def __init__(
        self,
        steps: list[tuple[str, Any]],
        proportion: float,
        neutralize_flag: bool,
        features: list[str],
    ):
        super().__init__(steps)
        self.steps = steps
        self.proportion = proportion
        self.neutralize_flag = neutralize_flag
        self.features = features

    def neutralize(self, x: pd.DataFrame):
        neutralizer = FeatureNeutralizer(
            proportion=self.proportion, features=self.features
        )
        neutralized_preds = neutralizer.transform(x)
        return neutralized_preds

    def predict(self, x: pd.DataFrame, **kwargs):
        # Step 1: Standard ML prediction
        predictions = super().predict(x)

        # Step 2: Apply neutralization

        if self.neutralize_flag:
            x["prediction"] = predictions
            neutralized_predictions = self.neutralize(x)
            return neutralized_predictions

        x["prediction"] = predictions
        return x


def build_sub_model_pipeline(
    features: list[str],
    estimator_cls,
    estimator_params: dict,
    proportion: float,
    neutralize_flag: bool,
):
    estimator = estimator_cls(**estimator_params)
    return SubModelPipeline(
        [("regressor", estimator)],
        proportion=proportion,
        features=features,
        neutralize_flag=neutralize_flag,
    )


def build_meta_model_pipeline(
    features: list[str],
    estimator_cls,
    estimator_params: dict,
    proportion: float,
    neutralize_flag: bool,
):
    estimator = estimator_cls(**estimator_params)
    return MetaModelPipeline(
        [("regressor", estimator)],
        proportion=proportion,
        features=features,
        neutralize_flag=neutralize_flag,
    )
