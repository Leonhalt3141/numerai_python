from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


def build_pipeline(estimator_cls, estimator_params: dict):
    estimator = estimator_cls(**estimator_params)
    return Pipeline([("regressor", estimator)], memory=None)
