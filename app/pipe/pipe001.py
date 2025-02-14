from sklearn.pipeline import Pipeline


def build_pipeline(estimator_cls, estimator_params: dict):
    estimator = estimator_cls(**estimator_params)
    return Pipeline([("regressor", estimator)], memory=None)
