import pandas as pd
from joblib import dump
from sklearn.base import BaseEstimator, RegressorMixin


class NumeraiBaseEstimator(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        train_data_path: str,
        live_data_path: str,
        valid_data_path: str,
        output_model_path: str = "xgboost_trained.model",
    ):
        self.train_data_path = train_data_path
        self.live_data_path = live_data_path
        self.valid_data_path = valid_data_path
        self.output_model_path = output_model_path

    def load_data(self, file_path: str = None):
        if file_path is None:
            return pd.read_parquet(self.train_data_path)
        return pd.read_parquet(file_path)

    def save_model(self, model):
        dump(model, self.output_model_path)

    def fit(self, x, y=None):
        pass

    def predict(self, x):
        pass

    def submission(self, **kwargs):
        pass

    def prediction_submit_pipeline(self):
        pass
