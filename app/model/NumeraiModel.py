import gc
import glob
import pickle

import joblib
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
from xgboost import XGBRegressor

from .base_model import NumeraiBaseEstimator


class NumeraiModel(NumeraiBaseEstimator):
    def __init__(
        self,
        train_data_path: str,
        live_data_path: str,
        fold_num: int = 5,
        sample_ratio: float = 0.1,
        target_col: str = "target",
        xgboost_params: dict = None,
    ):
        self.fold_num = fold_num
        self.sample_ratio = sample_ratio
        self.target_col = target_col
        if xgboost_params is None:
            self.xgboost_params = {
                "n_estimators": 3000,
                "boosting_type": "gbdt",
                "learning_rate": 0.01,
                "metric": "rmse",
                "max_bin": 5,
                "colsample_bytree": 0.1,
                "seed": 0,
                "force_row_wise": True,
            }
        else:
            self.xgboost_params = xgboost_params

        self.list_feature_cols_file = "list_feature_cols.pkl"

        super().__init__(train_data_path, live_data_path)

    def train_model(self):
        np.random.seed(0)

        train_df = self.load_data(self.train_data_path)

        drop_cols = [
            col for col in train_df.columns if "target" in col and col != "target"
        ]
        train_df.drop(drop_cols, axis=1, inplace=True)
        train_df.fillna(0, inplace=True)

        feature_cols = [col for col in train_df.columns if "feature" in col][:1205]
        pickle.dump(feature_cols, open(self.list_feature_cols_file, "wb"))

        cv = KFold(n_splits=self.fold_num)
        pbar = tqdm(cv.split(train_df), total=self.fold_num)
        for fold, (trn_idx, val_idx) in tqdm(
            enumerate(pbar, start=1), total=self.fold_num
        ):
            if fold != 1:
                train_df = self.load_data(self.train_data_path)
                train_df.drop(drop_cols, axis=1, inplace=True)
                train_df.fillna(0, inplace=True)

            train_size = int(trn_idx.shape[0] * self.sample_ratio)
            valid_size = int(val_idx.shape[0] * self.sample_ratio)

            trn_idx = np.random.choice(trn_idx, train_size)
            val_idx = np.random.choice(val_idx, valid_size)
            trn_idx.sort()
            val_idx.sort()

            pbar.set_description(desc="Slicing")
            trn_x = train_df.iloc[trn_idx, :][feature_cols].values
            trn_y = train_df.iloc[trn_idx, :][self.target_col].values
            val_x = train_df.iloc[val_idx, :][feature_cols].values
            val_y = train_df.iloc[val_idx, :][self.target_col].values

            del train_df
            gc.collect()

            pbar.set_description(desc="Train start")
            model = XGBRegressor(**self.xgboost_params)

            model.fit(
                trn_x,
                trn_y,
                eval_set=[(val_x, val_y)],
                verbose=100,
                early_stopping_rounds=50,
            )

            joblib.dump(model, open(f"model.lgb.fold_{fold}.pkl", "wb"))

            del trn_x, trn_y, val_x, val_y, model
            gc.collect()

    def valid_data_prediction(self):
        features = pickle.load(open(self.list_feature_cols_file, "rb"))
        models = []

        for path in glob.glob("model.lgb.fold_*.pkl"):
            models.append(pickle.load(open(path, "rb")))
