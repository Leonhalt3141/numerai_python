import gc
import glob
import logging
import pickle
from typing import List

import joblib
import numpy as np
import pandas as pd
import scipy.stats as st
from app.numerai_sbumit import submit
from app.tune.bayesian_tuning import tune
from sklearn.model_selection import KFold
from tqdm import tqdm
from xgboost import XGBRegressor

from .base_model import NumeraiBaseEstimator
from .feature_selection import FeatureSelection

logger = logging.getLogger()
generator = np.random.Generator(np.random.PCG64(seed=0))


class NumeraiModel(NumeraiBaseEstimator):
    def __init__(
        self,
        train_data_path: str = "train.parquet",
        live_data_path: str = "live.parquet",
        valid_data_path: str = "validation.parquet",
        fold_num: int = 5,
        sample_ratio: float = 0.2,
        chunk_size=200000,
        target_col: str = "target",
        xgboost_params: dict = None,
    ):
        self.fold_num = fold_num
        self.sample_ratio = sample_ratio
        self.target_col = target_col
        self.chunk_size = chunk_size
        self.feature_size = 1205
        if xgboost_params is None:
            self.xgboost_params = {
                "n_estimators": 3000,
                "boosting_type": "gbdt",
                "learning_rate": 0.01,
                "metric": "rmse",
                "max_bin": 5,
                "colsample_bytree": 0.4,
                "seed": 0,
                "force_row_wise": True,
            }
        else:
            self.xgboost_params = xgboost_params

        self.list_feature_cols_file = "list_feature_cols.pkl"
        self.drop_cols = None

        super().__init__(train_data_path, live_data_path, valid_data_path)

    def filter_df(self, df, drop_cols):
        df.drop(drop_cols, axis=1, inplace=True)
        df.fillna(0, inplace=True)

    def train_model(self):
        np.random.seed(0)

        train_df = self.load_data(self.train_data_path)
        size = train_df.shape[0]

        self.drop_cols = [
            col for col in train_df.columns if "target" in col and col != "target"
        ]
        self.filter_df(train_df, self.drop_cols)

        feature_cols = FeatureSelection.select_features_by_size(
            self.target_col, self.feature_size
        ).tolist()

        pickle.dump(feature_cols, open(self.list_feature_cols_file, "wb"))
        train_df = train_df[feature_cols + [self.target_col]]

        cv = KFold(n_splits=self.fold_num, random_state=0, shuffle=True)
        pbar = tqdm(cv.split(train_df), total=self.fold_num)
        for fold, (trn_idx, val_idx) in tqdm(
            enumerate(pbar, start=1), total=self.fold_num
        ):
            if fold != 1:
                train_df = self.load_data(self.train_data_path)
                self.filter_df(train_df, self.drop_cols)

            train_size = int(size * self.sample_ratio)
            valid_size = int(size * self.sample_ratio)

            trn_idx = generator.choice(trn_idx, train_size)
            val_idx = generator.choice(val_idx, valid_size)
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
        # feature_cols = FeatureSelection.select_features_by_size(
        #     self.target_col, self.feature_size
        # ).tolist()
        models = []

        for path in glob.glob("model.lgb.fold_*.pkl"):
            models.append(pickle.load(open(path, "rb")))

        valid = self.load_data(self.valid_data_path)

        if self.drop_cols is None:
            self.drop_cols = [
                col for col in valid.columns if "target" in col and col != "target"
            ]

        self.filter_df(valid, self.drop_cols)
        size = valid.shape[0]

        preds_valid = np.zeros(valid.shape[0])
        logger.info("Predicting")
        chunk_total = len(valid) // self.chunk_size + 1

        for i, model in enumerate(models):
            logger.info(f"{i + 1}/{len(models)}")
            for chunk_num in tqdm(range(chunk_total), total=chunk_total):
                start_index = chunk_num * self.chunk_size
                end_index = min(start_index + self.chunk_size, size)
                chunk = valid[start_index:end_index]
                preds_valid[start_index:end_index] += model.predict(
                    chunk[features].values
                ) / len(models)

        valid["prediction"] = preds_valid

        d = {}
        for era in tqdm(sorted(set(valid["era"]))):
            d[era] = st.spearmanr(
                valid.query("era == @era")["prediction"],
                valid.query("era == @era")["target"],
                nan_policy="omit",
            )[0]

        s = pd.Series(d)

        logger.info(s)
        mean = np.nanmean(s.values)
        std = np.nanstd(s.values)
        sharpe_ratio = mean / std
        print(
            f"Mean: {round(mean, 3)}, S.D.: {round(std, 3)}, Sharpe ratio: {round(sharpe_ratio, 3)}"
        )
        logger.info(
            f"Mean: {round(mean, 3)}, S.D.: {round(std, 3)}, Sharpe ratio: {round(sharpe_ratio, 3)}"
        )

        return models, features, valid, preds_valid

    def train_with_optuna(self, feature_cols: List[str]):
        logger.info("Load data")
        train_df = self.load_data(self.train_data_path)
        size = train_df.shape[0]

        self.drop_cols = [
            col for col in train_df.columns if "target" in col and col != "target"
        ]
        self.filter_df(train_df, self.drop_cols)

        train_size = int(size * self.sample_ratio)

        trn_idx = generator.integers(0, size, train_size)

        trn_idx.sort()

        trn_x = train_df.iloc[trn_idx, :][feature_cols].values
        trn_y = train_df.iloc[trn_idx, :][self.target_col].values

        logger.info("Set training")
        study = tune(trn_x, trn_y, logger=logger.getChild("Optuna-Hyper Parameters"))
        logger.info("End training")
        logger.info("Export trained model")
        best_params = {
            "n_estimators": study.best_params["n_estimators"],
            "boosting_type": "gbdt",
            "learning_rate": 0.01,
            "metric": "rmse",
            "colsample_bytree": study.best_params["colsample_bytree"],
            "max_bin": study.best_params["max_bin"],
            "max_depth": study.best_params["max_depth"],
            "min_child_weight": study.best_params["min_child_weight"],
            "gamma": study.best_params["gamma"],
            "reg_alpha": study.best_params["reg_alpha"],
            "reg_lambda": study.best_params["reg_lambda"],
            "tree_method": "gpu_hist",
        }
        self.xgboost_params = best_params
        del trn_x, trn_y, trn_idx
        gc.collect()
        self.train_model()

    def submission(self, models, features, valid, preds_valid, flag_submit):
        live = pd.read_parquet(self.live_data_path)
        preds_live = np.zeros(live.shape[0])
        feature_data = live[features].values

        for model in models:
            preds_live += model.predict(feature_data) / len(models)

        submit(live, preds_live, valid, preds_valid, flag_submit=flag_submit)

    def prediction_submit_pipeline(self):
        logger.info("Prediction with validation data")
        models, features, valid, preds_valid = self.valid_data_prediction()
        features = pickle.load(open("list_feature_cols.pkl", "rb"))

        models = []

        for path in glob.glob("model.lgb.fold_*.pkl"):
            models.append(pickle.load(open(path, "rb")))

        logger.info("Submission")
        self.submission(models, features, valid, preds_valid, "valid,live")
        logger.info("Complete")
