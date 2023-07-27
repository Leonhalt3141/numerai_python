import gc
import glob
import json
import logging
import pickle
import sys
import traceback
from pathlib import Path
from typing import Optional, Union

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy
import scipy.stats as st
from joblib import dump
from sklearn.model_selection import KFold
from tqdm import tqdm
from xgboost import XGBRegressor

from app.numerai_sbumit import submit

params = {
    "n_estimators": 3000,
    "boosting_type": "gbdt",
    "learning_rate": 0.01,
    "metric": "rmse",
    "max_bin": 5,
    "colsample_bytree": 0.1,
    "seed": 0,
    "force_row_wise": True,
}


ERA_COL = "era"
TARGET_COL = "target_nomi_v4_20"
DATA_TYPE_COL = "data_type"
EXAMPLE_PREDS_COL = "example_preds"
MODEL_DIRECTORY = "models"
MODEL_CONFIGS_DIRECTORY = "model_configs"
PREDICTION_FILES_DIRECTORY = "prediction_files"


logger = logging.getLogger("Numerai")


class EnsembleModel:
    def __init__(self):
        self.cv_num = 5
        self.downsampling_ratio = 0.1

    def save_prediction(self, df: pd.DataFrame, filename: str):
        try:
            Path(PREDICTION_FILES_DIRECTORY).mkdir(exist_ok=True, parents=True)
        except Exception:
            message = traceback.format_exc()
            logger.error(message)

        df.to_csv(f"{PREDICTION_FILES_DIRECTORY}/{filename}.csv", index=True)

    def save_model(self, model, filename):
        try:
            Path(MODEL_DIRECTORY).mkdir(exist_ok=True, parents=True)
        except Exception:
            message = traceback.format_exc()
            logger.error(message)
        pd.to_pickle(model, f"{MODEL_DIRECTORY}/{filename}.pkl")

    def load_model(self, filename):
        path = Path(f"{MODEL_DIRECTORY}/{filename}.pkl")
        if path.is_file():
            model = pd.read_pickle(f"{MODEL_DIRECTORY}/{filename}.pkl")
        else:
            model = None
        return model

    def load_model_config(self, model_name):
        path_str = f"{MODEL_CONFIGS_DIRECTORY}/{model_name}.json"
        path = Path(path_str)
        if path.is_file():
            with open(path_str, "r") as fp:
                model_config = json.load(fp)
        else:
            model_config = None
        return model_config

    def get_biggest_change_features(self, corrs: pd.DataFrame, n: int):
        all_eras = corrs.index.sort_values()
        h1_eras = all_eras[: len(all_eras) // 2]
        h2_eras = all_eras[len(all_eras) // 2 :]

        h1_corr_means = corrs.loc[h1_eras, :].mean()
        h2_corr_means = corrs.loc[h2_eras, :].mean()

        corr_diffs = h2_corr_means - h1_corr_means
        worst_n = corr_diffs.abs().sort_values(ascending=False).head(n).index.tolist()
        return worst_n

    def get_time_series_cross_val_splits(self, data: pd.DataFrame, embargo=12):
        all_train_eras = data[ERA_COL].unique()
        len_split = len(all_train_eras) // self.cv_num
        test_splits = [
            all_train_eras[i * len_split : (i + 1) * len_split]
            for i in range(self.cv_num)
        ]
        # fix the last test split to have all the last eras, in case the number of eras wasn't divisible by cv
        remainder = len(all_train_eras) % self.cv_num
        if remainder != 0:
            test_splits[-1] = np.append(test_splits[-1], all_train_eras[-remainder:])

        train_splits = []
        for test_split in test_splits:
            test_split_max = int(np.max(test_split))
            test_split_min = int(np.min(test_split))
            # get all of the eras that aren't in the test split
            train_split_not_embargoed = [
                e
                for e in all_train_eras
                if not (test_split_min <= int(e) <= test_split_max)
            ]
            # embargo the train split so we have no leakage.
            # one era is length 5, so we need to embargo by target_length/5 eras.
            # To be consistent for all targets, let's embargo everything by 60/5 == 12 eras.
            train_split = [
                e
                for e in train_split_not_embargoed
                if abs(int(e) - test_split_max) > embargo
                and abs(int(e) - test_split_min) > embargo
            ]
            train_splits.append(train_split)

        # convenient way to iterate over train and test splits
        train_test_zip = zip(train_splits, test_splits)
        return train_test_zip

    def neutralize(
        self,
        df,
        columns,
        neutralizers=None,
        proportion=1.0,
        normalize=True,
        verbose=False,
    ):
        if neutralizers is None:
            neutralizers = []

        unique_eras = df[ERA_COL].unique()
        computed = []

        if verbose:
            iterator = tqdm(unique_eras)
        else:
            iterator = unique_eras

        for u in iterator:
            df_era = df[df[ERA_COL] == u]
            scores = df_era[columns].values
            if normalize:
                scores2 = []

                for x in scores.T:
                    x = (scipy.stats.rankdata(x, method="ordinal") - 0.5) / len(x)
                    x = scipy.stats.norm.ppf(x)
                    scores2.append(x)
                scores = np.array(scores2).T
            exposures = df_era[neutralizers].values

            scores -= proportion * exposures.dot(
                np.linalg.pinv(exposures.astype(np.float32), rcond=1e-6).dot(
                    scores.astype(np.float32)
                )
            )

            scores /= scores.std(ddof=0)

            computed.append(scores)

        return pd.DataFrame(np.concatenate(computed), columns=columns, index=df.index)

    def neutralize_series(self, series, by, proportion=1.0):
        scores = series.values.reshape(-1, 1)
        exposures = by.values.reshape(-1, 1)

        # this line makes series neutral to a constant column
        # so that it's centered and for sure gets corr 0 with exposures
        exposures = np.hstack(
            (exposures, np.array([np.mean(series)] * len(exposures)).reshape(-1, 1))
        )

        correction = proportion * (
            exposures.dot(np.linalg.lstsq(exposures, scores, rcond=None)[0])
        )
        corrected_scores = scores - correction
        neutralized = pd.Series(corrected_scores.ravel(), index=series.index)
        return neutralized

    def unif(self, df):
        x = (df.rank(method="first") - 0.5) / len(df)
        return pd.Series(x, index=df.index)

    def get_feature_neutral_mean(
        self, df, prediction_col, target_col, features_for_neutralization=None
    ):
        if features_for_neutralization is None:
            features_for_neutralization = [
                c for c in df.columns if c.startswith("feature")
            ]

        df.loc[:, "neutral_sub"] = self.neutralize(
            df, [prediction_col], features_for_neutralization
        )[prediction_col]

        scores = (
            df.groupby("era")
            .apply(lambda x: (self.unif(x["neutral_sub"]).corr(x[target_col])))
            .mean()
        )

        return np.mean(scores)

    def get_feature_neutral_mean_tb_era(
        self, df, prediction_col, target_col, tb, features_for_neutralization=None
    ):
        if features_for_neutralization is None:
            features_for_neutralization = [
                c for c in df.columns if c.startswith("feature")
            ]

        temp_df = df.reset_index(
            drop=True
        ).copy()  # Reset index due to use of argsort later

        temp_df.loc[:, "neutral_sub"] = self.neutralize(
            temp_df, [prediction_col], features_for_neutralization
        )[prediction_col]

        temp_df_argsort = temp_df.loc[:, "neutral_sub"].argsort()
        temp_df_tb_idx = pd.concat(
            [temp_df_argsort.iloc[:tb], temp_df_argsort.iloc[-tb:]]
        )

        temp_df_tb = temp_df.loc[temp_df_tb_idx]
        tb_fnc = self.unif(temp_df_tb["neutral_sub"]).corr(temp_df_tb[target_col])
        return tb_fnc

    def fast_score_by_date(self, df, columns, target, tb=None, era_col="era"):
        unique_eras = df[era_col].unique()
        computed = []
        for u in unique_eras:
            df_era = df[df[era_col] == u]
            era_pred: Union[np.ndarray, list] = np.float64(df_era[columns].values.T)
            era_target = np.float64(df_era[target].values.T)

            if tb is None:
                ccs = np.corrcoef(era_target, era_pred)[0, 1:]
            else:
                tbidx = np.argsort(era_pred, axis=1)
                tbidx: Union[np.ndarray, list] = np.concatenate(
                    [tbidx[:, :tb], tbidx[:, -tb:]], axis=1
                )
                ccs = [
                    np.corrcoef(era_target[tmpidx], tmppred[tmpidx])[0, 1]
                    for tmpidx, tmppred in zip(tbidx, era_pred)
                ]
                ccs = np.array(ccs)

            computed.append(ccs)

        return pd.DataFrame(
            np.array(computed), columns=columns, index=df[era_col].unique()
        )
