import gc
import glob
import pickle

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
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
target = "target"


def load_data() -> pd.DataFrame:
    return pd.read_parquet("train.parquet")


def train_model():
    np.random.seed(0)
    n = 5
    train_df = load_data()
    drop_cols = [col for col in train_df.columns if "target" in col and col != "target"]
    train_df.drop(drop_cols, axis=1, inplace=True)
    train_df.fillna(0, inplace=True)

    feature_cols = [col for col in train_df.columns if "feature" in col][:1205]
    pickle.dump(feature_cols, open("list_feature_cols.pkl", "wb"))

    cv = KFold(n_splits=n)
    pbar = tqdm(cv.split(train_df), total=n)
    ratio = 0.1
    for fold, (trn_idx, val_idx) in tqdm(enumerate(pbar, start=1), total=n):
        if fold != 1:
            train_df = load_data()
            drop_cols = [
                col for col in train_df.columns if "target" in col and col != "target"
            ]
            train_df.drop(drop_cols, axis=1, inplace=True)
            train_df.fillna(0, inplace=True)

        train_size = int(trn_idx.shape[0] * ratio)
        valid_size = int(val_idx.shape[0] * ratio)
        trn_idx = np.random.choice(trn_idx, train_size)
        val_idx = np.random.choice(val_idx, valid_size)
        trn_idx.sort()
        val_idx.sort()

        pbar.set_description(desc="Slicing")
        trn_x = train_df.iloc[trn_idx, :][feature_cols].values
        trn_y = train_df.iloc[trn_idx, :][target].values
        val_x = train_df.iloc[val_idx, :][feature_cols].values
        val_y = train_df.iloc[val_idx, :][target].values
        del train_df
        gc.collect()

        pbar.set_description(desc="Train start")

        model = XGBRegressor(**params)
        model.fit(
            trn_x,
            trn_y,
            eval_set=[(val_x, val_y)],
            verbose=100,
            early_stopping_rounds=50,
        )

        # model = lgb.LGBMRegressor(**params)
        # model.fit(
        #     trn_x,
        #     trn_y,
        #     eval_set=[(val_x, val_y)],
        #     verbose=100,
        #     early_stopping_rounds=50,
        # )

        joblib.dump(model, open(f"model.lgb.fold_{fold}.pkl", "wb"))

        del trn_x, trn_y, val_x, val_y, model
        gc.collect()


def valid_data_prediction():
    features = pickle.load(open("list_feature_cols.pkl", "rb"))
    models = []
    for path in glob.glob("model.lgb.fold_*.pkl"):
        models.append(pickle.load(open(path, "rb")))

    valid = pd.read_parquet("validation.parquet")
    drop_cols = [col for col in valid.columns if "target" in col and col != "target"]
    valid.drop(drop_cols, axis=1, inplace=True)
    valid.fillna(0, inplace=True)

    preds_valid = np.zeros(len(valid))
    print("Predicting")
    chunk_size = 200000
    chunk_total = len(valid) // chunk_size + 1
    for i, model in tqdm(enumerate(models), total=len(models), position=0):
        for chunk_num in tqdm(
            range(chunk_total), total=chunk_total, position=1, leave=False
        ):
            start_index = chunk_num * chunk_size
            end_index = min(chunk_num * chunk_size + chunk_size, len(valid))
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
        )[0]

    s = pd.Series(d)
    mean = s.mean()
    std = s.std()
    sharpe_ratio = mean / std
    print(
        f"Mean: {round(mean, 3)}, S.D.: {round(std, 3)}, Sharpe ratio: {round(sharpe_ratio, 3)}"
    )

    return models, features, valid, preds_valid


def save_model(model):
    dump(model, "xgboost_trained.model")


def train_pipeline():
    train_model()
    # save_model(model)


def submission(models, features, valid, preds_valid, flag_submit):
    live = pd.read_parquet("live.parquet")
    preds_live = np.zeros(len(live))
    feature_data = live[features].values
    for model in models:
        preds_live += model.predict(feature_data) / len(models)

    submit(live, preds_live, valid, preds_valid, flag_submit=flag_submit)


def prediction_submit_pipeline():
    print("Prediction with validation data")
    models, features, valid, preds_valid = valid_data_prediction()
    features = pickle.load(open("list_feature_cols.pkl", "rb"))
    models = []
    for path in glob.glob("model.lgb.fold_*.pkl"):
        models.append(pickle.load(open(path, "rb")))

    print("Submission")
    submission(models, features, valid, preds_valid, "valid,live")
    print("Complete")
