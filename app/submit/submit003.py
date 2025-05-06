import json
from glob import glob
from typing import Optional

import joblib
import numerapi
import pandas as pd
from sklearn.pipeline import Pipeline

from ..data.data_import import read_features
from . import public_id, secret_key

model_id = "003"
feature_type = "medium"
target_col = "target"
feature_metadata = read_features()
features = ["era"] + feature_metadata["feature_sets"]["medium"]


def get_latest_sub_models():
    latest_datetime = (
        sorted(glob(f"model/*train{model_id}_model.pkl"))[-1]
        .split("_")[-1]
        .split(".")[0]
        .split("_")[0]
    )
    return sorted(glob(f"model/{latest_datetime}*train{model_id}_model.pkl"))[-1]


def load_live_data() -> pd.DataFrame:
    print("Loading live data")
    with open("data/features.json") as f:
        feature_metadata = json.load(f)

    features = ["era"] + feature_metadata["feature_sets"]["medium"]

    df = pd.read_parquet("data/live.parquet", columns=features)
    df["era"] = 0
    return df


def load_valid_data() -> (pd.DataFrame, list[str]):
    print("Loading validation data")
    with open("data/features.json") as f:
        feature_metadata = json.load(f)

    features = ["era"] + feature_metadata["feature_sets"]["medium"]

    df = pd.read_parquet("data/validation.parquet", columns=features + [target_col])
    df["era"] = df["era"].astype(int)
    return df, features


def predict(df: pd.DataFrame):
    predictions = pd.DataFrame(index=features)

    sub_model_files = get_latest_sub_models()
    for sub_model_file in sub_model_files:
        sub_model = joblib.load(sub_model_file)
        sub_target = "_".join(sub_model_file.split(".")[0].split("_")[-4:-3])
        predictions[sub_target] = sub_model.predict(df[features])

    ensemble = predictions.rank(pct=True).mean(axis=1)
    return ensemble


def live_predict(live_df: Optional[pd.DataFrame] = None):
    live_df = live_df if live_df is not None else load_live_data()

    predictions = predict(live_df)
    return predictions


def valid_predict(
    features: Optional[list[str]] = None,
    valid_df: Optional[pd.DataFrame] = None,
):

    if valid_df is None and features is None:
        valid_df, features = load_valid_data()

    predictions = predict(valid_df)
    predictions[target_col] = valid_df[target_col]
    return predictions


def submission(
    live_df: Optional[pd.DataFrame] = None,
    valid_df: Optional[pd.DataFrame] = None,
    model_name="kuwaken_gbt",
    live_flag=True,
    valid_flag=True,
):

    # Upload your predictions using API
    if valid_flag:
        valid_df = valid_df if valid_df is not None else valid_predict()

        valid_sub_df = valid_df[["prediction"]].reset_index()
        valid_sub_df.to_csv(f"preds_valid_{model_name}.csv", index=False)

        napi = numerapi.NumerAPI(public_id=public_id, secret_key=secret_key)
        model_id = napi.get_models()[model_name]
        submission_id = napi.upload_diagnostics(
            f"preds_valid_{model_name}.csv", model_id=model_id
        )
        print(submission_id)
    if live_flag:
        live_df = live_df if live_df is not None else live_predict()

        live_sub_df = live_df[["prediction"]].reset_index()
        live_sub_df.to_csv(f"preds_live_{model_name}.csv", index=False)

        napi = numerapi.NumerAPI(public_id=public_id, secret_key=secret_key)
        model_id = napi.get_models()[model_name]
        submission_id = napi.upload_predictions(
            f"preds_live_{model_name}.csv", model_id=model_id
        )
        print(submission_id)
