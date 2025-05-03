##

import datetime
from uuid import uuid4

import optuna
import pandas as pd
from app.data.data002 import prepare_data
from app.data.data_import import open_train_data, read_features
from app.ensemble.select import select_sub_target_array
from app.logger.log import get_logger
from app.metrics.scoring import evaluate_sharpe_ratio
from app.model.model003 import (
    get_estimator_params,
    get_estimator_params_from_trial,
    get_sub_model_estimator_params,
    get_sub_model_estimator_params_from_trial,
)
from app.pipe.pipe003 import build_meta_model_pipeline, build_sub_model_pipeline
from optuna import Trial
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

seed = 123
train_name = "train003"
uid = uuid4().hex
start_time = datetime.datetime.now()
n_trials = 1


def sub_model_objective(trial: Trial, x_data, y_data, features):
    estimator, params = get_sub_model_estimator_params(trial)

    proportion = trial.suggest_float("proportion", 0.1, 1)
    neutralize_flag = trial.suggest_categorical("neutralize_flag", [True, False])

    pipeline = build_sub_model_pipeline(
        features, estimator, params, proportion, neutralize_flag
    )

    pipeline.fit(x_data, y_data)

    predictions = pipeline.predict(x_data[features])
    return evaluate_sharpe_ratio(predictions, y_data)


def meta_model_objective(trial: Trial, x_data, y_data, features):
    estimator, params = get_estimator_params(trial)

    proportion = trial.suggest_float("proportion", 0.1, 1)
    neutralize_flag = trial.suggest_categorical("neutralize_flag", [True, False])

    pipeline = build_meta_model_pipeline(
        features, estimator, params, proportion, neutralize_flag
    )

    pipeline.fit(x_data, y_data)

    predictions = pipeline.predict(x_data[features])
    return evaluate_sharpe_ratio(predictions, y_data)


##
sub_target_array = select_sub_target_array()
sub_target = sub_target_array[0]

feature_metadata = read_features()
features = ["era"] + feature_metadata["feature_sets"]["medium"]

train_df = pd.read_parquet(
    "data/train.parquet", columns=features + sub_target_array.tolist()
)
train_df["era"] = train_df["era"].astype(int)
x = train_df[features]
y = train_df[[sub_target]]
# y.rename({sub_target: "target"}, axis="columns", inplace=True)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed
)

##
study = optuna.create_study(direction="maximize")

study.optimize(
    lambda trial_: sub_model_objective(trial_, x_train, y_train, features),
    n_trials=n_trials,
    n_jobs=1,
)
##
