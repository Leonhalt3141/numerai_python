import json
import uuid

import joblib
import numpy as np
import optuna
import pandas as pd
from app.logger.log import get_logger
from app.pipe.pipe001 import build_pipeline
from optuna.trial import Trial
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

logger = get_logger()
DATA_VERSION = 5.0
generator = np.random.Generator(np.random.PCG64(seed=0))


def get_estimator_params(trial: Trial, seed=123):

    return XGBRegressor, {
        "n_estimators": trial.suggest_int("n_estimators", 50, 3000),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "max_bin": trial.suggest_int("max_bin", 2, 10),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
        "gamma": trial.suggest_float("gamma", 0, 1),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 100, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 100, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5, log=True),
        "eval_metric": "rmse",
        "seed": seed,
        "tree_method": trial.suggest_categorical(
            "tree_method", ["hist", "exact", "approx"]
        ),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    }


def get_estimator_params_from_trial(trial: Trial):
    return {
        "n_estimators": trial.params.get("n_estimators"),
        "colsample_bytree": trial.params.get("colsample_bytree"),
        "max_bin": trial.params.get("max_bin"),
        "max_depth": trial.params.get("max_depth"),
        "min_child_weight": trial.params.get("min_child_weight"),
        "gamma": trial.params.get("gamma"),
        "reg_alpha": trial.params.get("reg_alpha"),
        "reg_lambda": trial.params.get("reg_lambda"),
        "learning_rate": trial.params.get("learning_rate"),
        "eval_metric": "rmse",
        "tree_method": trial.params.get("tree_method"),
        "subsample": trial.params.get("subsample"),
    }


def evaluate(predictions, targets):
    rmse = root_mean_squared_error(y_true=targets, y_pred=predictions)
    return rmse


def objective(trial: Trial, x_data, y_data):
    estimator, params = get_estimator_params(trial)
    pipeline = build_pipeline(estimator, params)
    pipeline.fit(x_data, y_data)

    predictions = pipeline.predict(x_data)
    return evaluate(predictions, y_data)


def train(x_train, y_train, x_test, y_test):

    try:
        logger.info("Create Optuna study")
        study = optuna.create_study(direction="minimize")

        logger.info("Start Optuna optimize")
        study.optimize(
            lambda trial: objective(trial, x_train, y_train), n_trials=5, n_jobs=5
        )

        trial = study.best_trial
        best_params_str = ""
        for key, value in trial.params.items():
            best_params_str += f"\t{key}: {value}\n"
        best_trial_message = f"""
        ----- Best Trial -----
        Value - {trial.value}
        {best_params_str}
        """
        logger.info("The best trial result")
        logger.info(best_trial_message)

        logger.info("Loading optimal parameters and setting optimal estimator pipeline.")
        estimator_params = get_estimator_params_from_trial(trial)
        optimal_pipeline = build_pipeline(XGBRegressor, estimator_params)
        optimal_pipeline.fit(x_train, y_train)

        logger.info("Predicting with test data")
        predicted_y = optimal_pipeline.predict(x_test)

        logger.info("Calculating metrics")
        sharpe_ratio_value = evaluate(predicted_y, y_test)
        mae = mean_absolute_error(y_true=y_test, y_pred=predicted_y)
        rmse = root_mean_squared_error(y_true=y_test, y_pred=predicted_y)
        r2 = r2_score(y_true=y_test, y_pred=predicted_y)

        logger.info(f"Sharpe ratio: {sharpe_ratio_value}")
        logger.info(f"MAE: {mae}")
        logger.info(f"RMSE: {rmse}")
        logger.info(f"R2: {r2}")

        return optimal_pipeline

    except Exception as e:
        logger.error(e)


def main():
    feature_metadata = json.load(open("data/features.json"))
    features = feature_metadata["feature_sets"][
        "medium"
    ]  # use "all" for better performance. Requires more RAM.

    logger.info("Loading data")
    train_df = pd.read_parquet(
        "data/train.parquet", columns=["era"] + features + ["target"]
    )
    x = train_df[features].values
    y = train_df["target"].values

    logger.info("Split data into train and test")
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)

    logger.info("Start training")
    optimal_pipeline = train(train_x, train_y, test_x, test_y)

    logger.info("Saving model")
    model_filename = f"{uuid.uuid4().hex}_model.pkl"
    joblib.dump(optimal_pipeline, model_filename)
