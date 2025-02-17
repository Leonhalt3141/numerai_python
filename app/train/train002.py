import datetime
from uuid import uuid4

import joblib
import numpy as np
import optuna
import pandas as pd
from app.data.data002 import prepare_data
from app.logger.log import get_logger
from app.metrics.scoring import calculate_sharpe_ratio
from app.model.model002 import get_estimator_params, get_estimator_params_from_trial
from app.pipe.pipe002 import build_pipeline
from optuna import Trial
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBRegressor

seed = 123
train_name = "train002"
uid = uuid4().hex
start_time = datetime.datetime.now()
n_trials = 5

logger = get_logger(
    name=train_name,
    log_file=f"log/{start_time.strftime('%Y%m%d%H%M%S')}_{train_name}.log",
)


def evaluate(predictions, targets):
    rmse = root_mean_squared_error(y_true=targets, y_pred=predictions)
    return rmse


def _score(sub_df: pd.DataFrame) -> np.float32:
    """Calculates Spearman correlation"""
    return spearmanr(sub_df["target"], sub_df["prediction"])[0]


def objective(trial: Trial, x_data, y_data, features):
    estimator, params = get_estimator_params(trial)

    proportion = trial.suggest_float("proportion", 0.1, 1)

    pipeline = build_pipeline(features, estimator, params, proportion)

    pipeline.fit(x_data, y_data)

    predictions = pipeline.predict(x_data[features])
    return evaluate(predictions["prediction"].values, y_data.values)


def train(x_train, y_train, x_test, y_test, features):
    logger.info("Create Optuna study")

    study = optuna.create_study(direction="minimize")

    logger.info("Start Optuna optimize")
    study.optimize(
        lambda trial_: objective(trial_, x_train, y_train, features),
        n_trials=n_trials,
        n_jobs=5,
    )

    best_trial = study.best_trial
    best_params_str = ""
    for key, value in best_trial.params.items():
        best_params_str += f"\t{key}: {value}\n"
    best_trial_message = f"""
                ----- Best Trial -----
                Value: {best_trial.value}
                {best_params_str}
                """
    logger.info("The best trial result")
    logger.info(best_trial_message)

    logger.info("Loading optimal parameters and setting optimal estimator pipeline.")
    estimator_params = get_estimator_params_from_trial(best_trial)
    optimal_pipeline = build_pipeline(
        features, XGBRegressor, estimator_params, best_trial.params.get("proportion")
    )
    optimal_pipeline.fit(x_train, y_train)

    logger.info("Predicting with test data")
    predicted_y = optimal_pipeline.predict(x_test)

    logger.info("Calculating metrics")

    x_test["target"] = y_test["target"].values
    corr = x_test.groupby("era").apply(_score)
    sharpe_ratio_value = calculate_sharpe_ratio(corr)
    mae = mean_absolute_error(
        y_true=y_test["target"].values, y_pred=predicted_y["prediction"]
    )
    rmse = root_mean_squared_error(
        y_true=y_test["target"].values, y_pred=predicted_y["prediction"]
    )
    r2 = r2_score(y_true=y_test["target"].values, y_pred=predicted_y["prediction"])

    # logger.info(f"Correlation: {corr}")
    logger.info(f"Sharpe ratio: {sharpe_ratio_value}")
    logger.info(f"MAE: {mae}")
    logger.info(f"RMSE: {rmse}")
    logger.info(f"R2: {r2}")

    return optimal_pipeline


def main():
    try:
        train_x, test_x, train_y, test_y, features = prepare_data(
            test_size=0.2, logger=logger
        )

        logger.info("Start training")
        optimal_pipeline = train(train_x, train_y, test_x, test_y, features)

        logger.info("Saving model")
        model_filename = (
            f"model/{start_time.strftime('%Y%m%d%H%M')}_{uid}_{train_name}_model.pkl"
        )
        joblib.dump(optimal_pipeline, model_filename)
        logger.info(f"Model saved to {model_filename}")
    except Exception as e:
        logger.error(e)
