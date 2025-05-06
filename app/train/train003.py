import datetime
from uuid import uuid4

import numpy as np
import optuna
import pandas as pd
from app.data.data_import import open_train_data, read_features
from app.ensemble.select import select_sub_target_array
from app.logger.log import get_logger
from app.metrics.scoring import evaluate_sharpe_ratio
from app.model.model003 import get_estimator_params, get_estimator_params_from_trial
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
sub_target_num = 2
np_gen = np.random.Generator(np.random.PCG64(seed=0))

logger = get_logger(
    name=train_name,
    log_file=f"log/{start_time.strftime('%Y%m%d%H%M%S')}_{train_name}.log",
)


def sub_model_objective(trial: Trial, x_data, y_data, features):
    estimator, params = get_estimator_params(trial)

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


def train_sub_model(x_train, y_train, x_test, y_test, features, target_name):
    study = optuna.create_study(direction="maximize")

    logger.info(f"Start Optuna optimize - {target_name}")
    sub_logger = logger.getChild(target_name)

    study.optimize(
        lambda trial_: sub_model_objective(trial_, x_train, y_train, features),
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
    sub_logger.info("The best trial result")
    sub_logger.info(best_trial_message)

    sub_logger.info("Loading optimal parameters and setting optimal estimator pipeline.")
    estimator_params = get_estimator_params_from_trial(best_trial)
    optimal_pipeline = build_sub_model_pipeline(
        features,
        XGBRegressor,
        estimator_params,
        best_trial.params.get("proportion"),
        best_trial.params.get("neutralize_flag"),
    )
    optimal_pipeline.fit(x_train, y_train)

    sub_logger.info("Predicting with test data")
    predicted_y = optimal_pipeline.predict(x_test)

    sub_logger.info("Calculating metrics")

    x_test["target"] = y_test[target_name].values
    sharpe_ratio_value = evaluate_sharpe_ratio(predicted_y, y_test)

    mae = mean_absolute_error(
        y_true=y_test[target_name].values, y_pred=predicted_y["prediction"]
    )
    rmse = root_mean_squared_error(
        y_true=y_test[target_name].values, y_pred=predicted_y["prediction"]
    )
    r2 = r2_score(y_true=y_test[target_name].values, y_pred=predicted_y["prediction"])

    sub_logger.info(f"Sharpe ratio: {sharpe_ratio_value}")
    sub_logger.info(f"MAE: {mae}")
    sub_logger.info(f"RMSE: {rmse}")
    sub_logger.info(f"R2: {r2}")

    return optimal_pipeline


def save_model(model, model_filename):
    logger.info(f"Saving model to {model_filename}")
    import joblib

    joblib.dump(model, model_filename)
    logger.info(f"Model saved to {model_filename}")


def train_all_sub_models(train_x, test_x, train_y, test_y, sub_target_array, features):

    for i, sub_target in enumerate(sub_target_array, 1):
        logger.info(f"Start training - {sub_target} ({i}/{sub_target_array.shape[0]})")

        sub_model_pipeline = train_sub_model(
            train_x,
            train_y[[sub_target]],
            test_x,
            test_y[[sub_target]],
            features,
            sub_target,
        )

        save_model(
            sub_model_pipeline,
            f"model/{start_time.strftime('%Y%m%d%H%M')}_{uid}_{train_name}_{sub_target}_model.pkl",
        )


def main():
    try:
        sub_target_array = select_sub_target_array()

        sub_target_array = (
            np_gen.choice(sub_target_array, size=5, replace=False)
            if sub_target_num != -1
            else sub_target_array
        )

        feature_metadata = read_features()
        features = ["era"] + feature_metadata["feature_sets"]["medium"]

        train_df = pd.read_parquet(
            "data/train.parquet", columns=features + sub_target_array.tolist()
        )
        train_df["era"] = train_df["era"].astype(int)
        x = train_df[features]
        y = train_df[sub_target_array]
        train_x, test_x, train_y, test_y = train_test_split(
            x, y, test_size=0.2, random_state=seed
        )
        train_all_sub_models(train_x, test_x, train_y, test_y, sub_target_array, features)
    except Exception as e:
        logger.error(e)
