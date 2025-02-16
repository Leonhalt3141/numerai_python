import datetime
from uuid import uuid4

import joblib
from app.data.data002 import prepare_data
from app.logger.log import get_logger
from app.model.model002 import get_estimator_params, get_estimator_params_from_trial
from app.pipe.pipe002 import build_pipeline
from optuna import Trial
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBRegressor

uid = uuid4().hex
start_time = datetime.datetime.now()

logger = get_logger(
    name="train002", log_file=f"{start_time.strftime('%Y%m%d%H%M%S')}_train002.log"
)


def evaluate(predictions, targets):
    rmse = root_mean_squared_error(y_true=targets, y_pred=predictions)
    return rmse


def objective(trial: Trial, x_data, y_data, features):
    estimator, params = get_estimator_params(trial)

    proportion = trial.suggest_float("proportion", 0.1, 1)

    pipeline = build_pipeline(features, estimator, params, proportion)

    pipeline.fit(x_data, y_data)

    predictions = pipeline.predict(x_data[features])
    return evaluate(predictions["prediction"].values, y_data.values)


def train(x_train, y_train, x_test, y_test, features):
    try:
        logger.info("Create Optuna study")

        import optuna

        study = optuna.create_study(direction="minimize")

        logger.info("Start Optuna optimize")
        study.optimize(
            lambda trial: objective(trial, x_train, y_train, features),
            n_trials=5,
            n_jobs=5,
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
        optimal_pipeline = build_pipeline(
            features, XGBRegressor, estimator_params, trial.params.get("proportion")
        )
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
    train_x, test_x, train_y, test_y, features = prepare_data(
        test_size=0.2, logger=logger
    )

    logger.info("Start training")
    optimal_pipeline = train(train_x, train_y, test_x, test_y, features)

    logger.info("Saving model")
    model_filename = f"{uid}_model.pkl"
    joblib.dump(optimal_pipeline, model_filename)
