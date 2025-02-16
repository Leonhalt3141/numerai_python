from optuna.trial import Trial
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor


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
