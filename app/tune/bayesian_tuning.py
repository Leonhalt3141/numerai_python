import logging
import time
from logging import Logger
from typing import Optional

import numpy as np
import optuna
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import cross_validate
from xgboost import XGBRegressor


class Objective:
    def __init__(self, x, y, seed: int = 0, cv: int = 3):
        # 変数X,yの初期化
        self.x = x
        self.y = y
        self.seed = seed
        self.cv = cv
        self.model: Optional[XGBRegressor] = None

    def root_mean_square_error(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=True)

    def __call__(self, trial):
        # ハイパーパラメータの設定
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 3000),
            "boosting_type": "gbdt",
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.1, 0.5),
            "max_bin": trial.suggest_int("max_bin", 2, 10),
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
            "gamma": trial.suggest_uniform("gamma", 0, 1),
            "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-5, 100),
            "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-5, 100),
            "learning_rate": 0.01,
            "metric": "rmse",
            "seed": self.seed,
        }

        net_marg = make_scorer(self.root_mean_square_error)

        self.model = XGBRegressor(**params)

        scores = cross_validate(
            self.model, X=self.x, y=self.y, cv=self.cv, scoring=net_marg, n_jobs=2
        )
        return scores["test_score"].mean()


def tune(x, y, seed: int = 1, n_trials: int = 10, logger: Optional[Logger] = None):
    start_time = time.time()
    logger = logger if logger is not None else logging.getLogger()

    objective = Objective(x, y)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )

    study.optimize(objective, n_trials=n_trials)

    best_parms = study.best_trial.params
    best_score = study.best_trial.value

    logger.info(f"Optimized parameters: {best_parms}\nBest score: {best_score}")
    logger.info(f"Process time {time.time() - start_time}")

    return study
