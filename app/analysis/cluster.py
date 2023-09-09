##
import gc
from typing import Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm


def feat_imp_mda(
    x: pd.DataFrame,
    y: pd.DataFrame,
    n_splits=10,
) -> pd.DataFrame:
    cv_gen = KFold(n_splits=n_splits)
    scr0, scr1 = pd.Series(), pd.DataFrame(columns=x.columns)

    clf = DecisionTreeClassifier(
        criterion="entropy",
        max_features=1,
        class_weight="balanced",
        min_weight_fraction_leaf=0,
    )
    clf: Union[DecisionTreeClassifier, BaggingClassifier] = BaggingClassifier(
        estimator=clf,
        n_estimators=100,
        max_features=1.0,
        max_samples=1.0,
        oob_score=False,
        n_jobs=20,
    )

    imp = None
    coef = np.unique(y.values)[1]
    print(coef)
    y = np.round(y / coef)

    for i, (train, test) in enumerate(cv_gen.split(X=x)):
        print(f"{i + 1}/{n_splits}")
        x0, y0 = x.iloc[train, :], y.iloc[train]

        x1, y1 = x.iloc[test, :], y.iloc[test]

        fit = clf.fit(X=x0, y=y0)

        prob = fit.predict_proba(x1)

        scr0.loc[i] = -log_loss(y1, prob, labels=clf.classes_)
        pbar2 = tqdm(x.columns)
        pbar2.set_description(desc=f"{i + 1}/{n_splits}")
        for j in pbar2:
            x1_ = x1.copy(deep=True)
            x1_[j] = np.random.permutation(x1_[j].values)
            prob = fit.predict_proba(x1_)
            scr1.loc[i, j] = -log_loss(y1, prob, labels=clf.classes_)
            del x1_, prob

        imp = (-1 * scr1).add(scr0, axis=0)
        imp = imp / (-1 * scr1)

        imp = pd.concat(
            {"mean": imp.mean(), "std": imp.std() * imp.shape[0] ** -0.5}, axis=1
        )
        gc.collect()

    return imp


def graph_mda(imp_sorted: pd.DataFrame):
    fig = plt.figure(figsize=(10, 4))
    n = 60
    sns.barplot(x=imp_sorted.loc[imp_sorted.index[:n], "mean"], y=imp_sorted.index[:n])
    fig.tight_layout()
    plt.show()


def process_all_target_mda(save=True):
    size = 1000
    np.random.seed(size)
    train_df = pd.read_parquet("train.parquet")
    index = np.random.permutation(train_df.shape[0])[:size]

    train_df = train_df.iloc[index]
    train_df.fillna(0, inplace=True)

    feature_columns = [c for c in train_df.columns if "feature" in c]

    target_columns = [c for c in train_df.columns if "target" in c]
    x = train_df[feature_columns]

    for t, target_col in enumerate(["target"], 1):
        print(f"{t}/{len(target_columns)}: {target_col}")
        y = train_df[target_col]
        if y.values[10] % 0.25 == 0:
            print(f"{target_col}: Classifier")
            imp = feat_imp_mda(x, y, 5)
        else:
            print(f"{target_col}: Regressor")
            imp = feat_imp_mda(x, y, 5)

        imp["mean_abs"] = np.abs(imp["mean"].values)
        imp_sorted = imp.sort_values("mean_abs", ascending=False)

        graph_mda(imp_sorted)

        if save:
            imp_sorted.to_csv(f"MDA_result_{target_col}.csv")
