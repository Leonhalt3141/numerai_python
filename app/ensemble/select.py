import numpy as np
import pandas as pd
from app.data.data_import import open_train_data, read_features


def select_sub_target_array() -> np.ndarray[str]:
    feature_metadata = read_features()

    targets = feature_metadata["targets"]

    train_df = open_train_data()

    target_list = []
    mean_list = []
    std_list = []
    for sub_target in targets:
        if sub_target != "target":
            corr_era_df = train_df[["era", sub_target, "target"]].groupby("era").corr()
            corr_era_df = corr_era_df.iloc[::2]
            corr_era_df["era_int"] = [int(x[0]) for x in corr_era_df.index]

            target_list.append(sub_target)
            mean_list.append(corr_era_df["target"].mean())
            std_list.append(corr_era_df["target"].std())

    corr_agg_df = pd.DataFrame(
        {"target": target_list, "mean": mean_list, "std": std_list}
    ).sort_values(by="mean", ascending=False, axis=0)

    cor_max = 0.75
    cor_min = 0.30

    corr_agg_selected_df = corr_agg_df[
        (corr_agg_df["mean"] >= cor_min) & (corr_agg_df["mean"] <= cor_max)
    ]
    del train_df, corr_era_df, corr_agg_df

    return corr_agg_selected_df["target"].values
