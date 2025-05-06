##
import json

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

##
with open("data/features.json") as f:
    feature_metadata = json.load(f)

train_df = pd.read_parquet("data/train.parquet")
##
features = feature_metadata["feature_sets"]["medium"]
targets = feature_metadata["targets"]

##
corr_df = train_df[targets].corr()

##
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr_df, vmin=-1, vmax=1, cmap="jet", ax=ax)
fig.tight_layout()
plt.show()
##
target_info_df = train_df[targets].describe()
##
sub_target = targets[5]
sub_target = "target_xerxes_20"
corr_era_df = train_df[["era", sub_target, "target"]].groupby("era").corr()
corr_era_df = corr_era_df.iloc[::2]
corr_era_df["era_int"] = [int(x[0]) for x in corr_era_df.index]
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(corr_era_df["era_int"], corr_era_df["target"])
mean_v = corr_era_df["target"].mean()
std_v = corr_era_df["target"].std()
ax.set_title(
    f"{sub_target} correlation with main target: mean-{mean_v:0.3f}, std-{std_v:0.3f}"
)
ax.set_xlabel("era")
ax.set_ylabel("Correlation")
ax.set_ylim([0, 1])
ax.grid()
fig.tight_layout()
plt.show()
##
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
##
cor_max = 0.75
cor_min = 0.30

corr_agg_selected_df = corr_agg_df[
    (corr_agg_df["mean"] >= cor_min) & (corr_agg_df["mean"] <= cor_max)
]
corr_agg_selected_df
##
corr_agg_selected_df.shape

##
