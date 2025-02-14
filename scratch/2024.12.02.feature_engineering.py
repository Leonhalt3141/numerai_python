import json

import cloudpickle
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
from numerai_tools.scoring import numerai_corr
from numerapi import NumerAPI

# initialize our API client
napi = NumerAPI()
DATA_VERSION = "v5.0"


##
def metrics(corr):
    corr_mean = corr.mean()
    corr_std = corr.std(ddof=0)
    corr_sharpe = corr_mean / corr_std
    max_drawdown = -(corr.cumsum().expanding(min_periods=1).max() - corr.cumsum()).max()

    eras = train.era.unique()
    halfway_era = len(eras) // 2
    corr_mean_first_half = corr.loc[eras[:halfway_era]].mean()
    corr_mean_second_half = corr.loc[eras[halfway_era:]].mean()
    delta = abs(corr_mean_first_half - corr_mean_second_half)

    return {
        "mean": corr_mean,
        "std": corr_std,
        "sharpe": corr_sharpe,
        "max_drawdown": max_drawdown,
        "delta": delta,
    }


##
features = json.load(open("scripts/features.json"))
# df = pd.read_parquet('scripts/train.parquet')
##
feature_sets = features["feature_sets"]

sizes = ["small", "medium", "all"]
groups = [
    "intelligence",
    "wisdom",
    "charisma",
    "dexterity",
    "strength",
    "constitution",
    "agility",
    "serenity",
    "all",
]

# compile the intersections of feature sets and feature groups
subgroups = {}
for size in sizes:
    subgroups[size] = {}
    for group in groups:
        subgroups[size][group] = set(feature_sets[size]).intersection(
            set(feature_sets[group])
        )

# convert to data frame and display the feature count of each intersection
pd.DataFrame(subgroups).applymap(len).sort_values(by="all", ascending=False)


##
# define the medium features and medium serenity features
medium_features = feature_sets["medium"]
med_serenity_feats = list(subgroups["medium"]["serenity"])

# Download the training data and feature metadata

# Load the just the medium feature set,
# this is a great feature of the parquet file format
train = pd.read_parquet(
    "scripts/train.parquet", columns=["era", "target"] + medium_features
)

# Downsample to every 4th era to reduce memory usage and
# speedup model training (suggested for Colab free tier).
train = train[train["era"].isin(train["era"].unique())]

##

# Compute the per-era correlation of each serenity feature to the target
per_era_corr = train.groupby("era").apply(
    lambda d: numerai_corr(d[med_serenity_feats], d["target"])
)

# Flip sign for negative mean correlation since we only care about magnitude
per_era_corr *= np.sign(per_era_corr.mean())
##
fig, ax = plt.subplots(figsize=(10, 5))
# Plot the per-era correlations
per_era_corr.cumsum().plot(
    title="Cumulative Absolute Value CORR of Features and the Target",
    figsize=(15, 5),
    legend=False,
    xlabel="Era",
    ax=ax,
)
plt.show()
##

# compute performance metrics for each feature
feature_metrics = [
    metrics(per_era_corr[feature_name]) for feature_name in med_serenity_feats
]

# convert to numeric DataFrame and sort
feature_metrics = (
    pd.DataFrame(feature_metrics, index=med_serenity_feats)
    .apply(pd.to_numeric)
    .sort_values("mean", ascending=False)
)

feature_metrics
##
fig, ax = plt.subplots(figsize=(12, 5))
# plot the performance metrics of the features as bar charts sorted by mean
feature_metrics.sort_values("mean", ascending=False).plot.bar(
    title="Performance Metrics of Features Sorted by Mean",
    subplots=True,
    figsize=(15, 6),
    layout=(2, 3),
    sharex=False,
    xticks=[],
    snap=False,
    ax=ax,
)
fig.tight_layout()
plt.show()
##

# plot the per era correlation of the feature with the highest vs lowest std
fig, ax = plt.subplots(figsize=(10, 5))
per_era_corr[[feature_metrics["std"].idxmin(), feature_metrics["std"].idxmax()]].plot(
    figsize=(15, 5),
    title="Per-era Correlation of Features to the Target",
    xlabel="Era",
    ax=ax,
)
ax.legend(["lowest std", "highest std"])
plt.show()
##
fig, ax = plt.subplots(figsize=(10, 5))

# plot the cumulative per era correlation of the feature with the highest vs lowest delta
per_era_corr[
    [feature_metrics["delta"].idxmin(), feature_metrics["delta"].idxmax()]
].cumsum().plot(
    figsize=(15, 5),
    title="Cumulative Correlation of Features to the Target",
    xlabel="Era",
    ax=ax,
)
plt.legend(["lowest delta", "highest delta"])
plt.show()

##

model = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=5,
    num_leaves=2**4 - 1,
    colsample_bytree=0.1,
)
# We've found the following "deep" parameters perform much better, but they require much more CPU and RAM
# model = lgb.LGBMRegressor(
#     n_estimators=30_000,
#     learning_rate=0.001,
#     max_depth=10,
#     num_leaves=2**10,
#     colsample_bytree=0.1
#     min_data_in_leaf=10000,
# )
model.fit(train[medium_features], train["target"])

##
joblib.dump(model, "lgb_regressor_model.lgb")
##
# Load the validation data, filtering for data_type == "validation"
validation = pd.read_parquet(
    "scripts/validation.parquet", columns=["era", "data_type", "target"] + medium_features
)
validation = validation[validation["data_type"] == "validation"]
del validation["data_type"]

# Downsample every 4th era to reduce memory usage and speedup validation (suggested for Colab free tier)
# Comment out the line below to use all the data
validation = validation[validation["era"].isin(validation["era"].unique()[::4])]

# Embargo overlapping eras from training data
last_train_era = int(train["era"].unique()[-1])
eras_to_embargo = [str(era).zfill(4) for era in [last_train_era + i for i in range(4)]]
validation = validation[~validation["era"].isin(eras_to_embargo)]

# Generate predictions against the medium feature set of the validation data
validation["prediction"] = model.predict(validation[medium_features])
##
fig, ax = plt.subplots(figsize=(10, 10))
# Compute the Peason correlation of the predictions with each of the
# serenity features of the medium feature set
feature_exposures = validation.groupby("era").apply(
    lambda d: d[med_serenity_feats].corrwith(d["prediction"])
)

# Plot the feature exposures as bar charts
feature_exposures.plot.bar(
    title="Feature Exposures",
    figsize=(16, 10),
    layout=(7, 5),
    xticks=[],
    subplots=True,
    sharex=False,
    legend=False,
    snap=False,
    ax=ax,
)
for ax in plt.gcf().axes:
    ax.set_xlabel("")
    ax.title.set_fontsize(10)
plt.tight_layout(pad=1.5)
plt.gcf().suptitle("Feature Exposures", fontsize=15)
plt.show()
##
fig, ax = plt.subplots(figsize=(10, 10))

# Plot the max feature exposure per era
max_feature_exposure = feature_exposures.max(axis=1)
max_feature_exposure.plot(
    title="Max Feature Exposure",
    kind="bar",
    figsize=(10, 5),
    xticks=[],
    snap=False,
    ax=ax,
)
plt.show()
# Mean max feature exposure across eras
print("Mean of max feature exposure", max_feature_exposure.mean())

##
# import neutralization from numerai-tools
from numerai_tools.scoring import neutralize

# Neutralize predictions per-era against features at different proportions
proportions = [0.25, 0.5, 0.75, 1.0]
for proportion in proportions:
    neutralized = (
        validation.groupby("era", group_keys=True)
        .apply(
            lambda d: neutralize(
                d[["prediction"]], d[med_serenity_feats], proportion=proportion
            )
        )
        .reset_index()
        .set_index("id")
    )
    validation[f"neutralized_{proportion*100:.0f}"] = neutralized["prediction"]

# Align the neutralized predictions with the validation data
prediction_cols = ["prediction"] + [f for f in validation.columns if "neutralized" in f]
validation[["era", "target"] + prediction_cols]

##
# Compute max feature exposure for each set of predictions
max_feature_exposures = pd.concat(
    [
        validation.groupby("era")
        .apply(lambda d: d[med_serenity_feats].corrwith(d[col]).abs().max())
        .rename(col)
        for col in prediction_cols
    ],
    axis=1,
)

# print mean feature exposure of each proportion
print("mean feature exposures:")
print(round(max_feature_exposures.mean(), 3))

fig, ax = plt.subplots(figsize=(10, 10))

# Plot max feature exposures
max_feature_exposures.plot.bar(
    title="Max Feature Exposures",
    figsize=(10, 5),
    xticks=[],
    snap=False,
    ax=ax,
)
plt.show()
##
# calculate per-era CORR for each set of predictions
correlations = validation.groupby("era").apply(
    lambda d: numerai_corr(d[prediction_cols], d["target"])
)

# calculate the cumulative corr across eras for each neutralization proportion
cumulative_correlations = correlations.cumsum().sort_index()

fig, ax = plt.subplots(figsize=(10, 10))
# Show the cumulative correlations
pd.DataFrame(cumulative_correlations).plot(
    title="Cumulative Correlation of Neutralized Predictions",
    figsize=(10, 6),
    xticks=[],
    ax=ax,
)
plt.show()
##
summary_metrics = {}
for col in prediction_cols:
    mean = correlations[col].mean()
    std = correlations[col].std(ddof=0)
    sharpe = mean / std
    rolling_max = cumulative_correlations[col].expanding(min_periods=1).max()
    max_drawdown = (rolling_max - cumulative_correlations[col]).max()
    summary_metrics[col] = {
        "mean": mean,
        "std": std,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }
pd.set_option("display.float_format", lambda x: "%f" % x)
pd.DataFrame(summary_metrics).T
##
# neutralize preds against each group
for group in groups:
    neutral_feature_subset = list(subgroups["medium"][group])
    neutralized = (
        validation.groupby("era", group_keys=True)
        .apply(lambda d: neutralize(d[["prediction"]], d[neutral_feature_subset]))
        .reset_index()
        .set_index("id")
    )
    validation[f"neutralized_{group}"] = neutralized["prediction"]

group_neutral_cols = ["prediction"] + [f"neutralized_{group}" for group in groups]
group_neutral_corr = validation.groupby("era").apply(
    lambda d: numerai_corr(d[group_neutral_cols], d["target"])
)
group_neutral_cumsum = group_neutral_corr.cumsum()
fig, ax = plt.subplots(figsize=(10, 10))

group_neutral_cumsum.plot(
    title="Cumulative Correlation of Neutralized Predictions",
    figsize=(10, 6),
    xticks=[],
    ax=ax,
)

plt.show()
##
from numerai_tools.scoring import correlation_contribution

# Download and join in the meta_model for the validation eras
napi.download_dataset(f"v4.3/meta_model.parquet", round_num=842)
validation["meta_model"] = pd.read_parquet(f"v4.3/meta_model.parquet")[
    "numerai_meta_model"
]

# Compute the per-era mmc between our predictions, the meta model, and the target values
per_era_mmc = (
    validation.dropna()
    .groupby("era")
    .apply(
        lambda x: correlation_contribution(
            x[group_neutral_cols], x["meta_model"], x["target"]
        )
    )
)

cumsum_mmc = per_era_mmc.cumsum()
fig, ax = plt.subplots(figsize=(10, 10))

cumsum_mmc.plot(
    title="Cumulative MMC of Neutralized Predictions", figsize=(10, 6), xticks=[], ax=ax
)
plt.show()
##
group_neutral_summary_metrics = {}
for col in group_neutral_cols:
    corr_mean = group_neutral_corr[col].mean()
    corr_std = group_neutral_corr[col].std()
    corr_sharpe = corr_mean / corr_std
    corr_rolling_max = group_neutral_cumsum[col].expanding(min_periods=1).max()
    corr_max_drawdown = (corr_rolling_max - group_neutral_cumsum[col]).max()
    mmc_mean = per_era_mmc[col].mean()
    mmc_std = per_era_mmc[col].std()
    mmc_sharpe = mmc_mean / mmc_std
    mmc_rolling_max = cumsum_mmc[col].expanding(min_periods=1).max()
    mmc_max_drawdown = (rolling_max - cumsum_mmc[col]).max()
    group_neutral_summary_metrics[col] = {
        "corr_mean": corr_mean,
        "mmc_mean": mmc_mean,
        "corr_std": corr_std,
        "mmc_std": mmc_std,
        "corr_sharpe": corr_sharpe,
        "mmc_sharpe": mmc_sharpe,
        "corr_max_drawdown": corr_max_drawdown,
        "mmc_max_drawdown": mmc_max_drawdown,
    }
pd.set_option("display.float_format", lambda x: "%f" % x)
pd.DataFrame(group_neutral_summary_metrics).T
##
# We copy this neutralization code here because Numerai's model upload framework
# does not currently include numerai-tools


def neutralize(
    df: pd.DataFrame,
    neutralizers: np.ndarray,
    proportion: float = 1.0,
) -> pd.DataFrame:
    """Neutralize each column of a given DataFrame by each feature in a given
    neutralizers DataFrame. Neutralization uses least-squares regression to
    find the orthogonal projection of each column onto the neutralizers, then
    subtracts the result from the original predictions.

    Arguments:
        df: pd.DataFrame - the data with columns to neutralize
        neutralizers: pd.DataFrame - the neutralizer data with features as columns
        proportion: float - the degree to which neutralization occurs

    Returns:
        pd.DataFrame - the neutralized data
    """
    assert not neutralizers.isna().any().any(), "Neutralizers contain NaNs"
    assert len(df.index) == len(neutralizers.index), "Indices don't match"
    assert (df.index == neutralizers.index).all(), "Indices don't match"
    df[df.columns[df.std() == 0]] = np.nan
    df_arr = df.values
    neutralizer_arr = neutralizers.values
    neutralizer_arr = np.hstack(
        # add a column of 1s to the neutralizer array in case neutralizer_arr is a single column
        (neutralizer_arr, np.array([1] * len(neutralizer_arr)).reshape(-1, 1))
    )
    inverse_neutralizers = np.linalg.pinv(neutralizer_arr, rcond=1e-6)
    adjustments = proportion * neutralizer_arr.dot(inverse_neutralizers.dot(df_arr))
    neutral = df_arr - adjustments
    return pd.DataFrame(neutral, index=df.index, columns=df.columns)


def predict_neutral(live_features: pd.DataFrame) -> pd.DataFrame:
    # make predictions using all features
    predictions = pd.DataFrame(
        model.predict(live_features[medium_features]),
        index=live_features.index,
        columns=["prediction"],
    )
    # neutralize predictions to a subset of features
    neutralized = neutralize(predictions, live_features[med_serenity_feats])
    return neutralized.rank(pct=True)


# Quick test
napi.download_dataset(f"{DATA_VERSION}/live.parquet")
live_features = pd.read_parquet(f"{DATA_VERSION}/live.parquet", columns=medium_features)
predict_neutral(live_features)
##
# Use the cloudpickle library to serialize your function and its dependencies

p = cloudpickle.dumps(predict_neutral)
with open("feature_neutralization.pkl", "wb") as f:
    f.write(p)
##
