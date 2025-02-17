import json

import joblib
import numpy as np
import pandas as pd

model = joblib.load("lgb_regressor_model.lgb")
##

validation_df = pd.read_parquet("data/validation.parquet")
##
features = json.load(open("data/features.json"))
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


medium_features = feature_sets["medium"]
med_serenity_feats = list(subgroups["medium"]["serenity"])


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


##
live_features = pd.read_parquet("../data/live.parquet", columns=medium_features)

##
predict_neutral(live_features)


##
