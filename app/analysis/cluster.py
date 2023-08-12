##
import pandas as pd
import xgboost as xgb
from joblib import load
from matplotlib import pyplot as plt

##
train_df = pd.read_parquet("train.parquet").head(100)
##

feature_columns = [c for c in train_df.columns if "feature" in c]
print(f"Feature columns: {len(feature_columns)}")
target_columns = [c for c in train_df.columns if "target" in c]
print(f"Target columns {len(target_columns)}")
##
model_path = "model.lgb.fold_1.pkl"
model = load(model_path)
##
fig, ax = plt.subplots(figsize=(10, 20))
xgb.plot_importance(model, ax=ax, importance_type="gain")
plt.show()

##
