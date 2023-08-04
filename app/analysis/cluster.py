##
import pandas as pd

##
train_df = pd.read_parquet("train.parquet").head(100)
##

feature_columns = [c for c in train_df.columns if "feature" in c]
print(f"Feature columns: {len(feature_columns)}")
target_columns = [c for c in train_df.columns if "target" in c]
print(f"Target columns {len(target_columns)}")
##
