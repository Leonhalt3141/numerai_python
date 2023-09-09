import pandas as pd


class FeatureSelection:
    @staticmethod
    def select_features_by_threshold(target_name: str, threshold: float = 0.01):
        df = pd.read_csv(f"MDA_result_{target_name}.csv", index_col=0)
        return df.loc[df["mean_abs"] > threshold].index.values

    @staticmethod
    def select_features_by_size(target_name: str, feature_size: int):
        df = pd.read_csv(f"MDA_result_{target_name}.csv", index_col=0)
        return df.index.values[:feature_size]
