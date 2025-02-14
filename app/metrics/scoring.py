import numpy as np
import pandas as pd
from numerai_tools.scoring import numerai_corr


def calculate_sharpe_ratio(corr: pd.Series) -> float:
    return corr.mean() / corr.std(ddof=0)


def calculate_max_drawdown(corr: pd.Series) -> float:
    return -(corr.cumsum().expanding(min_periods=1).max() - corr.cumsum()).max()


def calculate_delta(corr: pd.Series, eras: np.ndarray[float]) -> float:
    halfway_era = len(eras) // 2
    corr_mean_first_half = corr.loc[eras[:halfway_era]].mean()
    corr_mean_second_half = corr.loc[eras[halfway_era:]].mean()
    return abs(corr_mean_first_half - corr_mean_second_half)
