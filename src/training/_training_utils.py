import pandas as pd
import numpy as np
from loguru import logger
from src.label_engineering import compute_dynamic_threshold

def create_labels(close: pd.Series, horizon: int, threshold: pd.Series) -> pd.Series:
    """
    Create binary labels (0/1/NaN) based on future returns and dynamic threshold.
    """
    future_return = close.shift(-horizon) / close - 1
    labels = pd.Series(np.nan, index=close.index)
    labels[future_return > threshold] = 1
    labels[future_return < -threshold] = 0
    return labels

def select_multiplier(
    close_train: pd.Series,
    window: int,
    init_multiplier: float,
    horizon: int,
    max_nan_ratio: float,
    min_multiplier: float,
    multiplier_step: float = 0.05
) -> float:
    """
    Select the optimal threshold multiplier to keep the NaN (neutral) ratio below max_nan_ratio.
    This was moved from label_engineering to be executed exclusively inside Walk-Forward folds.
    """
    multiplier = init_multiplier
    total = len(close_train)
    if total == 0:
        return multiplier

    while multiplier > min_multiplier:
        threshold = compute_dynamic_threshold(close_train, window, multiplier)
        labels = create_labels(close_train, horizon, threshold)
        
        nan_ratio = labels.isna().sum() / total
        if nan_ratio <= max_nan_ratio:
            break
            
        new_mult = round(multiplier - multiplier_step, 2)
        logger.warning(
            f"  T+{horizon} train NaN ratio {nan_ratio*100:.1f}% > "
            f"{max_nan_ratio*100:.0f}%, reducing {multiplier:.2f} -> {new_mult:.2f}"
        )
        multiplier = new_mult
        
    return multiplier
