from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import TARGET


def split_data(
    dataframe: pd.DataFrame,
    test_ratio: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split a DataFrame into train/test feature matrices and target vectors.

    Raises:
        ValueError: if `dataframe` is None/empty, the target is missing,
                    or `test_ratio` is not in (0, 1).
    """
    if dataframe is None or dataframe.empty:
        raise ValueError("Cannot split: dataframe is empty or None")
    if TARGET not in dataframe.columns:
        raise ValueError(f"Target column {TARGET!r} not in dataframe")
    if not 0.0 < test_ratio < 1.0:
        raise ValueError(f"test_ratio must be in (0, 1), got {test_ratio}")

    X = dataframe.drop(TARGET, axis=1)
    y = dataframe[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_ratio,
        random_state=random_state,
    )
    return (
        cast(pd.DataFrame, X_train),
        cast(pd.DataFrame, X_test),
        cast(pd.Series, y_train),
        cast(pd.Series, y_test),
    )


def scale(
    X_train: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit a StandardScaler on X_train and transform both splits.

    Returns:
        (X_train_scaled, X_test_scaled, scaler) — the fitted scaler is returned
        so callers can persist it for inference.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
