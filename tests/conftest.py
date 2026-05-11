import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def dummy_df():
    """dummy datset for testing all modules"""
    np.random.seed(42)
    n = 100

    return pd.DataFrame({
        "price":       np.random.randint(80_000, 600_000, n),
        "year":        np.random.randint(1990, 2024, n),
        "age":         np.random.randint(0, 40, n),
        "beds":        np.random.randint(2, 6, n),
        "baths":       np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], n),
        "home_size":   np.random.randint(800, 4000, n),
        "parcel_size": np.random.randint(4000, 15000, n),
        "pool":        np.random.choice([0, 1], n),
        "dist_cbd":    np.round(np.random.uniform(1000, 25000, n), 2),
        "dist_lakes":  np.round(np.random.uniform(100, 10000, n), 2),
        "x_coord":     np.round(np.random.uniform(490000, 600000, n), 1),
        "y_coord":     np.round(np.random.uniform(1450000, 1560000, n), 1),
    })



@pytest.fixture
def dummy_csv(tmp_path, dummy_df):
    path = tmp_path / "dataset.csv"
    dummy_df.to_csv(path, index=False)
    return path