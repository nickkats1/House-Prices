import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

import config
from prices.data.preprocessing import scale, split_data


class TestSplitData:
    def test_data_split(self, dummy_df):
        X_train, X_test, y_train, y_test = split_data(
            dummy_df,
            test_ratio=config.TEST_SIZE,
            random_state=config.SEED,
        )

        assert len(X_train) > len(X_test)
        assert len(y_train) > len(y_test)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert len(X_train) + len(X_test) == len(dummy_df)
        assert config.TARGET not in X_train.columns
        assert config.TARGET not in X_test.columns
        assert len(X_test) == int(len(dummy_df) * config.TEST_SIZE)

    def test_raises_on_empty_df(self):
        with pytest.raises(ValueError):
            split_data(pd.DataFrame(), test_ratio=0.2, random_state=0)

    def test_raises_on_none(self):
        with pytest.raises(ValueError):
            split_data(None, test_ratio=0.2, random_state=0)

    def test_raises_on_missing_target(self, dummy_df):
        no_target = dummy_df.drop(columns=[config.TARGET])
        with pytest.raises(ValueError):
            split_data(no_target, test_ratio=0.2, random_state=0)

    @pytest.mark.parametrize("ratio", [0.0, 1.0, -0.1, 1.5])
    def test_raises_on_invalid_ratio(self, dummy_df, ratio):
        with pytest.raises(ValueError):
            split_data(dummy_df, test_ratio=ratio, random_state=0)


class TestScale:
    def test_scale(self, dummy_df):
        X_train, X_test, _, _ = split_data(
            dummy_df,
            test_ratio=config.TEST_SIZE,
            random_state=config.SEED,
        )

        X_train_scaled, X_test_scaled, scaler = scale(X_train, X_test)

        assert isinstance(scaler, StandardScaler)
        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape
        assert np.allclose(X_train_scaled.mean(axis=0), 0, atol=1e-7)
        assert np.allclose(X_train_scaled.std(axis=0), 1, atol=1e-7)
