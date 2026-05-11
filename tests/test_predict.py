from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import config
from prices import predict as predict_mod
from prices.predict import INPUT_FEATURES, _load_scaler, predict


@pytest.fixture
def X_new(dummy_df):
    """A few rows of features, no target column."""
    return dummy_df.drop(columns=[config.TARGET]).head(5)


def _patch_config(cfg, tmp_path):
    """Mirror the real config attributes onto a MagicMock."""
    cfg.ARTIFACTS_DIR = tmp_path
    cfg.FEATURES = config.FEATURES
    cfg.TARGET = config.TARGET
    cfg.DATA_PATH = config.DATA_PATH
    cfg.TEST_SIZE = config.TEST_SIZE
    cfg.SEED = config.SEED


# ---------- predict() ----------

def test_predict_returns_series_with_correct_index_and_name(tmp_path, X_new):
    fake_model = MagicMock()
    fake_model.predict.return_value = np.arange(len(X_new)) * 1000.0

    fake_scaler = MagicMock()
    fake_scaler.transform.side_effect = lambda x: x.values

    with patch.object(predict_mod, "config") as cfg, \
         patch.object(predict_mod, "load", return_value=fake_model) as mock_load, \
         patch.object(predict_mod, "_load_scaler", return_value=fake_scaler):

        _patch_config(cfg, tmp_path)
        model_path = tmp_path / "random_forest.joblib"
        model_path.touch()

        preds = predict("random_forest", X_new)

    assert isinstance(preds, pd.Series)
    assert preds.name == "predicted_price"
    assert list(preds.index) == list(X_new.index)
    assert list(preds) == list(np.arange(len(X_new)) * 1000.0)
    mock_load.assert_called_once_with(model_path)


def test_predict_raises_when_model_file_missing(tmp_path, X_new):
    with patch.object(predict_mod, "config") as cfg:
        _patch_config(cfg, tmp_path)
        with pytest.raises(FileNotFoundError):
            predict("does_not_exist", X_new)


def test_predict_raises_on_missing_features(tmp_path, X_new):
    bad = X_new.drop(columns=[INPUT_FEATURES[0]])
    with patch.object(predict_mod, "config") as cfg:
        _patch_config(cfg, tmp_path)
        (tmp_path / "random_forest.joblib").touch()
        with pytest.raises(ValueError, match="missing required features"):
            predict("random_forest", bad)


def test_predict_raises_on_empty_dataframe(tmp_path):
    empty = pd.DataFrame(columns=INPUT_FEATURES)
    with patch.object(predict_mod, "config") as cfg:
        _patch_config(cfg, tmp_path)
        (tmp_path / "random_forest.joblib").touch()
        with pytest.raises(ValueError, match="empty"):
            predict("random_forest", empty)


def test_predict_raises_on_blank_model_name(tmp_path, X_new):
    with patch.object(predict_mod, "config") as cfg:
        _patch_config(cfg, tmp_path)
        with pytest.raises(ValueError):
            predict("", X_new)


# ---------- _load_scaler() ----------

def test_load_scaler_uses_saved_artifact(tmp_path):
    sentinel = object()
    with patch.object(predict_mod, "config") as cfg, \
         patch.object(predict_mod, "load", return_value=sentinel) as mock_load:
        _patch_config(cfg, tmp_path)
        scaler_path = tmp_path / "scaler.joblib"
        scaler_path.touch()

        assert _load_scaler() is sentinel
        mock_load.assert_called_once_with(scaler_path)


def test_load_scaler_falls_back_to_refit(tmp_path, dummy_df):
    """When scaler.joblib is missing, refit from the training split."""
    with patch.object(predict_mod, "config") as cfg, \
         patch.object(predict_mod, "load_data", return_value=dummy_df):
        _patch_config(cfg, tmp_path)        # no scaler.joblib here
        cfg.DATA_PATH = tmp_path / "x.csv"

        scaler = _load_scaler()

    assert hasattr(scaler, "mean_")
    assert hasattr(scaler, "scale_")
