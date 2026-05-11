import json
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from prices.train import trainer as trainer_mod
from prices.train.trainer import (
    _grid_size,
    _make_search,
    _quantile_bins,
    run_training,
    stratified_oof_predict,
)

# ---------- _grid_size ----------

@pytest.mark.parametrize(
    "grid, expected",
    [
        ({}, 0),
        ({"a": [1, 2, 3]}, 3),
        ({"a": [1, 2], "b": [10, 20, 30]}, 6),
    ],
)
def test_grid_size(grid, expected):
    assert _grid_size(grid) == expected


# ---------- _make_search ----------

def test_make_search_picks_grid_for_small_grids():
    from sklearn.model_selection import GridSearchCV
    search = _make_search(LinearRegression(), {"fit_intercept": [True, False]})
    assert isinstance(search, GridSearchCV)


def test_make_search_picks_random_for_large_grids():
    from sklearn.model_selection import RandomizedSearchCV
    big_grid = {"a": list(range(20)), "b": list(range(20))}  # 400 > 200
    search = _make_search(LinearRegression(), big_grid, threshold=200)
    assert isinstance(search, RandomizedSearchCV)


# ---------- stratified OOF ----------

def test_quantile_bins_balanced():
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 1, size=200)
    bins = _quantile_bins(y, n_bins=5)
    counts = np.bincount(bins)
    assert len(counts) == 5
    # Quantile bins should be approximately balanced.
    assert counts.max() - counts.min() <= 2


def test_stratified_oof_returns_one_pred_per_row(dummy_df):
    X = dummy_df.drop(columns=["price"]).to_numpy(dtype=float)
    y = dummy_df["price"].to_numpy(dtype=float)
    oof = stratified_oof_predict(LinearRegression(), X, y, n_splits=3, n_bins=4)
    assert oof.shape == y.shape
    assert np.isfinite(oof).all()


# ---------- run_training ----------

def test_run_training_unknown_model_raises(tmp_path, dummy_csv):
    with patch.object(trainer_mod, "config") as cfg:
        cfg.DATA_PATH = dummy_csv
        cfg.ARTIFACTS_DIR = tmp_path
        cfg.TEST_SIZE = 0.2
        cfg.SEED = 42
        cfg.CV_FOLDS = 4
        with pytest.raises(KeyError, match="Unknown model"):
            run_training(out_dir=tmp_path, models=["not_a_real_model"])


def test_run_training_writes_artifacts_and_summary(tmp_path, dummy_csv):
    """End-to-end on one tiny model with a trivial grid."""
    tiny_registry = {
        "linear_regression": (
            LinearRegression(),
            {"fit_intercept": [True, False]},
        ),
    }

    with patch.object(trainer_mod, "MODELS", tiny_registry), \
         patch.object(trainer_mod, "config") as cfg:
        cfg.DATA_PATH = dummy_csv
        cfg.ARTIFACTS_DIR = tmp_path
        cfg.TEST_SIZE = 0.2
        cfg.SEED = 42
        cfg.CV_FOLDS = 4

        summary = run_training(out_dir=tmp_path)

    # Returned dict shape
    assert set(summary.keys()) == {"metadata", "per_model", "oof_best"}
    meta = summary["metadata"]
    assert meta["seed"] == 42
    assert meta["test_size"] == 0.2
    assert meta["n_train_rows"] > 0
    assert meta["n_test_rows"] > 0
    assert "trained_at" in meta
    assert "package_version" in meta
    entry = summary["per_model"]["linear_regression"]
    assert set(entry) == {"best_params", "cv_score", "test_mse", "test_r2"}
    assert isinstance(entry["cv_score"], float)
    assert isinstance(entry["test_mse"], float)
    assert entry["test_mse"] >= 0.0
    assert isinstance(entry["test_r2"], float)

    # OOF block
    oof = summary["oof_best"]
    assert oof["model"] == "linear_regression"
    assert oof["oof_mse"] >= 0.0
    assert oof["oof_rmse"] >= 0.0
    assert isinstance(oof["oof_r2"], float)

    # Files written
    assert (tmp_path / "linear_regression.joblib").exists()
    summary_path = tmp_path / "summary.json"
    assert summary_path.exists()
    on_disk = json.loads(summary_path.read_text())
    assert "per_model" in on_disk
    assert "oof_best" in on_disk


def test_run_training_oof_disabled(tmp_path, dummy_csv):
    tiny_registry = {
        "linear_regression": (LinearRegression(), {"fit_intercept": [True, False]}),
    }
    with patch.object(trainer_mod, "MODELS", tiny_registry), \
         patch.object(trainer_mod, "config") as cfg:
        cfg.DATA_PATH = dummy_csv
        cfg.ARTIFACTS_DIR = tmp_path
        cfg.TEST_SIZE = 0.2
        cfg.SEED = 42
        cfg.CV_FOLDS = 4

        summary = run_training(out_dir=tmp_path, oof=False)

    assert "per_model" in summary
    assert "metadata" in summary
    assert "oof_best" not in summary
