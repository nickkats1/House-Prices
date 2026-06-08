from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
from datetime import UTC, datetime
from math import prod
from pathlib import Path
from typing import Any

import numpy as np
from joblib import dump
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)

import config
import prices
from prices.data.ingestion import load_data
from prices.data.preprocessing import scale, split_data
from prices.model.registry import MODELS

logger = logging.getLogger(__name__)

# Leave a couple of cores free so the system stays responsive.
N_CORES = max(1, (os.cpu_count() or 4) - 2)

# RandomizedSearchCV kicks in once a grid has more combinations than this.
GRID_THRESHOLD = 200
RANDOM_ITER = 60

# Quantile bins used to stratify the regression target for OOF CV.
OOF_BINS = 10
OOF_SPLITS = 5


def _grid_size(grid: dict[str, list[Any]]) -> int:
    """Total number of combinations in a parameter grid."""
    return prod(len(v) for v in grid.values()) if grid else 0


def _safe_cv(n_samples: int) -> int:
    """Cap CV folds so each fold has at least 2 samples."""
    return max(2, min(config.CV_FOLDS, n_samples // 2))


def _make_search(
    estimator: BaseEstimator,
    param_grid: dict[str, list[Any]],
    threshold: int = GRID_THRESHOLD,
    n_iter: int = RANDOM_ITER,
    n_samples: int | None = None,
) -> GridSearchCV | RandomizedSearchCV:
    """GridSearchCV for small grids, RandomizedSearchCV for large ones."""
    cv = _safe_cv(n_samples) if n_samples is not None else config.CV_FOLDS
    common: dict[str, Any] = dict(
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=N_CORES,
        verbose=1,
        refit=True,
        error_score="raise",
    )
    if _grid_size(param_grid) > threshold:
        return RandomizedSearchCV(
            estimator,
            param_grid,
            n_iter=n_iter,
            random_state=config.SEED,
            **common,
        )
    return GridSearchCV(estimator, param_grid, **common)


def _quantile_bins(y: np.ndarray, n_bins: int) -> np.ndarray:
    """Map a continuous target to n_bins integer strata via quantiles.

    Falls back gracefully when the target has < n_bins unique values.
    """
    n_bins = max(2, min(n_bins, len(np.unique(y))))
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(y, quantiles))
    binned: np.ndarray = np.clip(
        np.digitize(y, edges[1:-1], right=True), 0, len(edges) - 2
    )
    return binned


def stratified_oof_predict(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = OOF_SPLITS,
    n_bins: int = OOF_BINS,
    random_state: int = config.SEED,
) -> np.ndarray:
    """Out-of-fold predictions using StratifiedKFold on quantile-binned y.

    Stratifying on quantile bins keeps the price distribution roughly
    balanced across folds — useful when the target is skewed.
    """
    strata = _quantile_bins(y, n_bins=n_bins)
    n_splits = max(2, min(n_splits, np.bincount(strata).min()))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof = np.empty_like(y, dtype=float)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, strata), start=1):
        fold_estimator = clone(estimator)
        fold_estimator.fit(X[train_idx], y[train_idx])
        oof[val_idx] = fold_estimator.predict(X[val_idx])
        logger.debug("OOF fold %d/%d done", fold, n_splits)

    return oof


def _pick_best(results: dict[str, dict[str, Any]]) -> str:
    """Return the model name with the best CV score (least-negative MSE)."""
    return max(results, key=lambda name: results[name]["cv_score"])


def _git_sha() -> str | None:
    """Current git commit SHA, or None outside a working tree."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def _run_metadata(n_train: int, n_test: int) -> dict[str, Any]:
    return {
        "package_version": prices.__version__,
        "git_sha": _git_sha(),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "n_train_rows": n_train,
        "n_test_rows": n_test,
        "seed": config.SEED,
        "test_size": config.TEST_SIZE,
        "cv_folds": config.CV_FOLDS,
    }


def run_training(
    out_dir: Path | str = config.ARTIFACTS_DIR,
    models: list[str] | None = None,
    oof: bool = True,
) -> dict[str, Any]:
    """Train (and tune) the requested models, persist artifacts.

    After tuning, the model with the best CV score is re-evaluated with
    StratifiedKFold OOF predictions on the *training* set (folds stratified
    on quantile bins of the target) and that score is recorded in
    `summary.json` under `oof_best`.

    Raises:
        KeyError: if any requested model is not in the registry.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(file_path=config.DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(
        df,
        test_ratio=config.TEST_SIZE,
        random_state=config.SEED,
    )
    X_train_scaled, X_test_scaled, scaler = scale(X_train, X_test)

    dump(scaler, out_dir / "scaler.joblib")

    if models is None:
        items = list(MODELS.items())
    else:
        unknown = set(models) - set(MODELS.keys())
        if unknown:
            raise KeyError(
                f"Unknown model(s): {sorted(unknown)}. "
                f"Available: {sorted(MODELS.keys())}"
            )
        items = [(name, MODELS[name]) for name in models]

    per_model: dict[str, dict[str, Any]] = {}
    best_estimators: dict[str, Any] = {}
    for name, (estimator, param_grid) in items:
        logger.info("Training %s", name)
        search = _make_search(estimator, param_grid, n_samples=len(X_train_scaled))
        search.fit(X_train_scaled, y_train)

        best = search.best_estimator_
        preds = best.predict(X_test_scaled)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        dump(best, out_dir / f"{name}.joblib")
        best_estimators[name] = best
        per_model[name] = {
            "best_params": search.best_params_,
            "cv_score": float(search.best_score_),
            "test_mse": float(mse),
            "test_r2": float(r2),
        }
        logger.info("%s — test MSE=%.2f, R2=%.4f", name, mse, r2)

    summary: dict[str, Any] = {
        "metadata": _run_metadata(n_train=len(X_train_scaled), n_test=len(X_test_scaled)),
        "per_model": per_model,
    }

    if oof and per_model:
        best_name = _pick_best(per_model)
        logger.info("Best by CV score: %s — running stratified OOF CV", best_name)
        y_train_arr = np.asarray(y_train)
        oof_preds = stratified_oof_predict(
            best_estimators[best_name],
            X_train_scaled,
            y_train_arr,
        )
        oof_mse = float(mean_squared_error(y_train_arr, oof_preds))
        oof_r2 = float(r2_score(y_train_arr, oof_preds))
        summary["oof_best"] = {
            "model": best_name,
            "n_splits": OOF_SPLITS,
            "n_bins": OOF_BINS,
            "oof_mse": oof_mse,
            "oof_rmse": float(np.sqrt(oof_mse)),
            "oof_r2": oof_r2,
        }
        logger.info(
            "%s OOF — MSE=%.2f, RMSE=%.2f, R2=%.4f",
            best_name, oof_mse, np.sqrt(oof_mse), oof_r2,
        )

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train house-price models.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=config.ARTIFACTS_DIR,
        help="Directory to write artifacts to.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        choices=sorted(MODELS.keys()),
        help="Subset of models to train. Defaults to all.",
    )
    parser.add_argument(
        "--no-oof",
        action="store_true",
        help="Skip stratified OOF cross-validation of the best model.",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_arg_parser().parse_args()
    run_training(out_dir=args.out_dir, models=args.models, oof=not args.no_oof)
