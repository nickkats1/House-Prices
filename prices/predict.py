from __future__ import annotations

import logging

import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler

import config
from prices.data.ingestion import load_data
from prices.data.preprocessing import split_data

logger = logging.getLogger(__name__)


def _input_features() -> list[str]:
    return [f for f in config.FEATURES if f != config.TARGET]


# Kept for backwards compatibility with existing imports/tests.
INPUT_FEATURES = _input_features()


def _load_scaler() -> StandardScaler:
    """Load the saved scaler, or refit it from the training split as a fallback."""
    scaler_path = config.ARTIFACTS_DIR / "scaler.joblib"
    if scaler_path.exists():
        return load(scaler_path)

    logger.warning(
        "scaler.joblib not found at %s — refitting from raw data as a fallback. "
        "Predictions may not match the training distribution. Run "
        "`python -m prices.train.trainer` to regenerate the scaler.",
        scaler_path,
    )
    df = load_data(file_path=config.DATA_PATH)
    X_train, *_ = split_data(
        df, test_ratio=config.TEST_SIZE, random_state=config.SEED
    )
    return StandardScaler().fit(X_train)


def predict(model_name: str, X_new: pd.DataFrame) -> pd.Series:
    """Predict house prices for new rows using a saved model.

    Raises:
        FileNotFoundError: if the named model artifact does not exist.
        ValueError:        if `X_new` is empty or missing required features.
    """
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model_name must be a non-empty string")

    if X_new is None or len(X_new) == 0:
        raise ValueError("X_new is empty; nothing to predict")

    model_path = config.ARTIFACTS_DIR / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No model at {model_path}. Run `python -m prices.train.trainer` first."
        )

    features = _input_features()
    missing = set(features) - set(X_new.columns)
    if missing:
        raise ValueError(f"X_new is missing required features: {sorted(missing)}")

    model = load(model_path)
    scaler = _load_scaler()

    X = X_new[features]
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)

    return pd.Series(preds, index=X_new.index, name="predicted_price")
