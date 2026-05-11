from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from pandas.api import types as ptypes

import config

logger = logging.getLogger(__name__)


class SchemaError(ValueError):
    """Raised when a loaded CSV does not match the expected schema."""


def _validate_schema(df: pd.DataFrame, source: Path | str) -> None:
    """Verify every configured feature is present and numeric."""
    missing = [c for c in config.FEATURES if c not in df.columns]
    if missing:
        raise SchemaError(
            f"{source}: missing required columns {missing}. "
            f"Got columns: {list(df.columns)}"
        )

    non_numeric = [
        c for c in config.FEATURES if not ptypes.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        raise SchemaError(
            f"{source}: non-numeric columns {non_numeric}. "
            "All configured features must be numeric."
        )


def load_data(file_path: str | Path | None) -> pd.DataFrame:
    """Load a CSV from disk, validate its schema, drop NaNs / duplicates.

    Args:
        file_path: path to a CSV file.

    Returns:
        Cleaned DataFrame with exactly the columns in `config.FEATURES`,
        in that order.

    Raises:
        FileNotFoundError: if `file_path` is None or does not exist.
        SchemaError:       if required columns are missing or non-numeric.
        ValueError:        if the file is empty after cleaning.
    """
    if file_path is None:
        raise FileNotFoundError("file_path must be provided, got None")

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")

    df = pd.read_csv(path)
    _validate_schema(df, source=path)

    df = df[config.FEATURES]
    before = len(df)
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d rows (NaNs/duplicates) from %s", dropped, path)

    if df.empty:
        raise ValueError(f"Dataset is empty after cleaning: {path}")

    return df
