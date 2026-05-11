from pathlib import Path

# --- Main Paths ---
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_PATH: Path = BASE_DIR / "data" / "raw" / "dataset.csv"

# --- Artifacts Path ---
ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"

# --- Features ---
FEATURES: list[str] = [
    "price", "year", "age",
    "beds", "baths", "home_size",
    "parcel_size", "pool", "dist_cbd",
    "dist_lakes", "x_coord", "y_coord",
]

# --- Target ---
TARGET: str = "price"

# --- Test Size ---
TEST_SIZE: float = 0.20

# --- Seed ---
SEED: int = 42

# --- CV folds ---
CV_FOLDS: int = 4
