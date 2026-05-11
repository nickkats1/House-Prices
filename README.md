# house-prices

A small house-price regression playground:

- A `prices` Python package with a thin data → preprocess → train → predict pipeline.
- A registry of 8 sklearn / xgboost / lightgbm regressors with hyperparameter search.
- A Flask web UI + JSON API for interactive predictions.
- A six-step Jupyter notebook walkthrough showing the same pipeline cell-by-cell.

## Quickstart

```bash
python -m venv venv
source venv/bin/activate

# Reproducible install (recommended): pinned versions from the lockfile.
pip install -r requirements-dev.txt && pip install -e . --no-deps

# Or, unpinned (latest compatible versions of every dependency):
pip install -e ".[dev]"
```

The two lockfiles — `requirements.txt` (runtime only) and `requirements-dev.txt` (runtime + dev + notebook extras) — are generated with `pip-compile`. Regenerate after editing `pyproject.toml`:

```bash
pip-compile pyproject.toml -o requirements.txt
pip-compile --extra=dev pyproject.toml -o requirements-dev.txt
```

### Train

```bash
python -m prices.train.trainer                  # all 8 models
python -m prices.train.trainer --models ridge   # subset
python -m prices.train.trainer --out-dir ./artifacts
python -m prices.train.trainer --no-oof         # skip out-of-fold cross-val
```

This writes one `<model>.joblib` per model, plus `scaler.joblib` and a `summary.json` containing two blocks:

- `per_model` — best CV score, best params, and held-out test MSE/R² per model.
- `oof_best`  — out-of-fold MSE / RMSE / R² for the model with the best CV score, computed with `StratifiedKFold` over quantile bins of the target. Stratifying on quantile bins keeps the price distribution roughly balanced across folds — useful when the target is right-skewed.

### Serve the web UI

```bash
flask --app app run --port 8000
# or
gunicorn -b 0.0.0.0:8000 app:app
```

Then open <http://localhost:8000>.

### Predict via the API

```bash
curl -X POST http://localhost:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "ridge",
    "year": 2010, "age": 5, "beds": 3, "baths": 2,
    "home_size": 1800, "parcel_size": 6000, "pool": 0,
    "dist_cbd": 8000, "dist_lakes": 4000,
    "x_coord": 540000, "y_coord": 1500000
  }'
# -> {"model":"ridge","predicted_price":234567.0}
```

### Docker

```bash
docker compose up --build
```

The compose file mounts `./artifacts` and `./data` read-only into the container, so you train on the host and serve in the container.

## Project layout

```
.
├── app.py                          # Flask app (factory: create_app())
├── config.py                       # Paths, features, target, hyperparams
├── prices/
│   ├── data/
│   │   ├── ingestion.py            # load_data() — read + clean CSV
│   │   └── preprocessing.py        # split_data(), scale()
│   ├── model/registry.py           # MODELS: name -> (estimator, param_grid)
│   ├── train/trainer.py            # run_training() + CLI
│   └── predict.py                  # predict() + scaler-fallback
├── templates/index.html            # web UI
├── static/{app.js,styles.css}
├── tests/                          # pytest, 118 tests
├── notebooks/                      # see below — six-step walkthrough
├── Dockerfile
└── docker-compose.yml
```

## Notebooks

A six-step walkthrough — each step is its own folder so the pipeline reads top-to-bottom. The notebooks share intermediate state through `notebooks/_artifacts/` (gitignored).

| # | Folder | What it does |
|---|---|---|
| 1 | `notebooks/01_ingestion`     | Load CSV, drop NaNs / dups, write `clean.parquet`. |
| 2 | `notebooks/02_eda`           | Distributions, correlation heatmap, scatterplots vs. target. |
| 3 | `notebooks/03_preprocessing` | Train/test split + standardization, persist `scaler.joblib`. |
| 4 | `notebooks/04_training`      | LinearRegression baseline + Ridge tuned via `run_training`. |
| 5 | `notebooks/05_evaluation`    | MSE / RMSE / R², predicted-vs-actual, residual plots. |
| 6 | `notebooks/06_inference`     | Score new rows via `prices.predict.predict`. |

Each notebook re-uses the production code paths (`prices.data.*`, `prices.train.trainer`, `prices.predict`) so they don't drift from what the trainer and Flask app actually run.

The committed notebooks already include their executed outputs. Re-execute them in order with:

```bash
pip install -e ".[notebook]"
for nb in notebooks/0*/*.ipynb; do
  jupyter nbconvert --to notebook --execute --inplace "$nb"
done
```

## API

| Method | Path           | Description                                         |
|--------|----------------|-----------------------------------------------------|
| GET    | `/`            | Web UI form.                                        |
| GET    | `/healthz`     | Liveness probe — returns `{"status":"ok"}`.         |
| POST   | `/api/predict` | JSON in/out. `model` field optional (default = `random_forest`). All feature fields required. |

`POST /api/predict` returns:

| Status | When                                                                |
|--------|---------------------------------------------------------------------|
| 200    | Success.                                                            |
| 400    | Missing/non-numeric/NaN/Inf field, or non-string `model`.           |
| 503    | Model name not in the on-disk allowlist, or scaler is missing.      |
| 500    | Unexpected error during `model.predict` (logged server-side).       |

`model` is validated against the actual joblib files on disk — arbitrary path components like `"../etc/passwd"` are rejected.

## Development

```bash
pytest                     # 118 tests
ruff check .               # lint
mypy .                     # types
```

CI runs all three on every push and pull request — see `.github/workflows/ci.yml`.

## Configuration

`config.py` is the single source of truth:

| Name           | Default                          | Meaning                                  |
|----------------|----------------------------------|------------------------------------------|
| `BASE_DIR`     | repo root                        | resolved at import time                  |
| `DATA_PATH`    | `data/raw/dataset.csv`           | input CSV                                |
| `ARTIFACTS_DIR`| `artifacts/`                     | model + scaler joblibs land here         |
| `FEATURES`     | 12 columns incl. `price`         | full schema                              |
| `TARGET`       | `price`                          | regression target                        |
| `TEST_SIZE`    | `0.20`                           | held-out test fraction                   |
| `SEED`         | `42`                             | numpy/sklearn random state               |
| `CV_FOLDS`     | `4`                              | k-fold count for hyperparameter search   |

`FLASK_SECRET_KEY` and `FLASK_DEBUG=1` are read from the environment at app startup.
