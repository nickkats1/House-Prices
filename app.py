from __future__ import annotations

import logging
import math
import os
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request
from joblib import load

import config

load_dotenv()

logger = logging.getLogger(__name__)

ResponseT = Response | tuple[Response, int] | str

DEFAULT_MODEL = "random_forest"
SCALER_FILENAME = "scaler.joblib"


def _input_features() -> list[str]:
    return [f for f in config.FEATURES if f != config.TARGET]


def list_available_models() -> list[str]:
    """Names of trained models present on disk (excluding the scaler)."""
    if not config.ARTIFACTS_DIR.exists():
        return []
    return sorted(
        p.stem
        for p in config.ARTIFACTS_DIR.glob("*.joblib")
        if p.stem != "scaler"
    )


def _load_artifacts(model_name: str) -> tuple[Any, Any]:
    """Load (model, scaler) by name, validating against the allowlist."""
    available = list_available_models()
    if model_name not in available:
        raise FileNotFoundError(
            f"Unknown model {model_name!r}. Available: {available or 'none'}"
        )

    scaler_path = config.ARTIFACTS_DIR / SCALER_FILENAME
    if not scaler_path.exists():
        raise FileNotFoundError(
            "Missing scaler.joblib. Train first: `python -m prices.train.trainer`"
        )

    model_path = config.ARTIFACTS_DIR / f"{model_name}.joblib"
    return load(model_path), load(scaler_path)


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

    @app.get("/")
    def index() -> str:
        available = list_available_models()
        default: str | None
        if DEFAULT_MODEL in available:
            default = DEFAULT_MODEL
        else:
            default = available[0] if available else None

        return render_template(
            "index.html",
            features=_input_features(),
            models=available,
            default_model=default,
            no_artifacts=not available,
        )

    @app.get("/healthz")
    def healthz() -> ResponseT:
        return jsonify({"status": "ok"}), 200

    @app.post("/api/predict")
    def predict() -> ResponseT:
        payload = request.get_json(silent=True) or {}

        features = _input_features()
        model_name = payload.get("model") or DEFAULT_MODEL
        if not isinstance(model_name, str):
            return jsonify({"error": "Field 'model' must be a string."}), 400

        try:
            row = {}
            for feature in features:
                if feature not in payload:
                    raise KeyError(feature)
                value = float(payload[feature])
                if not math.isfinite(value):
                    raise ValueError(f"{feature} must be finite")
                row[feature] = value
        except KeyError as exc:
            return jsonify({"error": f"Missing field: {exc.args[0]}"}), 400
        except (TypeError, ValueError) as exc:
            return jsonify({"error": f"Invalid input: {exc}"}), 400

        try:
            model, scaler = _load_artifacts(model_name)
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 503

        try:
            X = pd.DataFrame([row], columns=features)
            prediction = float(model.predict(scaler.transform(X))[0])
        except Exception:
            logger.exception("Prediction failed for model=%s", model_name)
            return jsonify({"error": "Prediction failed."}), 500

        return jsonify({"model": model_name, "predicted_price": prediction})

    return app


app = create_app()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app.run(host="0.0.0.0", port=8000, debug=os.environ.get("FLASK_DEBUG") == "1")
