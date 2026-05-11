from unittest.mock import patch

import pytest

import app as app_module
import config


@pytest.fixture
def client(tmp_path, dummy_df):
    """Flask test client with a fake model + scaler on disk."""
    # Build a real artifacts dir with stub joblibs so list_available_models works.
    from joblib import dump
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    features = [f for f in config.FEATURES if f != config.TARGET]
    X = dummy_df[features]
    y = dummy_df[config.TARGET]

    scaler = StandardScaler().fit(X)
    model = LinearRegression().fit(scaler.transform(X), y)

    dump(scaler, tmp_path / "scaler.joblib")
    dump(model, tmp_path / "random_forest.joblib")

    with patch.object(app_module.config, "ARTIFACTS_DIR", tmp_path):
        flask_app = app_module.create_app()
        flask_app.testing = True
        with flask_app.test_client() as c:
            yield c


def _valid_payload():
    return {
        "year": 2010, "age": 5, "beds": 3, "baths": 2,
        "home_size": 1800, "parcel_size": 6000, "pool": 0,
        "dist_cbd": 8000, "dist_lakes": 4000,
        "x_coord": 540000, "y_coord": 1500000,
    }


class TestHealthz:
    def test_returns_ok(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.get_json() == {"status": "ok"}


class TestIndex:
    def test_renders(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"House Price Predictor" in resp.data


class TestPredictEndpoint:
    def test_happy_path(self, client):
        resp = client.post("/api/predict", json=_valid_payload())
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["model"] == "random_forest"
        assert isinstance(body["predicted_price"], float)

    def test_missing_field_returns_400(self, client):
        bad = _valid_payload()
        bad.pop("beds")
        resp = client.post("/api/predict", json=bad)
        assert resp.status_code == 400
        assert "beds" in resp.get_json()["error"]

    def test_non_numeric_returns_400(self, client):
        bad = _valid_payload()
        bad["beds"] = "not-a-number"
        resp = client.post("/api/predict", json=bad)
        assert resp.status_code == 400

    def test_nan_returns_400(self, client):
        bad = _valid_payload()
        bad["beds"] = float("nan")
        resp = client.post("/api/predict", json=bad)
        assert resp.status_code == 400

    def test_inf_returns_400(self, client):
        bad = _valid_payload()
        bad["beds"] = float("inf")
        resp = client.post("/api/predict", json=bad)
        assert resp.status_code == 400

    def test_unknown_model_returns_503(self, client):
        payload = _valid_payload()
        payload["model"] = "totally_made_up"
        resp = client.post("/api/predict", json=payload)
        assert resp.status_code == 503

    def test_path_traversal_blocked(self, client):
        """A model name like '../etc/passwd' must not load arbitrary files."""
        payload = _valid_payload()
        payload["model"] = "../etc/passwd"
        resp = client.post("/api/predict", json=payload)
        assert resp.status_code == 503

    def test_empty_body_returns_400(self, client):
        resp = client.post("/api/predict", data="", content_type="application/json")
        assert resp.status_code == 400

    def test_non_string_model_returns_400(self, client):
        payload = _valid_payload()
        payload["model"] = 123
        resp = client.post("/api/predict", json=payload)
        assert resp.status_code == 400


class TestListAvailableModels:
    def test_missing_dir_returns_empty(self, tmp_path):
        with patch.object(app_module.config, "ARTIFACTS_DIR", tmp_path / "nope"):
            assert app_module.list_available_models() == []

    def test_excludes_scaler(self, tmp_path):
        (tmp_path / "scaler.joblib").touch()
        (tmp_path / "ridge.joblib").touch()
        with patch.object(app_module.config, "ARTIFACTS_DIR", tmp_path):
            assert app_module.list_available_models() == ["ridge"]
