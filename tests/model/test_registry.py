import pytest
from sklearn.base import BaseEstimator

from prices.model.registry import MODELS

EXPECTED_MODELS = {
    "linear_regression",
    "ridge",
    "lasso",
    "decision_tree",
    "knn",
    "random_forest",
    "xgboost",
    "lightgbm",
}


class TestRegistryStructure:
    """Test the overall shape of the MODELS registry."""

    def test_is_dict(self):
        assert isinstance(MODELS, dict)

    def test_not_empty(self):
        assert len(MODELS) > 0

    def test_expected_keys_present(self):
        assert set(MODELS.keys()) == EXPECTED_MODELS

    def test_keys_are_strings(self):
        assert all(isinstance(k, str) for k in MODELS.keys())

    def test_values_are_tuples_of_two(self):
        for name, value in MODELS.items():
            assert isinstance(value, tuple), f"{name} value should be a tuple."
            assert len(value) == 2, f"{name} tuple should have (estimator, params)."


class TestRegistryEstimators:
    """Test the estimator part of each entry."""

    @pytest.mark.parametrize("name", list(EXPECTED_MODELS))
    def test_is_sklearn_compatible_estimator(self, name):
        estimator, _ = MODELS[name]
        assert isinstance(estimator, BaseEstimator), (
            f"{name} estimator must inherit from sklearn BaseEstimator."
        )

    @pytest.mark.parametrize("name", list(EXPECTED_MODELS))
    def test_has_fit_and_predict(self, name):
        estimator, _ = MODELS[name]
        assert hasattr(estimator, "fit"), f"{name} must have a .fit method."
        assert hasattr(estimator, "predict"), f"{name} must have a .predict method."


class TestRegistryParamGrids:
    """Test the param grid part of each entry."""

    @pytest.mark.parametrize("name", list(EXPECTED_MODELS))
    def test_params_is_dict(self, name):
        _, params = MODELS[name]
        assert isinstance(params, dict), f"{name} params should be a dict."

    @pytest.mark.parametrize("name", list(EXPECTED_MODELS))
    def test_params_not_empty(self, name):
        _, params = MODELS[name]
        assert len(params) > 0, f"{name} param grid should not be empty."

    @pytest.mark.parametrize("name", list(EXPECTED_MODELS))
    def test_param_keys_are_strings(self, name):
        _, params = MODELS[name]
        assert all(isinstance(k, str) for k in params.keys()), (
            f"{name} param keys should be strings."
        )

    @pytest.mark.parametrize("name", list(EXPECTED_MODELS))
    def test_param_values_are_lists(self, name):
        _, params = MODELS[name]
        assert all(isinstance(v, list) and len(v) > 0 for v in params.values()), (
            f"{name} param values should be non-empty lists."
        )

    @pytest.mark.parametrize("name", list(EXPECTED_MODELS))
    def test_param_keys_match_estimator(self, name):
        """Every param key must be a valid hyperparameter on the estimator."""
        estimator, params = MODELS[name]
        valid = set(estimator.get_params().keys())
        invalid = set(params.keys()) - valid
        assert not invalid, f"{name} has invalid params: {invalid}"


class TestRegistryFitsOnDummyData:
    """Smoke test: each estimator can fit on the dummy dataset."""

    @pytest.mark.parametrize("name", list(EXPECTED_MODELS))
    def test_fit_predict(self, name, dummy_df):
        from config import TARGET

        estimator, _ = MODELS[name]
        X = dummy_df.drop(columns=[TARGET])
        y = dummy_df[TARGET]

        estimator.fit(X, y)
        preds = estimator.predict(X)
        assert len(preds) == len(y)