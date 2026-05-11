from pathlib import Path

import config


class TestPaths:
    def test_base_dir_exists(self):
        assert config.BASE_DIR.exists()

    def test_data_path_is_path(self):
        assert isinstance(config.DATA_PATH, Path)

    def test_artifacts_path_is_path(self):
        assert isinstance(config.ARTIFACTS_DIR, Path)


class TestFeatures:
    def test_target_in_features(self):
        assert config.TARGET in config.FEATURES

    def test_features_match_dummy(self, dummy_df):
        assert list(dummy_df.columns) == config.FEATURES


class TestScalars:
    def test_test_size(self):
        assert 0.0 < config.TEST_SIZE < 1.0
        assert config.TEST_SIZE == 0.20

    def test_seed(self):
        assert config.SEED == 42

    def test_cv_folds(self):
        assert config.CV_FOLDS >= 2
