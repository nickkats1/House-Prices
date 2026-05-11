import pandas as pd
import pytest

from prices.data.ingestion import SchemaError, load_data


class TestLoadDataErrors:
    def test_raises_for_none_path(self):
        with pytest.raises(FileNotFoundError):
            load_data(file_path=None)

    def test_raises_for_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_data(file_path=tmp_path / "does_not_exist.csv")

    def test_raises_for_empty_file(self, tmp_path):
        empty = tmp_path / "empty.csv"
        empty.write_text("price,year\n")  # header only — no data rows
        with pytest.raises(ValueError):
            load_data(file_path=empty)


class TestLoadData:
    def test_loads_data(self, dummy_csv):
        df = load_data(file_path=dummy_csv)
        assert "price" in df.columns
        assert not df.empty

    def test_drops_duplicates(self, tmp_path, dummy_df):
        path = tmp_path / "dup.csv"
        head = dummy_df.iloc[:5]
        doubled = pd.concat([head, head], ignore_index=True)
        doubled.to_csv(path, index=False)

        df = load_data(file_path=path)
        assert len(df) == 5


class TestSchemaValidation:
    def test_raises_on_missing_column(self, tmp_path, dummy_df):
        path = tmp_path / "missing_col.csv"
        dummy_df.drop(columns=["beds"]).to_csv(path, index=False)
        with pytest.raises(SchemaError, match="missing required columns"):
            load_data(file_path=path)

    def test_raises_on_non_numeric_column(self, tmp_path, dummy_df):
        path = tmp_path / "bad_dtype.csv"
        bad = dummy_df.copy()
        bad["beds"] = "three"  # string in a numeric column
        bad.to_csv(path, index=False)
        with pytest.raises(SchemaError, match="non-numeric"):
            load_data(file_path=path)

    def test_extra_columns_are_dropped(self, tmp_path, dummy_df):
        path = tmp_path / "extra.csv"
        extra = dummy_df.copy()
        extra["junk"] = "ignore_me"
        extra.to_csv(path, index=False)

        df = load_data(file_path=path)
        assert "junk" not in df.columns
