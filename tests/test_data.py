import pandas as pd
import pytest

from adp_forecast.data import get_series, list_categories, load_data


def _sample_df():
    rows = [
        ("2024-01-01", "National", "U.S.", "M", 100),
        ("2024-01-01", "Census Divisions", "Pacific", "M", 10),
        ("2024-01-05", "National", "U.S.", "W", 999),
        ("2024-02-01", "National", "U.S.", "M", 110),
        ("2024-01-01", "Industry", "Construction", "M", 20),
        ("2024-02-01", "Industry", "Construction", "M", 22),
        ("2024-01-01", "Establishment Size", "1-19 employees", "M", 30),
    ]
    df = pd.DataFrame(rows, columns=["date", "agg_RIS", "category", "timestep", "NER_SA"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_get_series_defaults_to_national_monthly():
    series = get_series(_sample_df())
    assert list(series.items()) == [
        (pd.Timestamp("2024-01-01"), 100),
        (pd.Timestamp("2024-02-01"), 110),
    ]


def test_get_series_filters_segment():
    series = get_series(_sample_df(), agg_RIS="Industry", category="Construction")
    assert list(series.values) == [20, 22]


def test_get_series_excludes_weekly_rows():
    series = get_series(_sample_df())
    assert pd.Timestamp("2024-01-05") not in series.index


def test_get_series_can_return_weekly_rows():
    series = get_series(_sample_df(), timestep="W")
    assert list(series.items()) == [(pd.Timestamp("2024-01-05"), 999)]


def test_list_categories_returns_sorted_unique_categories():
    df = _sample_df()
    assert list_categories(df, "Industry") == ["Construction"]
    assert list_categories(df, "Establishment Size") == ["1-19 employees"]


def test_load_data_rejects_missing_required_columns(tmp_path):
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["2024-01-01"], "NER_SA": [100]}).to_csv(csv_path, index=False)
    with pytest.raises(ValueError):
        load_data(csv_path)
