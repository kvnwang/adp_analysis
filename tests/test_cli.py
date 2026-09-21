from typer.testing import CliRunner

from adp_forecast.cli import app

runner = CliRunner()


def test_history_default_shows_national_table():
    result = runner.invoke(app, ["history", "--limit", "3"])
    assert result.exit_code == 0
    assert "ADP Employment History" in result.output
    assert "Total Employment" in result.output
    assert "Jobs Added" in result.output


def test_history_by_industry_shows_each_category():
    result = runner.invoke(app, ["history", "--by", "industry", "--limit", "1"])
    assert result.exit_code == 0
    assert "Construction" in result.output
    assert "Information" in result.output


def test_history_by_size_weekly_runs():
    result = runner.invoke(app, ["history", "--by", "size", "--freq", "weekly", "--limit", "1"])
    assert result.exit_code == 0
    assert "employees" in result.output


def test_history_rejects_invalid_by():
    result = runner.invoke(app, ["history", "--by", "bogus"])
    assert result.exit_code != 0
    assert "bogus" in result.output


def test_history_rejects_invalid_freq():
    result = runner.invoke(app, ["history", "--freq", "daily"])
    assert result.exit_code != 0


def test_history_rejects_non_integer_limit():
    result = runner.invoke(app, ["history", "--limit", "abc"])
    assert result.exit_code != 0


def test_evaluate_default_shows_all_models():
    result = runner.invoke(app, ["evaluate"])
    assert result.exit_code == 0
    assert "ridge_lag" in result.output
    assert "seasonal_naive" in result.output


def test_evaluate_by_size_shows_segment_column():
    result = runner.invoke(app, ["evaluate", "--by", "size"])
    assert result.exit_code == 0
    assert "Segment" in result.output
    assert "National" in result.output


def test_evaluate_all_shows_group_column():
    result = runner.invoke(app, ["evaluate", "--all"])
    assert result.exit_code == 0
    assert "Group" in result.output
    assert "Census" in result.output


def test_evaluate_rejects_invalid_by():
    result = runner.invoke(app, ["evaluate", "--by", "bogus"])
    assert result.exit_code != 0


def test_evaluate_rejects_invalid_freq():
    result = runner.invoke(app, ["evaluate", "--freq", "daily"])
    assert result.exit_code != 0


def test_forecast_default_shows_detail_view():
    result = runner.invoke(app, ["forecast"])
    assert result.exit_code == 0
    assert "Next ADP Employment Report" in result.output
    assert "Forecast for" in result.output
    assert "Recent Accuracy (Backtested)" in result.output


def test_forecast_by_size_shows_compact_table():
    result = runner.invoke(app, ["forecast", "--by", "size"])
    assert result.exit_code == 0
    assert "Forecast Date" in result.output
    assert "National" in result.output


def test_forecast_all_weekly_runs():
    result = runner.invoke(app, ["forecast", "--all", "--freq", "weekly"])
    assert result.exit_code == 0
    assert "Group" in result.output


def test_forecast_rejects_invalid_by():
    result = runner.invoke(app, ["forecast", "--by", "bogus"])
    assert result.exit_code != 0


def test_forecast_rejects_invalid_freq():
    result = runner.invoke(app, ["forecast", "--freq", "daily"])
    assert result.exit_code != 0
