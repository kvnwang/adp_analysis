import enum
from pathlib import Path
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from .data import load_data, get_series, list_categories, MONTHLY_TIMESTEP, WEEKLY_TIMESTEP
from .evaluate import evaluate_models, walk_forward_predict
from .forecast import get_models

app = typer.Typer(help="Forecast the next ADP National Employment Report.")
console = Console()
DEFAULT_DATA = Path("data/adp_history.csv")

class Segment(str, enum.Enum):
    industry = "industry"
    size = "size"

SEGMENT_AGG_RIS = {
    Segment.industry: "Industry",
    Segment.size: "Establishment Size",
}

ALL_AGG_RIS = ["Industry", "Establishment Size", "Census Divisions"]

class Frequency(str, enum.Enum):
    monthly = "monthly"
    weekly = "weekly"

FREQUENCY_TIMESTEP = {
    Frequency.monthly: MONTHLY_TIMESTEP,
    Frequency.weekly: WEEKLY_TIMESTEP,
}

FREQUENCY_DATE_FORMAT = {
    Frequency.monthly: "%Y-%m",
    Frequency.weekly: "%Y-%m-%d",
}

FREQUENCY_SEASONAL_PERIOD = {
    Frequency.monthly: 12,
    Frequency.weekly: 52,
}

FREQUENCY_UNIT = {
    Frequency.monthly: "month",
    Frequency.weekly: "week",
}

FREQUENCY_STEP = {
    Frequency.monthly: pd.DateOffset(months=1),
    Frequency.weekly: pd.Timedelta(weeks=1),
}

def load_series(agg_RIS: str = "National", category: str = "U.S.", timestep: str = MONTHLY_TIMESTEP):
    return get_series(load_data(DEFAULT_DATA), agg_RIS, category, timestep)

def _segments(by: Segment, all_segments: bool, timestep: str):
    segments = [("National", "National", load_series(timestep=timestep))]
    if all_segments:
        df = load_data(DEFAULT_DATA)
        for agg_RIS in ALL_AGG_RIS:
            segments += [
                (agg_RIS, category, get_series(df, agg_RIS, category, timestep))
                for category in list_categories(df, agg_RIS)
            ]
    elif by is not None:
        df = load_data(DEFAULT_DATA)
        agg_RIS = SEGMENT_AGG_RIS[by]
        segments += [
            (agg_RIS, category, get_series(df, agg_RIS, category, timestep))
            for category in list_categories(df, agg_RIS)
        ]
    return segments

def _next_date(last_date, freq: Frequency):
    return last_date + FREQUENCY_STEP[freq]

def _print_history_table(series, title: str, limit: int, date_format: str):
    table = Table(title=title)
    table.add_column("Date")
    table.add_column("Total Employment", justify="right")
    table.add_column("Jobs Added", justify="right")
    changes = series.diff()
    for date, value in series.tail(limit).items():
        change = changes.loc[date]
        change_str = f"{change:+,.0f}" if pd.notna(change) else "—"
        table.add_row(date.strftime(date_format), f"{value:,.0f}", change_str)
    console.print(table)

@app.command()
def history(
    limit: int = typer.Option(36, "--limit", "-n"),
    by: Segment = typer.Option(None, "--by", help="Break down by industry or size instead of the national total."),
    freq: Frequency = typer.Option(Frequency.monthly, "--freq", help="Monthly or weekly cadence."),
):
    """Show ADP employment history, optionally by industry/size and monthly/weekly."""
    timestep = FREQUENCY_TIMESTEP[freq]
    date_format = FREQUENCY_DATE_FORMAT[freq]

    if by is None:
        _print_history_table(load_series(timestep=timestep), "ADP Employment History", limit, date_format)
        return

    df = load_data(DEFAULT_DATA)
    agg_RIS = SEGMENT_AGG_RIS[by]
    for category in list_categories(df, agg_RIS):
        series = get_series(df, agg_RIS, category, timestep)
        _print_history_table(series, f"ADP Employment History — {category}", limit, date_format)

@app.command()
def evaluate(
    by: Segment = typer.Option(None, "--by", help="Also evaluate industry or size segments alongside National."),
    all_segments: bool = typer.Option(
        False, "--all", help="Evaluate National plus every industry, size, and census division segment, combined."
    ),
    freq: Frequency = typer.Option(Frequency.monthly, "--freq", help="Monthly or weekly cadence."),
):
    """Walk-forward validate each forecasting model, nationally or across all segments."""
    timestep = FREQUENCY_TIMESTEP[freq]
    period = FREQUENCY_SEASONAL_PERIOD[freq]
    segments = _segments(by, all_segments, timestep)

    show_segment = by is not None or all_segments
    show_group = all_segments
    title = "Walk-Forward Evaluation" if not show_segment else "Walk-Forward Evaluation — All Segments"

    table = Table(title=title)
    columns = [*(["Group"] if show_group else []), *(["Segment"] if show_segment else []), "Model", "MAE", "RMSE", "Forecasts"]
    for col in columns:
        table.add_column(col, justify="right" if col not in ("Model", "Segment", "Group") else "left")

    for group_label, segment_label, series in segments:
        results = evaluate_models(series.tolist(), get_models(period=period), min_train_size=period)
        for r in results:
            row = [
                *([group_label] if show_group else []),
                *([segment_label] if show_segment else []),
                r.model, f"{r.mae:,.0f}", f"{r.rmse:,.0f}", str(r.predictions),
            ]
            table.add_row(*row)
    console.print(table)

def _print_forecast_detail(series, freq: Frequency, period: int):
    values = series.tolist()
    dates = series.index.tolist()
    date_format = FREQUENCY_DATE_FORMAT[freq]
    models = get_models(period=period)
    results = evaluate_models(values, models, min_train_size=period)
    best = next(m for m in models if m.name == results[0].model)
    prediction = best.predict(values)
    next_date = _next_date(dates[-1], freq)

    console.print("\n[bold]Next ADP Employment Report[/bold]\n")
    console.print(f"Forecast for {next_date.strftime(date_format)}: [bold]{prediction:,.0f} jobs[/bold] (not yet reported)")
    console.print(f"Model: {best.name}")
    console.print(f"Historical MAE: {results[0].mae:,.0f} jobs")
    accuracy_pct = results[0].mae / abs(prediction) * 100
    console.print(
        f"Accuracy: predictions have typically been within ±{results[0].mae:,.0f} jobs "
        f"({accuracy_pct:.2f}% of the forecast value), based on {results[0].predictions} "
        f"historical {freq.value} tests."
    )
    recent = values[-3:]
    unit = FREQUENCY_UNIT[freq]
    console.print("\n[bold]Why?[/bold]")
    console.print(f"- Last 3 prints: {', '.join(f'{x:,.0f}' for x in recent)}")
    console.print(f"- 3-{unit} average: {sum(recent)/len(recent):,.0f}")
    console.print(f"- Selected {best.name} using lowest walk-forward MAE")

    backtest_predictions, backtest_actuals = walk_forward_predict(values, best, min_train_size=period)
    backtest_dates = dates[period:]
    table = Table(title="\nRecent Accuracy (Backtested)")
    table.add_column("Date")
    table.add_column("Actual", justify="right")
    table.add_column("Would-Have-Predicted", justify="right")
    table.add_column("Difference", justify="right")
    for date, actual, predicted in zip(backtest_dates[-3:], backtest_actuals[-3:], backtest_predictions[-3:]):
        table.add_row(
            date.strftime(date_format),
            f"{actual:,.0f}",
            f"{predicted:,.0f}",
            f"{predicted - actual:+,.0f}",
        )
    console.print(table)

@app.command()
def forecast(
    by: Segment = typer.Option(None, "--by", help="Forecast industry or size segments instead of just National."),
    all_segments: bool = typer.Option(
        False, "--all", help="Forecast National plus every industry, size, and census division segment, combined."
    ),
    freq: Frequency = typer.Option(Frequency.monthly, "--freq", help="Monthly or weekly cadence."),
):
    """Predict the next ADP National Employment Report print, nationally or across segments."""
    timestep = FREQUENCY_TIMESTEP[freq]
    period = FREQUENCY_SEASONAL_PERIOD[freq]
    date_format = FREQUENCY_DATE_FORMAT[freq]
    segments = _segments(by, all_segments, timestep)

    if by is None and not all_segments:
        _print_forecast_detail(segments[0][2], freq, period)
        return

    show_group = all_segments
    title = "Next ADP Employment Report — All Segments" if all_segments else "Next ADP Employment Report — By Segment"
    table = Table(title=title)
    columns = [*(["Group"] if show_group else []), "Segment", "Forecast Date", "Forecast", "Model", "MAE"]
    for col in columns:
        table.add_column(col, justify="right" if col in ("Forecast", "MAE") else "left")

    for group_label, segment_label, series in segments:
        values = series.tolist()
        models = get_models(period=period)
        results = evaluate_models(values, models, min_train_size=period)
        best = next(m for m in models if m.name == results[0].model)
        prediction = best.predict(values)
        next_date = _next_date(series.index[-1], freq)
        row = [
            *([group_label] if show_group else []),
            segment_label,
            next_date.strftime(date_format),
            f"{prediction:,.0f}",
            best.name,
            f"{results[0].mae:,.0f}",
        ]
        table.add_row(*row)
    console.print(table)

if __name__ == "__main__":
    app()
