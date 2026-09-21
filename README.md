# ADP Employment Forecast

A small CLI that tracks the ADP National Employment Report and forecasts the next monthly private-sector employment print.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Place the ADP historical CSV at `data/adp_history.csv`.

## Commands

```bash
adp-forecast history                             # national total, monthly (default)
adp-forecast history --by industry               # one table per industry, monthly
adp-forecast history --by size                   # one table per establishment-size bucket, monthly
adp-forecast history --freq weekly               # national total, weekly
adp-forecast history --by industry --freq weekly # one table per industry, weekly
adp-forecast evaluate                            # national only, monthly
adp-forecast evaluate --by size                  # national + each size bucket, one combined table
adp-forecast evaluate --by industry --freq weekly
adp-forecast evaluate --all                      # national + every industry, size, and census division, one combined table
adp-forecast forecast                            # national only, monthly, full detail
adp-forecast forecast --by size                  # national + each size bucket, one compact table
adp-forecast forecast --all --freq weekly        # every segment, weekly, one compact table
```

## Data

The source CSV includes both `NER` (raw) and `NER_SA` (seasonally adjusted)
columns. This project uses `NER_SA` throughout, since raw counts swing with
predictable seasonal hiring/layoff patterns (e.g. `NER` can drop
month-over-month even when `NER_SA` is still rising), and seasonally
adjusted figures are the standard convention for reporting monthly
employment change.

The CSV also stacks multiple dimensions in one file — `agg_RIS`/`category`
(National, Industry, Establishment Size, Census Divisions) and `timestep`
(`M` monthly vs. `W` weekly) — so every read filters on all three to avoid
mixing incompatible rows for the same date.

`NER_SA` itself is a **level** (total people employed, e.g. ~132.8 million),
not a month-over-month change. `adp-forecast history` shows both: `Total
Employment` is the raw level, and `Jobs Added` is that level's period-over-
period change (what ADP's headline "private employment rose by X" figure
refers to) — computed as a diff over the full series before truncating to
`--limit`, so every displayed row has a real change value.

## Approach

Models are compared with walk-forward validation (`adp-forecast evaluate`):
starting after the first 12 months, each model predicts one month ahead
using only the data available up to that point, then the window slides
forward and the actual value is revealed — repeated across the whole
series. MAE (mean absolute error, in jobs) is the primary metric because
it's directly interpretable; RMSE is also reported since it penalizes large
misses more heavily.

### Models tried

| Model | Idea | Why we tried it | Walk-forward MAE |
|---|---|---|---|
| `naive` | Predict next month = last month's value | Simplest possible baseline — anything more complex should beat this | 222,468 |
| `moving_average` | Predict next month = average of last 3 months | Smooths out single-month noise vs. naive | 434,250 |
| `seasonal_naive` | Predict next period = same period last year (period = 12 for monthly, 52 for weekly) | Employment has real seasonal hiring/layoff patterns, worth checking for | 2,494,569 |
| `ridge_lag` | Ridge regression on the last 3 months, their mean, and their trend, refit fresh each prediction | Lets a model weigh recent lags/trend rather than a fixed rule | **82,698** |

### Why `ridge_lag` is used

`adp-forecast forecast` always picks whichever model has the lowest
walk-forward MAE, and on this series that's consistently `ridge_lag` — by a
wide margin (2.7x better than `naive`, 5.3x better than `moving_average`,
30x better than `seasonal_naive`). It wins because the national monthly
series is trending and autocorrelated (each month is close to recent
months, not just noise around a flat average), and a regression can learn
*how much* weight to put on each recent lag instead of assuming a fixed
rule like "just average the last 3" or "just repeat last month."

`seasonal_naive` performs worst specifically because `NER_SA` is already
seasonally adjusted — the seasonal pattern it's designed to exploit has
already been removed from the data, so it just discards a year of
trend/growth instead of adding information. It's kept in the comparison as
a baseline, not because it was expected to win.

Because model selection is automatic (lowest MAE wins) and re-evaluated
every run, if a future model is added that outperforms `ridge_lag`,
`forecast` will switch to it without any code change.

### Reading a forecast

`adp-forecast forecast`'s output, line by line:

```
Forecast for 2026-09: 132,841,601 jobs (not yet reported)
```
The one real prediction. It's for the *next* month, which hasn't been
reported yet — produced by fitting the winning model on all known history
and predicting one step past the end of it. This is the only line in the
output describing something that hasn't happened; everything below it
describes past performance, used to judge whether to trust this number.

```
Model: ridge_lag
```
Whichever model had the lowest walk-forward MAE this run (see "Models
tried" above) — selection is automatic and re-checked every time you run
the command, not hardcoded.

```
Historical MAE: 82,698 jobs
Accuracy: predictions have typically been within ±82,698 jobs (0.06% of the
forecast value), based on 188 historical monthly tests.
```
How far off that model's one-step-ahead predictions have been, on average,
across every walk-forward test run so far (188 monthly tests here). The
Accuracy line is the same MAE restated as a percentage of the forecast
value, since "82,698" alone doesn't say whether that's a good or bad error
until you know the number is ~132 million.

```
Why?
- Last 3 prints: 132,719,000, 132,765,000, 132,803,000
- 3-month average: 132,762,333
- Selected ridge_lag using lowest walk-forward MAE
```
Context for sanity-checking the forecast by eye: the actual recent values
feeding the model, a naive average for comparison, and an explicit
statement of the selection rule.

```
Recent Accuracy (Backtested)
Date      Actual        Would-Have-Predicted   Difference
2026-06   132,719,000   132,739,677            +20,677
2026-07   132,765,000   132,796,943            +31,943
2026-08   132,803,000   132,795,329            -7,671
```
The last 3 *already-reported* months: what the model would have predicted
for each, using only data available before that month, next to what
actually happened. Both columns here are known values — this is a track
record, not a forecast. It exists so you can see recent accuracy directly
rather than only trusting a single aggregated MAE number.

## Further analysis: things to build upon

Concrete gaps and opportunities noticed while building this, not just a
generic wishlist:

### Modeling

- **Confidence intervals, not just a point estimate.** `forecast` currently
  prints a single number and a historical MAE, but no interval (e.g. "132.8M
  ± 165K at 95% confidence," using RMSE or backtest residual spread). A
  point forecast without an interval understates how much walk-forward error
  actually varies month to month.
- **Multi-step forecasts.** Everything here is one-step-ahead (predict next
  month using known history). Extending `ridge_lag` to forecast 2-3 months
  out would need either a recursive approach (feed predictions back in as
  lags) or direct multi-horizon models — worth testing since error likely
  compounds fast with the current lag-3 feature set.
- **Ensembling.** In the `--all` breakdown, `ridge_lag` wins on every single
  segment (25/25), which is a strong signal but was only checked
  informally by eye. An ensemble (e.g. average of `ridge_lag` and `naive`)
  is cheap to add and worth walk-forward-testing against `ridge_lag` alone
  — it may or may not help, but the comparison hasn't been run.
- **Hyperparameter tuning.** `RidgeLagModel`'s `alpha=10.0` and `lags=3` are
  fixed constants, chosen once and never tuned against the walk-forward
  loop itself. A small grid search over `alpha`/`lags`, scored by the same
  MAE the CLI already computes, could plausibly beat 82,698 without adding
  a new model class.

### Data and segments

- **Census Divisions has no `--by` option.** `history`/`evaluate`/`forecast`
  support `--by industry` and `--by size`, but the 4th `agg_RIS` dimension
  (9 Census Divisions) is only reachable via `--all` — there's no
  `--by region` for a focused view the way there is for industry/size.
- **Segment forecasts aren't reconciled with the national one.** The 10
  industry forecasts and the national forecast are produced completely
  independently — they could disagree even though industries should
  roughly sum to the national total. A reconciliation step (e.g. scale
  segment forecasts to match the national total, or the reverse) would
  make the `--all` output internally consistent.
- **The weekly series is underused.** It exists and is queryable
  (`--freq weekly`), but `forecast`'s monthly prediction doesn't use recent
  weekly data as a leading indicator ("nowcasting") — e.g. using partial-month
  weekly prints to adjust the in-progress month's forecast before it's
  officially reported.

### Operational

- **Automate monthly ingestion.** `data/adp_history.csv` is manually placed;
  there's no fetch/refresh step, so the tool is only as current as whoever
  last updated the file.
- **Store forecast vintages and actual outcomes.** The backtest table
  simulates history from the current CSV, but doesn't persist *actual past
  forecasts this tool made* to compare against outcomes as they arrive —
  without that, there's no way to verify the walk-forward MAE numbers match
  real-world performance going forward, only historical simulation.
- **Add monitoring/alerting.** No mechanism today flags a forecast that
  turns out to be way off, or notices when a newly-added model starts
  consistently beating `ridge_lag` (which `evaluate` could detect, but
  nothing currently surfaces that as an alert rather than requiring someone
  to run the command).

### Testing

`tests/test_cli.py` covers `history`/`evaluate`/`forecast` via typer's
`CliRunner` — happy paths for default/`--by`/`--all`/`--freq`, plus invalid
input (`--by bogus`, `--freq daily`, non-integer `--limit`). Error paths on
the model classes and `load_data`'s missing-column check are covered too
(34 tests total). One tradeoff: the `--all` CLI tests genuinely run all 4
models across all 25 real segments against the real CSV, which makes the
full suite take ~110s instead of ~3s — worth trimming to a lighter smoke
test if that becomes annoying, or worth injecting a fixture CSV so CLI
tests don't depend on `data/adp_history.csv` at all.
