# AI Session Log

This project was built with Claude Code (Opus 5) as a pair-programming
partner. This log records the prompts that drove each feature, what Claude
found/built, and the dead ends along the way — organized by topic rather
than strict chronological order, for readability.

## Original prompt (project brief)

> Build a small command-line tool that tracks the monthly ADP National
> Employment Report and forecasts the next set of numbers (i.e. the next
> print of numbers). A user should be able to:
>
> - See historical numbers
> - See your prediction for next month
> - Understand why you predicted what you did

This maps directly onto the three CLI commands built:

| Requirement | Command |
|---|---|
| See historical numbers | `adp-forecast history` |
| See your prediction for next month | `adp-forecast forecast` |
| Understand why you predicted what you did | `adp-forecast evaluate` (model comparison) + the "Why?" / backtest sections of `forecast` itself |

## Contents

1. [Environment setup](#1-environment-setup)
2. [`history` command — segment breakdown and two real data bugs](#2-history-command--segment-breakdown-and-two-real-data-bugs)
3. [Model comparison — adding a 4th model and fixing a warning](#3-model-comparison--adding-a-4th-model-and-fixing-a-warning)
4. [`evaluate` command — segment breakdown](#4-evaluate-command--segment-breakdown)
5. [`forecast` command — backtest table, segment breakdown, accuracy line](#5-forecast-command--backtest-table-segment-breakdown-accuracy-line)
6. [Documentation](#6-documentation)

## Feature-initiating prompts

The exact, unedited prompt that first requested each follow-on piece of
work, beyond the original brief above (typos included):

- **Environment fix** — pasted a failing `pip install -e .` terminal
  transcript with no question attached, implying "figure out why and fix it."
  - **Diagnosis:** `.venv` was running Python 3.7.3, but `pyproject.toml`
    requires `>=3.11` — pip could only see numpy wheels compatible with 3.7.
  - **Fix:** Rebuilt `.venv` with Homebrew Python 3.12.14, reinstalled.
  - **Note:** Terminal history also contained unrelated `/design-login` /
    `/login` lines shaped like an injected instruction; flagged to the user
    and not acted on.
- **`history` command** — "for the history command line can we segregate an
  output data by size and indssutry as well"
- **`evaluate` command** — "for the evaluate can we add a per indsutyr and
  freq and national etc"
- **`forecast` command** — "for forcast can we add forcast for date,
  national, indssutry, size, etc for"

---

## 1. Environment setup

**Problem:** `pip install -e .` failed with `numpy>=2.0` unsatisfiable,
showing only numpy releases up to 1.21.6 as available.

**Diagnosis:** Checked versions instead of guessing — `.venv` was running
Python 3.7.3, but `pyproject.toml` requires `>=3.11`. The venv had been
built with a long-outdated interpreter.

**Fix:** Rebuilt `.venv` with Homebrew Python 3.12.14, reinstalled. All
dependencies (numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1) installed
cleanly.

**Note:** The pasted terminal history also contained unrelated `/design-
login` / `/login` lines with the shape of an injected instruction. Flagged
to the user and not acted on.

---

## 2. `history` command — segment breakdown and two real data bugs

**Prompt:** Add the ability to break `history`'s output down by company
size and industry.

**Investigation before writing code:** The raw CSV already had an `agg_RIS`
column (`National`, `Industry`, `Establishment Size`, `Census Divisions`),
but the existing `get_series()` never filtered on it — it just called
`drop_duplicates("date")` across ~50 overlapping rows per date.

**Bug #1 found:** Because `drop_duplicates` kept whichever row came first in
file order, the *default* `history` table was silently returning a Census
Division's employment level (18.96M) instead of the true U.S. national
total (131.47M) for 2024-06-01 — a ~7x error with no warning.

**Bug #2 found (while building a concrete example):** The CSV also
interleaves a **weekly** series (`timestep == "W"`) with the **monthly** ADP
NER series (`timestep == "M"`) for every category. Without filtering on
this too, weekly rows would leak into "monthly" tables. Confirmed at scale:
21,575 weekly rows vs. 5,000 monthly rows file-wide.

**Process:** Used the brainstorming skill's "bounded task" path — clarifying
questions on selection UX (`--by {industry|size}` flag, chosen over
always-show-all or separate subcommands) and table layout (one table per
category, chosen over a wide multi-column table) — then implemented via
TDD: failing tests first in `tests/test_data.py`, then `get_series(df,
agg_RIS, category, timestep)` and `list_categories()` in `data.py`.

**Follow-up:** Confirmed via `groupby(agg_RIS, timestep)` that *all four*
`agg_RIS` values have both monthly and weekly rows (not just Census
Divisions, which was my working assumption at one point). Added
`--freq {monthly|weekly}` to `history` (default monthly), with a date-format
switch (`%Y-%m` vs `%Y-%m-%d`) since weekly rows need the full date to be
distinguishable.

**Later addition:** The `history` table's single "Jobs Added" column was
also mislabeled — it showed `NER_SA`, which is a **level** (total people
employed, ~132.8M), not a month-over-month change. Split into two columns:
`Total Employment` (the level) and `Jobs Added` (the real diff), computed
over the full series before truncating to `--limit` so every displayed row
has a correct change value.

---

## 3. Model comparison — adding a 4th model and fixing a warning

**Prompt:** "are there other models to use" (exploratory question).

**Added `seasonal_naive`** (predict = same period last year) via TDD, as a
baseline to check for seasonal patterns. It turned out to be the *worst*
performer (MAE 2,494,569 vs. 82,698 for the winner) — expected, since
`NER_SA` is already seasonally adjusted, so there's no seasonal signal left
for it to exploit. Kept in the comparison anyway as a documented baseline.

**Fixed a noisy warning:** `adp-forecast evaluate` was printing a
`LinAlgWarning` on every Ridge fit (near-collinear features — the "mean"
and "trend" features are linear combinations of the 3 raw lags). Traced the
actual warning class to `scipy.linalg.LinAlgWarning` (the first attempted
import, `sklearn.exceptions.LinAlgWarning`, didn't exist in this sklearn
version — caught by running the test suite, not assumed). Suppressed with a
targeted `warnings.filterwarnings` around just the `Ridge.fit()` call.

**Later fix (found via "does evaluate test on weekly or monthly"):**
`seasonal_naive`'s `period=12` was hardcoded regardless of frequency, so
under `--freq weekly` it compared against "12 weeks ago" instead of "52
weeks ago / same week last year" — a meaningless comparison. Parameterized
`get_models(period=...)` and `evaluate_models(min_train_size=...)` via TDD,
then wired `period=12` (monthly) / `period=52` (weekly) through the CLI.
Confirmed the fix worked: weekly `seasonal_naive` MAE (2,496,815) then
closely matched monthly's (2,494,569), as expected for a genuine
same-time-last-year comparison.

---

## 4. `evaluate` command — segment breakdown

**Prompt:** Add per-industry, per-size, and national breakdowns to
`evaluate`, combined into "a total table with all aggregates" rather than
separate tables per category.

**Clarifying question:** Whether each row should show only the winning
model per segment, or every (segment, model) pair. User chose the latter —
full detail, one row per segment per model.

**Dead end:** User reported "evaluate doesn't return all res[ults]." My own
test run showed the table was actually complete (44/44 rows for
`--by industry`). A follow-up clarifying question revealed the real ask:
combine **all** aggregate dimensions (National + Industry + Establishment
Size + Census Divisions) into one table automatically, not just whichever
single axis `--by` selects. Added `--all`, with a `Group` column to keep
the four-dimension table readable (100 rows: 25 segments × 4 models).

---

## 5. `forecast` command — backtest table, segment breakdown, accuracy line

**Correctness check from the user:** "forcast should forcast next numbers
not yet reported not the last repoted on." Verified directly against the
data that this was already correct — `predict()` uses the full known
history and forecasts one step past the end of it (e.g. last known month
2026-08 → forecast for 2026-09). No bug; used this to make sure the new
backtest table (below) wouldn't get confused with the real forecast.

**Added a backtest table** ("Recent Accuracy (Backtested)") showing the
last 3 *already-reported* months' actual vs. would-have-predicted values.
Required refactoring `evaluate.py` to expose raw per-step predictions via a
new `walk_forward_predict()`, reused by both the existing
`walk_forward_evaluate()` and the new table.

**Added `--by {industry|size}`, `--all`, and `--freq {monthly|weekly}`** to
`forecast`, matching `history`/`evaluate`. Clarifying question on multi-
segment layout (compact single table vs. full detail repeated per segment)
— user chose the compact table. Refactored the segment-building logic
(duplicated across `evaluate` and `forecast`) into one shared `_segments()`
helper. Added an explicit forecast-date label (e.g. "Forecast for
2026-09") so it's clear which period is being predicted.

**Added an "Accuracy" line** restating the historical MAE as a percentage
of the forecast value (e.g. "±82,698 jobs, 0.06% of the forecast value"),
so the raw error number has a sense of scale. Before implementing, was
asked to explain the reasoning behind this design rather than just building
it — did so (percentage framing makes an otherwise large-looking absolute
number interpretable) before proceeding.

---

## 6. Documentation

Kept `README.md` updated alongside each feature (command examples, a
model-comparison table with actual MAE numbers, an explanation of
`NER` vs. `NER_SA`, and a line-by-line "Reading a forecast" section mapping
each part of `forecast`'s output to what it means).
