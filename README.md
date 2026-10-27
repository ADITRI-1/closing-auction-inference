# Is the closing-auction imbalance already in the price?

CSE558 Data Science, Monsoon 2026, midterm project. Group: Closing Cross.

Every Nasdaq closing auction publishes its order imbalance in the last ten minutes of trading. At 15:55 two rules
change at once: indicative clearing prices start being published, and new market-on-close orders stop being accepted.
We test whether the published imbalance predicts a stock's next-minute move against the index, whether that edge is
larger than the bid-ask spread, and whether the 15:55 rule change makes the move more predictable and less volatile.

The report is in `report/main.tex`; every result in it is read from `results/numbers.tex`, which the code generates.

## Data

Optiver, "Trading at the Close" (Kaggle, 2023): 5,237,980 snapshots of 200 Nasdaq stocks over 481 trading days.

1. Accept the competition rules at https://www.kaggle.com/competitions/optiver-trading-at-the-close
2. Download `train.csv` and place it at `data/raw/train.csv`.

The data is not in this repository (it is about 600 MB and covered by the competition rules).

## Setup

Use Python 3.11 or 3.12; the pinned versions in `requirements.txt` do not install on Python 3.13 or newer.

```bash
pip install -r requirements.txt
```

The results were also reproduced with Python 3.14 and newer libraries (numpy 2.5, pandas 3.1, scipy 1.18,
statsmodels 0.15). Every number in the report matched, except the raw file's in-memory size, because newer pandas
stores text columns more compactly.

## Run

```bash
python run_all.py
```

The full run takes about three minutes on a laptop and peaks at about 4 GB of memory. It stops with an error if a
stock-day is missing snapshots, since the 60-second returns depend on a complete panel. It writes the cleaned data to
`data/processed/clean.parquet`, the results to `results/` and the figures to `figures/`.

## Report

`report/main.tex` reads `results/numbers.tex` and `figures/`, so compile it inside this folder layout:

- locally: `cd report && pdflatex main.tex` (run it twice);
- on Overleaf: upload `report/main.tex`, `results/numbers.tex` and the `figures/` folder with the same folder names,
  and set `report/main.tex` as the main document.

## Outputs

| Path | Contents |
|---|---|
| `results/prep.json` | every preparation step with row counts, before and after summaries, recovered index weights |
| `results/eda.json` | descriptive statistics, autocorrelation, auction-time profile, decile response |
| `results/tests.json` | the five hypothesis tests, their alternatives, day-level checks, and Holm-adjusted p-values for both the main and the day-level p-values |
| `results/bounds.json` | t, Chebyshev and Hoeffding intervals for the Test 1 mean |
| `results/extensions.json` | placebo boundaries, subgroup robustness, power, edge against cost, permutation tests, distribution fits, Count-Min sketch |
| `results/numbers.tex` | every number quoted in the report, as LaTeX macros |
| `figures/*.pdf` | the report figures |

## Code

| File | Role |
|---|---|
| `run_all.py` | runs the whole pipeline from the raw file |
| `src/config.py` | paths and every threshold, fixed before testing |
| `src/load.py` | reads the raw file |
| `src/clean.py` | preparation steps, one-hot encoding, and before and after summaries |
| `src/features.py` | engineered features and recovery of the hidden index |
| `src/eda.py` | descriptive analysis |
| `src/inference.py` | the five hypothesis tests, their day-level checks and the Holm correction |
| `src/bounds.py` | concentration-bound intervals |
| `src/robustness.py` | subgroups, placebo boundaries, permutation tests |
| `src/power.py` | minimum detectable effects |
| `src/costs.py` | edge against trading cost |
| `src/fits.py` | maximum-likelihood fits and goodness of fit |
| `src/sketch.py` | Count-Min sketch over the row stream |
| `src/plotting.py` | figures |
| `src/numbers.py` | writes report numbers from the results |
| `report/main.tex` | the report |

## How the tests are set up

- Targets overlap for 50 of their 60 seconds, so tests use one snapshot per minute (952,320 rows).
- Stocks on the same day share market conditions, so the trading day is the unit: Tests 1 and 3 use one value per day
  (n = 481). Tests 2, 4 and 5 use the course's test for their question (χ², F, one-proportion z) on rows, and each is
  paired with a day-level check that gives the same decision.
- Every threshold that "matters" was fixed before testing. A result matters when its whole 95% interval lies beyond
  the threshold. Holm's correction is applied across the five tests, to both the main and the day-level p-values.
- Analyses found after looking at the results are labelled exploratory in the report.

## Results in one line each

1. The imbalance side predicts the next minute on 458 of 481 days (+0.47 bps per trade), but the edge is about a
   quarter of the median half-spread, so it does not matter for trading.
2. At 15:50, side and direction are dependent but weakly: the up-share gap between buy and sell imbalances is 1.8
   points (95% CI 0.8 to 2.8), far below the 10-point threshold.
3. After 15:55 the imbalance predicts the next minute about 2.5 times better (IC 0.032 to 0.080), in all 11
   subgroups tested.
4. After 15:55 the variance of the next minute is about 23% lower, or about 16% using only minutes without an
   announcement.
5. Trading the imbalance and paying the half-spread wins 39% of trades and loses 2.39 bps per trade on average,
   with a loss on every one of the 481 days.
6. Placebo boundaries show most of the 15:55 gain comes from the final minute, whose window contains the 16:00
   closing cross; without it the change is inconclusive. That minute is also the only place the edge clears the
   half-spread, an exploratory lead.
