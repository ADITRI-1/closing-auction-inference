# Is the closing-auction imbalance already in the price?

CSE558 Data Science, Monsoon 2026, midterm project.

Every Nasdaq closing auction publishes its order imbalance in the last ten minutes of trading. At 15:55 two rules
change at once: indicative clearing prices start being published, and new market-on-close orders stop being accepted.
We test whether the published imbalance predicts a stock's next-minute move against the index, whether that edge is
larger than the bid-ask spread, and whether the 15:55 rule change makes the move more predictable and less volatile.

## Data

Optiver, "Trading at the Close" (Kaggle, 2023): 5,237,980 snapshots of 200 Nasdaq stocks over 481 trading days.

1. Accept the competition rules at https://www.kaggle.com/competitions/optiver-trading-at-the-close
2. Download `train.csv` and place it at `data/raw/train.csv`.

## Run

```bash
pip install -r requirements.txt
python run_all.py
```

The full run takes about two minutes on a laptop and needs about 4 GB of memory.

## Outputs

| Path | Contents |
|---|---|
| `results/prep.json` | every preparation step with row counts, before and after summaries, recovered index weights |
| `results/eda.json` | descriptive statistics, autocorrelation, auction-time profile, decile response |
| `results/tests.json` | the five hypothesis tests, their alternative tests and Holm-adjusted p-values |
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
| `src/clean.py` | preparation steps and before and after summaries |
| `src/features.py` | engineered features and recovery of the hidden index |
| `src/eda.py` | descriptive analysis |
| `src/inference.py` | hypothesis tests |
| `src/bounds.py` | concentration-bound intervals |
| `src/robustness.py` | subgroups, placebo boundaries, permutation tests |
| `src/power.py` | minimum detectable effects |
| `src/costs.py` | edge against trading cost |
| `src/fits.py` | maximum-likelihood fits and goodness of fit |
| `src/sketch.py` | Count-Min sketch over the row stream |
| `src/plotting.py` | figures |
| `src/numbers.py` | writes report numbers from the results |

## Results in one line each

1. The imbalance side predicts the next minute on almost every day, but the edge is about a quarter of the half-spread.
2. Side and direction are dependent, with a negligible effect size.
3. After 15:55 the imbalance predicts the next minute about 2.5 times better, in every subgroup tested.
4. After 15:55 the next minute is about a fifth less variable.
5. A rule that trades the imbalance and pays the half-spread wins only 39% of the time.
6. Placebo boundaries show the gain is mostly the final minute, whose window contains the closing cross; the only
   place the edge clears the half-spread is that minute, an exploratory lead.
