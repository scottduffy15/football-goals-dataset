# Modeling: goal-timing prediction

This builds on `data/goals_dataset.csv` (produced by the scraper) to train
a baseline model predicting, for each 5-minute window of a match, the
probability that a given team scores in that window.

## Why this framing

Predicting *when* goals happen is naturally a hazard-rate / point-process
problem, not plain regression -- goal-scoring rate isn't flat across 90
minutes (it typically rises late in each half). Bucketing the match into
intervals and predicting "does team X score in this window, given the
match situation so far" turns it into ordinary binary classification,
which standard ML libraries handle well.

## Step 1: Build the modeling panel

```bash
cd scripts/modeling
python3 build_panel.py --input ../../data/goals_dataset.csv --output ../../data/panel.csv
```

This turns the long goal-events CSV into a wide panel: one row per
(match, 5-minute bucket, side), with:

| column | meaning |
|---|---|
| bucket_start | minute the bucket begins (0, 5, 10, ... 85; last bucket also absorbs stoppage time) |
| side | 'home' or 'away' -- whose scoring chance this row represents |
| team_attack_strength | that team's rolling average goals scored over their last 5 matches (computed with no leakage -- only matches strictly before this one) |
| opponent_defense_strength | opponent's rolling average goals conceded over their last 5 matches |
| score_diff_entering_bucket | that team's goal difference at the START of this bucket (captures game-state effects, e.g. teams chasing a goal push harder) |
| scored | label: did this side score in this bucket? (1/0) |

## Step 2: Train and evaluate the baseline model

```bash
python3 train_model.py --panel ../../data/panel.csv
```

This:
- Splits by **season** (trains on all-but-the-most-recent season, tests on
  the most recent) rather than a random row split, since rows from the
  same match are correlated -- a random split would leak information
  between train and test
- Trains a `HistGradientBoostingClassifier` (from scikit-learn, no extra
  install needed) on the panel
- Reports Brier score and log loss
- Saves `model_calibration_by_bucket.png` -- a plot comparing the
  model's average predicted scoring rate per bucket against the actual
  rate in the test season. This is your sanity check: if the model's
  learned anything real, its curve should roughly track the actual one,
  including the late-game rise.

## Known limitations / natural next steps

- **Red cards, injuries, and other in-match events aren't included yet**
  (the scraped dataset doesn't currently capture these) -- these
  meaningfully affect scoring rate and would be a strong addition later.
- **Rolling window of 5 matches** for team strength is a reasonable
  starting point but somewhat arbitrary -- worth experimenting with
  different window sizes.
- **HistGradientBoostingClassifier** doesn't expose feature importances
  directly; if you want to inspect which features matter most, look into
  `permutation_importance` from scikit-learn, or swap in LightGBM/XGBoost
  which report importances natively.
- Currently treats every league the same structurally (only a categorical
  feature distinguishes them) -- could also add league-average goals per
  game as an explicit feature if leagues differ a lot in scoring rate.
