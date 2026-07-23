# Football Goals Timing Dataset

Scrapes goal-by-minute data for the top 5 European leagues (Premier League,
La Liga, Bundesliga, Serie A, Ligue 1) from Understat, going back to the
2014/15 season, and keeps itself updated automatically once a month.

## ⚠️ Important — test locally first

This was written using the well-established, widely-used pattern for
reading Understat's embedded page data, but **it has not been run against
a live network connection** in the environment it was built in (that
sandbox can't reach understat.com). Please run the steps below locally
once to confirm it actually pulls data before you push it live and rely
on the scheduled automation.

## What you get

`data/goals_dataset.csv` with one row per goal (plus one row per 0-0 draw
so those matches aren't silently missing), containing:

| column | meaning |
|---|---|
| match_id | Understat's internal match ID |
| league | Premier League / La Liga / Bundesliga / Serie A / Ligue 1 |
| season | season start year, e.g. 2025 = the 2025/26 season |
| match_date | kickoff date/time |
| home_team / away_team | team names |
| final_home_goals / final_away_goals | final score |
| scoring_team | 'home' or 'away' (blank for the 0-0 filler row) |
| goal_minute | minute the goal was scored |
| scorer | player name |
| shot_xg | xG of the shot that led to the goal |

## One-time local setup

```bash
git clone <your-repo-url>
cd football-goals-dataset
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Initial historical backfill (run this first, locally)

```bash
cd scripts
python build_dataset.py --full
```

This will take a while — roughly 12 seasons × 5 leagues × ~380 matches
each, at a polite ~1 request/second, so budget a few hours. It's fully
resumable: if you stop it (Ctrl+C) or it crashes partway through, just
run `python build_dataset.py` again (without `--full`) and it'll pick up
exactly where it left off, using `data/checkpoint.json`.

Once this completes, commit and push the resulting `data/goals_dataset.csv`
and `data/checkpoint.json` to your GitHub repo.

## Setting up the monthly automation (GitHub Actions)

1. If you don't already have one: create a free account at github.com
2. Create a new repository and push this whole folder to it
3. Push the already-backfilled `data/goals_dataset.csv` and
   `data/checkpoint.json` (see above) — don't let the very first Actions
   run attempt the full historical backfill, it'll likely time out
4. That's it — the workflow in `.github/workflows/monthly_update.yml` will
   run automatically on the 1st of every month, fetch any newly completed
   matches, and commit the updated CSV back to the repo
5. You can also trigger it manually any time from the repo's "Actions" tab
   → "Monthly goals dataset update" → "Run workflow"

## Reading the dataset in Python

```python
import pandas as pd
df = pd.read_csv("data/goals_dataset.csv")
```

## Notes on modeling goal timing

This is naturally a hazard-rate / point-process problem rather than plain
regression — consider bucketing matches into time intervals (e.g. every 5
minutes) and predicting goal probability per bucket, conditioned on
current score difference and time elapsed, using a gradient-boosted model
(XGBoost/LightGBM) as a strong practical starting point.
