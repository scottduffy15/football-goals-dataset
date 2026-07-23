"""
Turns data/goals_dataset.csv (one row per goal, plus 0-0 filler rows) into
a modeling-ready panel: one row per (match, time-bucket, side), with the
label being whether that side scored a goal in that bucket.

Why this shape: predicting *when* goals happen is a hazard/point-process
problem, not a plain regression. Bucketing the match into intervals (e.g.
every 5 minutes) and asking "did team X score in this window, given the
match situation so far" turns it into an ordinary binary classification
problem, which any standard ML library handles well -- while still
capturing the real pattern that goal-scoring *rate* isn't flat across 90
minutes (it typically rises late in each half).

Usage:
    python3 build_panel.py --input ../data/goals_dataset.csv --output ../data/panel.csv
"""

import argparse

import numpy as np
import pandas as pd

BUCKET_SIZE = 5          # minutes per bucket
MATCH_LENGTH = 90         # goals beyond 90 (stoppage time) get folded into the final bucket
ROLLING_WINDOW = 5        # number of prior matches used for team strength features


def load_goals(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["match_date"])
    return df


def build_match_table(goals_df):
    """
    Collapse the long goal-events table into one row per match, with the
    list of minutes each side scored in.
    """
    match_cols = [
        "match_id", "league", "season", "match_date",
        "home_team", "away_team", "final_home_goals", "final_away_goals",
    ]
    matches = goals_df[match_cols].drop_duplicates(subset="match_id").reset_index(drop=True)

    def minutes_for(match_id, side):
        rows = goals_df[
            (goals_df["match_id"] == match_id) & (goals_df["scoring_team"] == side)
        ]
        return sorted(int(m) for m in rows["goal_minute"].dropna().tolist())

    matches["home_goal_minutes"] = matches["match_id"].apply(lambda mid: minutes_for(mid, "home"))
    matches["away_goal_minutes"] = matches["match_id"].apply(lambda mid: minutes_for(mid, "away"))
    matches = matches.sort_values("match_date").reset_index(drop=True)
    return matches


def compute_rolling_team_strength(matches, window=ROLLING_WINDOW):
    """
    Adds trailing rolling-average attack/defense strength for both the
    home and away team in each match, computed strictly from matches
    *before* the current one (no leakage).
    """
    # Build a "team perspective" long table: one row per team per match
    home_view = matches[["match_id", "match_date", "home_team", "final_home_goals", "final_away_goals"]].rename(
        columns={"home_team": "team", "final_home_goals": "goals_for", "final_away_goals": "goals_against"}
    )
    away_view = matches[["match_id", "match_date", "away_team", "final_away_goals", "final_home_goals"]].rename(
        columns={"away_team": "team", "final_away_goals": "goals_for", "final_home_goals": "goals_against"}
    )
    team_matches = pd.concat([home_view, away_view], ignore_index=True)
    team_matches = team_matches.sort_values(["team", "match_date"]).reset_index(drop=True)

    # Rolling mean over the PREVIOUS `window` matches (shift(1) excludes the current match itself)
    team_matches["rolling_attack"] = (
        team_matches.groupby("team")["goals_for"]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )
    team_matches["rolling_defense"] = (
        team_matches.groupby("team")["goals_against"]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )

    # League-wide average goals per match, used as a fallback for teams with no prior history yet
    league_avg_goals = matches["final_home_goals"].mean() + matches["final_away_goals"].mean()
    league_avg_goals = league_avg_goals / 2 if not np.isnan(league_avg_goals) else 1.3
    team_matches["rolling_attack"] = team_matches["rolling_attack"].fillna(league_avg_goals)
    team_matches["rolling_defense"] = team_matches["rolling_defense"].fillna(league_avg_goals)

    strength_lookup = team_matches.set_index(["match_id", "team"])[["rolling_attack", "rolling_defense"]]

    def get_strength(row, side):
        team = row["home_team"] if side == "home" else row["away_team"]
        try:
            vals = strength_lookup.loc[(row["match_id"], team)]
            return vals["rolling_attack"], vals["rolling_defense"]
        except KeyError:
            return league_avg_goals, league_avg_goals

    strengths = matches.apply(lambda r: get_strength(r, "home"), axis=1, result_type="expand")
    matches["home_attack_strength"], matches["home_defense_strength"] = strengths[0], strengths[1]

    strengths = matches.apply(lambda r: get_strength(r, "away"), axis=1, result_type="expand")
    matches["away_attack_strength"], matches["away_defense_strength"] = strengths[0], strengths[1]

    return matches


def build_time_bucket_panel(matches, bucket_size=BUCKET_SIZE):
    """
    Explodes each match into (bucket, side) rows with features and a
    'scored' label -- this is the final modeling-ready panel.
    """
    bucket_starts = list(range(0, MATCH_LENGTH, bucket_size))
    rows = []

    for _, m in matches.iterrows():
        home_minutes = m["home_goal_minutes"]
        away_minutes = m["away_goal_minutes"]

        home_score_so_far = 0
        away_score_so_far = 0

        for bucket_start in bucket_starts:
            bucket_end = bucket_start + bucket_size
            is_last_bucket = bucket_start == bucket_starts[-1]

            # goals in this bucket (last bucket also absorbs stoppage-time goals, i.e. minute >= 90)
            if is_last_bucket:
                home_goals_in_bucket = sum(1 for x in home_minutes if x >= bucket_start)
                away_goals_in_bucket = sum(1 for x in away_minutes if x >= bucket_start)
            else:
                home_goals_in_bucket = sum(1 for x in home_minutes if bucket_start <= x < bucket_end)
                away_goals_in_bucket = sum(1 for x in away_minutes if bucket_start <= x < bucket_end)

            base_row = {
                "match_id": m["match_id"],
                "league": m["league"],
                "season": m["season"],
                "bucket_start": bucket_start,
            }

            # home team's perspective
            rows.append({
                **base_row,
                "side": "home",
                "team_attack_strength": m["home_attack_strength"],
                "opponent_defense_strength": m["away_defense_strength"],
                "score_diff_entering_bucket": home_score_so_far - away_score_so_far,
                "scored": int(home_goals_in_bucket > 0),
            })
            # away team's perspective
            rows.append({
                **base_row,
                "side": "away",
                "team_attack_strength": m["away_attack_strength"],
                "opponent_defense_strength": m["home_defense_strength"],
                "score_diff_entering_bucket": away_score_so_far - home_score_so_far,
                "scored": int(away_goals_in_bucket > 0),
            })

            home_score_so_far += home_goals_in_bucket
            away_score_so_far += away_goals_in_bucket

    return pd.DataFrame(rows)


def run(input_path, output_path):
    print(f"Loading goal events from {input_path} ...")
    goals_df = load_goals(input_path)

    print("Building match-level table ...")
    matches = build_match_table(goals_df)
    print(f"  {len(matches)} matches found")

    print("Computing rolling team strength features (no leakage) ...")
    matches = compute_rolling_team_strength(matches)

    print("Building time-bucket panel ...")
    panel = build_time_bucket_panel(matches)
    print(f"  {len(panel)} panel rows ({len(panel) // (len(matches) * 2) if matches.shape[0] else 0} buckets/side x 2 sides x {len(matches)} matches)")

    panel.to_csv(output_path, index=False)
    print(f"Saved panel dataset to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../data/goals_dataset.csv")
    parser.add_argument("--output", default="../data/panel.csv")
    args = parser.parse_args()
    run(args.input, args.output)
