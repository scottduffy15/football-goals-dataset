"""
Builds and incrementally updates goals_dataset.csv

Usage:
    python build_dataset.py            # incremental update (default)
    python build_dataset.py --full     # force a full historical rebuild

The dataset is resumable: if a run gets interrupted (e.g. GitHub Actions
job timeout during the initial big backfill), just run it again and it
will pick up where it left off, using data/checkpoint.json to track
which (league, season) combinations are already fully scraped.

Recommended: run the FIRST full backfill locally rather than via GitHub
Actions, since scraping ~12 seasons x 5 leagues x ~380 matches each with
a polite 1-second delay per request will take several hours. After that,
the scheduled monthly job only needs to fetch newly completed matches.
"""

import argparse
import csv
import json
import os

from understat_scraper import (
    LEAGUES,
    all_seasons,
    get_matches_for_season,
    get_match_goal_events,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CSV_PATH = os.path.join(DATA_DIR, "goals_dataset.csv")
CHECKPOINT_PATH = os.path.join(DATA_DIR, "checkpoint.json")

CSV_FIELDS = [
    "match_id",
    "league",
    "season",
    "match_date",
    "home_team",
    "away_team",
    "final_home_goals",
    "final_away_goals",
    "scoring_team",  # 'home' or 'away'
    "goal_minute",
    "scorer",
    "shot_xg",
]


def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r") as f:
            return json.load(f)
    return {"completed_league_seasons": [], "processed_match_ids": []}


def save_checkpoint(checkpoint):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(checkpoint, f, indent=2)


def ensure_csv_exists():
    if not os.path.exists(CSV_PATH):
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()


def append_rows(rows):
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        for row in rows:
            writer.writerow(row)


def process_match(match_meta, league_name, season):
    """Fetch goal events for one match and turn them into CSV rows."""
    match_id = match_meta["id"]
    if not match_meta.get("isResult"):
        return [], match_id  # match hasn't been played yet

    goals = get_match_goal_events(match_id)

    home_team = match_meta["h"]["title"]
    away_team = match_meta["a"]["title"]
    final_home_goals = int(match_meta["goals"]["h"])
    final_away_goals = int(match_meta["goals"]["a"])
    match_date = match_meta.get("datetime", "")

    rows = []
    if not goals:
        # Match finished 0-0 (or goal data unavailable) -- still record
        # the match itself with no goal rows so it's not silently missing.
        rows.append(
            {
                "match_id": match_id,
                "league": league_name,
                "season": season,
                "match_date": match_date,
                "home_team": home_team,
                "away_team": away_team,
                "final_home_goals": final_home_goals,
                "final_away_goals": final_away_goals,
                "scoring_team": "",
                "goal_minute": "",
                "scorer": "",
                "shot_xg": "",
            }
        )
    else:
        for g in goals:
            rows.append(
                {
                    "match_id": match_id,
                    "league": league_name,
                    "season": season,
                    "match_date": match_date,
                    "home_team": home_team,
                    "away_team": away_team,
                    "final_home_goals": final_home_goals,
                    "final_away_goals": final_away_goals,
                    "scoring_team": "home" if g["side"] == "h" else "away",
                    "goal_minute": g["minute"],
                    "scorer": g["player"],
                    "shot_xg": g["xg"],
                }
            )
    return rows, match_id


def run(full_rebuild=False):
    ensure_csv_exists()
    checkpoint = {"completed_league_seasons": [], "processed_match_ids": []} if full_rebuild else load_checkpoint()
    completed = set(tuple(x) for x in checkpoint["completed_league_seasons"])
    processed_ids = set(checkpoint["processed_match_ids"])

    for league_slug, league_name in LEAGUES.items():
        for season in all_seasons():
            key = (league_slug, season)
            if key in completed:
                continue

            print(f"Fetching {league_name} {season}/{str(season + 1)[-2:]} ...")
            matches = get_matches_for_season(league_slug, season)

            season_fully_played = True
            new_rows = []
            for match_meta in matches:
                match_id = match_meta["id"]
                if match_id in processed_ids:
                    continue
                if not match_meta.get("isResult"):
                    season_fully_played = False
                    continue  # skip fixtures not yet played; will retry next run

                rows, mid = process_match(match_meta, league_name, season)
                new_rows.extend(rows)
                processed_ids.add(mid)

            if new_rows:
                append_rows(new_rows)
                print(f"  -> added {len(new_rows)} goal rows")

            # Only mark a season as fully "completed" (skip forever) once
            # every match in it has been played and processed -- e.g. a
            # past, finished season. The current in-progress season will
            # keep being re-checked on every run to pick up new matches.
            if season_fully_played:
                completed.add(key)

            checkpoint["completed_league_seasons"] = [list(x) for x in completed]
            checkpoint["processed_match_ids"] = list(processed_ids)
            save_checkpoint(checkpoint)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="Force a full historical rebuild, ignoring the checkpoint file.",
    )
    args = parser.parse_args()
    run(full_rebuild=args.full)
