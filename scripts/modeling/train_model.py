"""
Trains a baseline goal-timing model: for each (match, time-bucket, side),
predicts the probability that side scores a goal in that bucket.

Uses a season-based train/test split (not a random row split) -- since
multiple rows come from the same match, a random split would leak
information between train and test. Training on earlier seasons and
testing on the most recent one mimics how you'd actually use this model
in practice: predicting a season you haven't seen yet.

Usage:
    python3 train_model.py --panel ../data/panel.csv
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss
import matplotlib
matplotlib.use("Agg")  # so this works even without a display, e.g. run from terminal
import matplotlib.pyplot as plt

FEATURE_COLS = [
    "bucket_start",
    "team_attack_strength",
    "opponent_defense_strength",
    "score_diff_entering_bucket",
    "side_is_home",
]


def prepare_features(panel):
    panel = panel.copy()
    panel["side_is_home"] = (panel["side"] == "home").astype(int)
    # one-hot encode league so the model can learn league-specific scoring rates
    league_dummies = pd.get_dummies(panel["league"], prefix="league")
    panel = pd.concat([panel, league_dummies], axis=1)
    feature_cols = FEATURE_COLS + list(league_dummies.columns)
    return panel, feature_cols


def train_test_split_by_season(panel):
    seasons = sorted(panel["season"].unique())
    if len(seasons) < 2:
        raise ValueError(
            f"Only found {len(seasons)} season(s) in the data ({seasons}). "
            "Need at least 2 seasons to do a meaningful train/test split by season."
        )
    test_season = seasons[-1]
    train = panel[panel["season"] != test_season]
    test = panel[panel["season"] == test_season]
    print(f"Training on seasons {seasons[:-1]}, testing on season {test_season}")
    return train, test


def evaluate(model, X_test, y_test, panel_test):
    probs = model.predict_proba(X_test)[:, 1]
    brier = brier_score_loss(y_test, probs)
    ll = log_loss(y_test, probs)
    print(f"\nBrier score: {brier:.4f} (lower is better; 0 = perfect, 0.25 = a coin flip on a balanced problem)")
    print(f"Log loss:    {ll:.4f}")

    # Sanity-check plot: does the model's average predicted rate per bucket
    # track the ACTUAL average scoring rate per bucket in the test season?
    check = panel_test.copy()
    check["predicted"] = probs
    by_bucket = check.groupby("bucket_start")[["scored", "predicted"]].mean()

    plt.figure(figsize=(9, 5))
    plt.plot(by_bucket.index, by_bucket["scored"], marker="o", label="Actual scoring rate")
    plt.plot(by_bucket.index, by_bucket["predicted"], marker="o", label="Model's average predicted rate")
    plt.xlabel("Minute (bucket start)")
    plt.ylabel("Probability of a goal in this bucket")
    plt.title("Goal-timing model: predicted vs actual, by minute bucket (test season)")
    plt.legend()
    plt.tight_layout()
    out_path = "model_calibration_by_bucket.png"
    plt.savefig(out_path, dpi=120)
    print(f"Saved calibration plot to {out_path}")

    return brier, ll


def run(panel_path):
    print(f"Loading panel from {panel_path} ...")
    panel = pd.read_csv(panel_path)

    panel, feature_cols = prepare_features(panel)
    train, test = train_test_split_by_season(panel)

    X_train, y_train = train[feature_cols], train["scored"]
    X_test, y_test = test[feature_cols], test["scored"]

    print(f"\nTraining rows: {len(X_train)}, Test rows: {len(X_test)}")
    print("Training HistGradientBoostingClassifier ...")

    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    )
    model.fit(X_train, y_train)

    evaluate(model, X_test, y_test, test)

    # Simple feature importance via permutation isn't included here to keep
    # runtime short -- HistGradientBoostingClassifier doesn't expose
    # feature_importances_ directly, that's a natural next step to add.

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="../data/panel.csv")
    args = parser.parse_args()
    run(args.panel)
