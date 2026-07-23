"""
Core scraping functions for understat.com

REBUILT VERSION: Understat's site no longer embeds match data as JSON
inside <script> tags (the old, previously-documented approach). Instead,
the page's own JavaScript calls two internal JSON endpoints directly:

    https://understat.com/getLeagueData/<league>/<season>
    https://understat.com/getMatchData/<match_id>

These were found by inspecting Chrome DevTools -> Network -> Fetch/XHR
while browsing the site, and confirmed by manually checking their
response structure. This version calls those endpoints directly instead
of parsing HTML, which is both simpler and more robust.
"""

import time

import requests

BASE_URL = "https://understat.com"

# The five major European leagues (Understat's internal slugs).
# Russian Premier League intentionally excluded per requirements.
LEAGUES = {
    "EPL": "Premier League",
    "La_liga": "La Liga",
    "Bundesliga": "Bundesliga",
    "Serie_A": "Serie A",
    "Ligue_1": "Ligue 1",
}

# Understat's data starts at the 2014/15 season.
EARLIEST_SEASON = 2014

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    # These two headers mimic the AJAX request the site's own page makes
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://understat.com/",
}

REQUEST_DELAY_SECONDS = 1.0  # be polite -- understat has no official API or published rate limit


def current_season_start_year():
    """
    Returns the season 'start year' that is currently in progress or most
    recently completed. Understat labels a season by the year it starts
    (e.g. the 2025/26 season is '2025'). European seasons run roughly
    August to May, so:
      - Jan-Jun -> season that started the previous year
      - Jul-Dec -> season starting this year (may not have data yet if
        the season hasn't kicked off)
    """
    from datetime import datetime

    today = datetime.utcnow()
    if today.month >= 7:
        return today.year
    return today.year - 1


def all_seasons():
    """All season start-years from 2014 up to the current one, inclusive."""
    return list(range(EARLIEST_SEASON, current_season_start_year() + 1))


def get_matches_for_season(league_slug, season):
    """
    Fetch the list of matches (with basic metadata) for a given league/season
    via understat's internal getLeagueData JSON endpoint.

    Returns a list of dicts, each containing at least:
        id, datetime, isResult, h {id, title, short_title},
        a {id, title, short_title}, goals {h, a}
    """
    url = f"{BASE_URL}/getLeagueData/{league_slug}/{season}"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY_SECONDS)

    data = resp.json()
    return data.get("dates", [])


def get_match_goal_events(match_id):
    """
    Fetch shot-level data for a single match via understat's internal
    getMatchData JSON endpoint, and return only the shots that resulted
    in a goal, each with its minute.

    Returns a list of dicts:
        {minute, side ('h' or 'a'), player, xg}
    """
    url = f"{BASE_URL}/getMatchData/{match_id}"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY_SECONDS)

    data = resp.json()
    shots = data.get("shots", {})

    goals = []
    # shots has the shape {"h": [...], "a": [...]}
    for side in ("h", "a"):
        for shot in shots.get(side, []):
            if shot.get("result") == "Goal":
                goals.append(
                    {
                        "minute": int(shot.get("minute", -1)),
                        "side": side,  # 'h' = home team scored, 'a' = away team scored
                        "player": shot.get("player"),
                        "xg": shot.get("xG"),
                    }
                )
    return goals
