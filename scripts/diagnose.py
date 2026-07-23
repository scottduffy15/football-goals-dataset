"""
Diagnostic script -- confirms the two live JSON endpoints work end-to-end:
    getLeagueData/<league>/<season>
    getMatchData/<match_id>

Run with:
    python3 diagnose.py
"""

import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://understat.com/",
}

print("1) Requesting league data: EPL, 2023 season...")
url = "https://understat.com/getLeagueData/EPL/2023"
resp = requests.get(url, headers=HEADERS, timeout=30)
print(f"   Status code: {resp.status_code}")
data = resp.json()
print(f"   Top-level keys: {list(data.keys())}")
dates = data.get("dates", [])
print(f"   Number of matches found: {len(dates)}")

# find the first finished match to test the match endpoint with
first_result = next((m for m in dates if m.get("isResult")), None)
if not first_result:
    print("   No finished matches found -- can't test match endpoint.")
else:
    print(f"   First finished match: {first_result['h']['title']} vs {first_result['a']['title']}"
          f" ({first_result['goals']['h']}-{first_result['goals']['a']}), id={first_result['id']}")

    print(f"\n2) Requesting match data for match id {first_result['id']}...")
    match_url = f"https://understat.com/getMatchData/{first_result['id']}"
    match_resp = requests.get(match_url, headers=HEADERS, timeout=30)
    print(f"   Status code: {match_resp.status_code}")
    match_data = match_resp.json()
    print(f"   Top-level keys: {list(match_data.keys())}")
    shots = match_data.get("shots", {})
    goals = [s for side in ("h", "a") for s in shots.get(side, []) if s.get("result") == "Goal"]
    print(f"   Number of goals found in this match: {len(goals)}")
    for g in goals:
        print(f"     minute {g['minute']}: {g['player']} ({'home' if g['h_a']=='h' else 'away'})")

print("\nDone. Please copy/paste this entire output back to Claude.")
