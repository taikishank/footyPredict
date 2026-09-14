"""Builds a per-fixture table of de-vigged bookmaker closing probabilities,
so the model can be scored against the market on the archive rather than on
the handful of fixtures the live pipeline has graded so far.

Why this source. The project already pulls prices from The Odds API
(ingestor-go/internal/odds), but only for *upcoming* fixtures - there is no
historical odds anywhere in the archive, which carries has_odds as a flag and
no payload. The Odds API does sell history, and it was priced before this was
written: its archive starts 2020-06-06, which reaches 56.6% of the rolling
origin validation set, snapshots cost 10 quota credits each, and one snapshot
per league per match-day over that span is ~36,900 credits - a paid tier, and
a partial answer.

football-data.co.uk publishes the same five leagues back to the 1990s as free
CSVs, including Pinnacle's *closing* prices. That covers 18,907 of the 19,797
validation fixtures (96%) at no cost, and closing Pinnacle is the sharper
benchmark besides - it is the price after the market has absorbed team news,
which is the number a model has to beat to be worth anything.

Two columns per method are emitted, not one. See market_lib.devig_shin.

Output: data/features/market.csv, one row per matched fixture, joined onto
anything else by fixture_id - the same separate-table discipline
build_rating_features.py keeps, so nothing the served model trains on moves.

Run:
    ./inference-py/bin/python ml/build_market_index.py --download
    ./inference-py/bin/python ml/build_market_index.py --diagnose
"""
from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import pandas as pd

from market_lib import DEVIG_METHODS, learn_name_map, overround

RAW_DIR = Path(__file__).parent / "data" / "raw" / "market"
FEATURES_DIR = Path(__file__).parent / "data" / "features"
INDEX_PATH = FEATURES_DIR / "results_index.parquet"
OUT_PATH = FEATURES_DIR / "market.csv"

BASE_URL = "https://www.football-data.co.uk/mmz4281"
# football-data.co.uk serves an empty body to a request with no User-Agent,
# with a 200 and no error - so an absent one looks like a season with no
# fixtures rather than like a failure.
USER_AGENT = "Mozilla/5.0 (compatible; footyPredict research)"
DOWNLOAD_PAUSE_SECONDS = 0.5

# football-data's league file codes, against the SportMonks league_ids the
# rest of the pipeline uses (sm_league_id_to_raw_name_mapping.txt). Same five
# leagues as ingestor-go/internal/odds/types.go's leagueSportKeys.
LEAGUE_CODES = {
    "E0": 8,      # Premier League
    "D1": 82,     # Bundesliga
    "F1": 301,    # Ligue 1
    "I1": 384,    # Serie A
    "SP1": 564,   # La Liga
}

# Seasons to pull, as football-data spells them. The rolling-origin
# validation set starts 2015-08-09, so this is its span plus the current
# season; earlier seasons exist upstream but would only add rows the
# benchmark never scores.
SEASONS = ["1516", "1617", "1718", "1819", "1920", "2021",
           "2122", "2223", "2324", "2425", "2526"]

# Price columns, best first. Pinnacle closing is the benchmark; the others
# are fallbacks for the rows and seasons where it is missing, and are kept
# distinguishable on the row so a later analysis can restrict to one source.
PRICE_SETS = [
    ("pinnacle_closing", ("PSCH", "PSCD", "PSCA")),
    ("pinnacle_open", ("PSH", "PSD", "PSA")),
    ("market_avg_closing", ("AvgCH", "AvgCD", "AvgCA")),
    ("bet365_open", ("B365H", "B365D", "B365A")),
]

# How far a football-data date may sit from a SportMonks starting_at and
# still be the same fixture. The two disagree by a day around late kickoffs
# in different timezones; a day either side covers that, and a league does
# not play the same pair of clubs twice inside three days.
DATE_TOLERANCE_DAYS = 1


def download() -> None:
    """Fetches each league-season CSV that isn't already on disk."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fetched = 0
    for season in SEASONS:
        for code in LEAGUE_CODES:
            path = RAW_DIR / f"{code}-{season}.csv"
            if path.exists() and path.stat().st_size > 0:
                continue
            url = f"{BASE_URL}/{season}/{code}.csv"
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    body = response.read()
            except urllib.error.URLError as exc:
                print(f"  {code}-{season}: {exc}")
                continue
            if not body:
                print(f"  {code}-{season}: empty response")
                continue
            path.write_bytes(body)
            fetched += 1
            time.sleep(DOWNLOAD_PAUSE_SECONDS)
    print(f"downloaded {fetched} new file(s) into {RAW_DIR}")


def load_prices() -> pd.DataFrame:
    """Every downloaded league-season CSV, as one frame of dates and prices."""
    frames = []
    for path in sorted(RAW_DIR.glob("*.csv")):
        code = path.name.split("-")[0]
        if code not in LEAGUE_CODES:
            continue
        # latin-1 rather than utf-8: the older files carry accented club names
        # in a single-byte encoding. on_bad_lines because a few seasons have
        # trailing blank-ish rows with a different column count.
        frame = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
        if "HomeTeam" not in frame.columns:
            continue
        frame = frame[frame["HomeTeam"].notna()].copy()
        frame["league_id"] = LEAGUE_CODES[code]
        # format="mixed" because the day/month/year separator and the year's
        # width both change across seasons; dayfirst because it is a UK source.
        frame["date"] = pd.to_datetime(
            frame["Date"], format="mixed", dayfirst=True, errors="coerce"
        )
        for _, columns in PRICE_SETS:
            for column in columns:
                if column not in frame.columns:
                    frame[column] = float("nan")
        keep = ["league_id", "date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
        keep += [c for _, cols in PRICE_SETS for c in cols]
        frames.append(frame[keep])

    if not frames:
        raise SystemExit(f"no CSVs in {RAW_DIR} - run with --download first")
    prices = pd.concat(frames, ignore_index=True).dropna(subset=["date"])
    prices["day"] = prices["date"].dt.normalize()
    return prices.reset_index(drop=True)


def _day_buckets(frame: pd.DataFrame) -> dict:
    buckets = defaultdict(list)
    for position, day in zip(frame.index, frame["day"]):
        buckets[day].append(position)
    return buckets


def _nearby(row, buckets: dict):
    for offset in range(-DATE_TOLERANCE_DAYS, DATE_TOLERANCE_DAYS + 1):
        yield from buckets.get(row.day + timedelta(days=offset), ())


def match(index: pd.DataFrame, prices: pd.DataFrame) -> tuple[dict, list]:
    """fixture_id -> row position in `prices`, via a learned club-name map.

    Two passes. The first matches only where league, date and scoreline
    identify exactly one candidate, consulting no names at all, and harvests
    what those matches imply each club is called on the other side. The
    second maps every fixture's club names through what was learned and
    matches on names and date, with no scoreline condition - so a fixture
    whose scoreline was ambiguous (three 1-0s in a league on one Saturday)
    or whose two sources disagree about the score is still matched.

    See market_lib.learn_name_map for why the mapping is learned.
    """
    pairs = []
    for league_id in LEAGUE_CODES.values():
        fixtures = index[index["league_id"] == league_id]
        candidates = prices[prices["league_id"] == league_id]
        buckets = _day_buckets(candidates)
        for row in fixtures.itertuples():
            hits = [
                position
                for position in _nearby(row, buckets)
                if prices.at[position, "FTHG"] == row.home_goals
                and prices.at[position, "FTAG"] == row.away_goals
            ]
            if len(hits) == 1:
                position = hits[0]
                pairs.append((league_id, row.home_name, prices.at[position, "HomeTeam"]))
                pairs.append((league_id, row.away_name, prices.at[position, "AwayTeam"]))

    name_map, ambiguous = learn_name_map(pairs)
    matched: dict[int, int] = {}
    # A price row is consumed once, so two fixtures cannot claim the same
    # match - which is what stops a postponed-and-replayed pairing inside the
    # date window from being counted twice.
    used: set[int] = set()
    for league_id in LEAGUE_CODES.values():
        fixtures = index[index["league_id"] == league_id]
        candidates = prices[prices["league_id"] == league_id]
        buckets = _day_buckets(candidates)
        for row in fixtures.itertuples():
            home = name_map.get((league_id, row.home_name))
            away = name_map.get((league_id, row.away_name))
            if home is None or away is None:
                continue
            for position in _nearby(row, buckets):
                if position in used:
                    continue
                if (prices.at[position, "HomeTeam"] == home
                        and prices.at[position, "AwayTeam"] == away):
                    used.add(position)
                    matched[row.fixture_id] = position
                    break

    print(f"learned {len(name_map)} club names, {len(ambiguous)} ambiguous")
    for key, counts in ambiguous:
        print(f"  ambiguous {key}: {counts}")
    return matched, ambiguous


def build(index: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    matched, _ = match(index, prices)
    joined = prices.loc[list(matched.values())].reset_index(drop=True)
    joined.insert(0, "fixture_id", list(matched.keys()))

    rows = []
    for row in joined.itertuples():
        record = {"fixture_id": row.fixture_id, "date": row.date,
                  "league_id": row.league_id}
        for source, columns in PRICE_SETS:
            odds = tuple(getattr(row, column) for column in columns)
            if any(pd.isna(o) or o <= 1.0 for o in odds):
                continue
            record[f"{source}_overround"] = overround(odds) - 1.0
            for method, devig in DEVIG_METHODS.items():
                home, draw, away = devig(odds)
                record[f"{source}_{method}_home"] = home
                record[f"{source}_{method}_draw"] = draw
                record[f"{source}_{method}_away"] = away
        rows.append(record)
    return pd.DataFrame(rows)


def diagnose(index: pd.DataFrame, market: pd.DataFrame) -> None:
    """Coverage and sanity, in the shape build_rating_features.py --diagnose
    established: the numbers that would make the benchmark wrong without
    making it look wrong."""
    merged = index.merge(market[["fixture_id"]], on="fixture_id", how="left",
                         indicator=True)
    print(f"\n{len(market)} of {len(index)} archive fixtures priced "
          f"({len(market) / len(index) * 100:.1f}%)")

    missing = merged[merged["_merge"] == "left_only"]
    print(f"\nunpriced fixtures by season:")
    by_year = missing["date"].dt.year.value_counts().sort_index()
    for year, count in by_year.items():
        print(f"  {year}: {count}")

    print("\ncoverage by price source:")
    for source, _ in PRICE_SETS:
        column = f"{source}_shin_home"
        if column not in market.columns:
            print(f"  {source:20s} absent")
            continue
        have = market[column].notna().sum()
        print(f"  {source:20s} {have:6d} rows ({have / len(market) * 100:5.1f}%)")

    print("\nmean overround (the margin being removed):")
    for source, _ in PRICE_SETS:
        column = f"{source}_overround"
        if column in market.columns and market[column].notna().any():
            print(f"  {source:20s} {market[column].mean() * 100:5.2f}%")

    # The rating phase found three separate bugs by correlating every emitted
    # column against the fixture date, none of which showed up in a log loss.
    # A price is not a fitted quantity so it cannot drift the same way, but a
    # *coverage* trend would silently reweight the benchmark toward one era.
    print("\nr(date) per column - a trend here means an era effect:")
    numeric = market.select_dtypes("number").drop(columns=["fixture_id", "league_id"])
    date_ordinal = market["date"].map(pd.Timestamp.toordinal)
    for column in numeric.columns:
        r = numeric[column].corr(date_ordinal)
        flag = "  <-- check" if abs(r) > 0.15 else ""
        print(f"  {column:42s} {r:+.3f}{flag}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true",
                        help="fetch any league-season CSVs not already on disk")
    parser.add_argument("--diagnose", action="store_true",
                        help="print coverage and sanity checks after building")
    args = parser.parse_args()

    if args.download:
        download()

    index = pd.read_parquet(INDEX_PATH)
    index = index[index["league_id"].isin(LEAGUE_CODES.values())].copy()
    index["date"] = pd.to_datetime(index["date"])
    index = index[index["date"] >= "2015-07-01"].reset_index(drop=True)
    index["day"] = index["date"].dt.normalize()

    prices = load_prices()
    print(f"{len(prices)} priced fixtures from football-data, "
          f"{len(index)} archive fixtures in span")

    market = build(index, prices)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    market.to_csv(OUT_PATH, index=False)
    print(f"wrote {len(market)} rows to {OUT_PATH}")

    if args.diagnose:
        diagnose(index, market)


if __name__ == "__main__":
    main()
