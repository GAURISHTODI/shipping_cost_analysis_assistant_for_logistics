"""The two grading-critical baselines, computed exactly as specified.

vs. own history: trailing 8-week rolling average cost_per_tonne_km for the
    SAME route, using only weeks strictly before the current week. If fewer
    than 8 prior weeks exist, average whatever prior weeks are available
    (never pad, never extrapolate) -- and say so via `own_history_weeks_used`.
    A route's very first observed week has zero prior weeks, so the baseline
    is undefined (NaN) rather than guessed.

vs. similar routes: the average cost_per_tonne_km, in that SAME week,
    across all OTHER routes sharing the same route_type. The route itself is
    excluded from its own peer average. If a route_type has only one route
    in a given week, the peer average is undefined (NaN).
"""
from __future__ import annotations

import pandas as pd

from src.config import ROLLING_WINDOW_WEEKS


def add_own_history_baseline(weekly: pd.DataFrame) -> pd.DataFrame:
    df = weekly.sort_values(["route", "week_of"]).reset_index(drop=True).copy()

    # shift(1) excludes the current week; min_periods=1 allows a partial
    # window while `own_history_weeks_used` records exactly how many prior
    # weeks fed the average. Grouped by route so each route's rolling
    # window never crosses into another route's history.
    shifted = df.groupby("route")["cost_per_tonne_km"].shift(1)
    grouped_shifted = shifted.groupby(df["route"])
    rolling = grouped_shifted.rolling(window=ROLLING_WINDOW_WEEKS, min_periods=1)

    df["own_history_avg"] = rolling.mean().reset_index(level=0, drop=True)
    df["own_history_weeks_used"] = (
        rolling.count().reset_index(level=0, drop=True).fillna(0).astype(int)
    )

    df["vs_own_history_pct"] = (
        df["cost_per_tonne_km"] - df["own_history_avg"]
    ) / df["own_history_avg"]

    return df


def add_peer_baseline(weekly: pd.DataFrame) -> pd.DataFrame:
    df = weekly.copy()

    # Sum and count per (route_type, week_of) so each row can subtract
    # itself out to get the "all OTHER routes" average in O(n).
    grp = df.groupby(["route_type", "week_of"])["cost_per_tonne_km"]
    type_week_sum = grp.transform("sum")
    type_week_count = grp.transform("count")

    other_sum = type_week_sum - df["cost_per_tonne_km"]
    other_count = type_week_count - 1

    df["peer_avg"] = (other_sum / other_count).where(other_count > 0)
    df["peer_routes_used"] = other_count

    df["vs_similar_routes_pct"] = (
        df["cost_per_tonne_km"] - df["peer_avg"]
    ) / df["peer_avg"]

    return df


def compute_baselines(weekly: pd.DataFrame) -> pd.DataFrame:
    df = add_own_history_baseline(weekly)
    df = add_peer_baseline(df)
    return df
