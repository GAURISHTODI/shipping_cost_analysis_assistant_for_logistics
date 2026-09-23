"""Decide which route-weeks are "rising & out of the ordinary" candidates.

A row is a candidate if it is rising *and* anomalous against either
baseline -- i.e. vs_own_history_pct or vs_similar_routes_pct exceeds
ANOMALY_THRESHOLD_PCT. Only candidates go on to note-grounding; ordinary
weeks are reported (for full auditability) but never need a reason.
"""
from __future__ import annotations

import pandas as pd

from src.config import ANOMALY_THRESHOLD_PCT


def add_candidate_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    own_rise = df["vs_own_history_pct"].fillna(-float("inf")) > ANOMALY_THRESHOLD_PCT
    peer_rise = df["vs_similar_routes_pct"].fillna(-float("inf")) > ANOMALY_THRESHOLD_PCT

    df["is_candidate"] = own_rise | peer_rise
    return df
