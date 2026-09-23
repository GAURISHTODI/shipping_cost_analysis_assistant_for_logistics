import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.baselines import add_own_history_baseline, add_peer_baseline


def _weekly(route, route_type, weeks, costs):
    return pd.DataFrame(
        {
            "route": [route] * len(weeks),
            "route_type": [route_type] * len(weeks),
            "week_of": pd.to_datetime(weeks),
            "cost_per_tonne_km": costs,
        }
    )


def test_own_history_no_look_ahead():
    df = _weekly(
        "A-B", "Short",
        ["2024-01-01", "2024-01-08", "2024-01-15"],
        [10.0, 20.0, 100.0],
    )
    out = add_own_history_baseline(df)
    first, second, third = out.iloc[0], out.iloc[1], out.iloc[2]

    assert pd.isna(first["own_history_avg"])  # no prior weeks at all
    assert first["own_history_weeks_used"] == 0

    assert second["own_history_avg"] == 10.0  # only week 1 is "prior"
    assert second["own_history_weeks_used"] == 1

    # crucially: week 3's baseline must NOT include week 3's own 100.0
    assert third["own_history_avg"] == 15.0  # avg(10, 20), not avg(10,20,100)
    assert third["own_history_weeks_used"] == 2


def test_own_history_uses_at_most_8_prior_weeks():
    weeks = pd.date_range("2024-01-01", periods=10, freq="7D").strftime("%Y-%m-%d").tolist()
    costs = list(range(1, 11))  # 1..10
    df = _weekly("A-B", "Short", weeks, costs)
    out = add_own_history_baseline(df)

    tenth = out.iloc[9]
    # prior weeks are costs 1..9; trailing 8 of those are 2..9
    assert tenth["own_history_weeks_used"] == 8
    assert abs(tenth["own_history_avg"] - sum(range(2, 10)) / 8) < 1e-9


def test_peer_average_excludes_self():
    df = pd.concat(
        [
            _weekly("A-B", "Short", ["2024-01-01"], [10.0]),
            _weekly("C-D", "Short", ["2024-01-01"], [20.0]),
            _weekly("E-F", "Short", ["2024-01-01"], [30.0]),
        ],
        ignore_index=True,
    )
    out = add_peer_baseline(df)
    a_row = out[out["route"] == "A-B"].iloc[0]
    # peers of A-B are C-D (20) and E-F (30) -> avg 25, NOT including A-B's own 10
    assert a_row["peer_avg"] == 25.0
    assert a_row["peer_routes_used"] == 2


def test_peer_average_nan_when_no_peers():
    df = _weekly("A-B", "Short", ["2024-01-01"], [10.0])
    out = add_peer_baseline(df)
    assert pd.isna(out.iloc[0]["peer_avg"])
