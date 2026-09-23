import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.weekly_metrics import load_shipments, weekly_route_metrics


def test_week_of_is_monday():
    shipments = load_shipments()
    assert (shipments["week_of"].dt.weekday == 0).all()


def test_week_of_never_after_shipment_date():
    shipments = load_shipments()
    assert (shipments["week_of"] <= shipments["shipment_date"]).all()
    assert (shipments["shipment_date"] - shipments["week_of"]).dt.days.max() <= 6


def test_cost_per_tonne_km_matches_manual_formula():
    rows = pd.DataFrame(
        {
            "shipment_id": ["A", "B"],
            "origin": ["X", "X"],
            "destination": ["Y", "Y"],
            "route_type": ["Short", "Short"],
            "material": ["m", "m"],
            "quantity_tonnes": [10.0, 5.0],
            "distance_km": [100.0, 100.0],
            "freight_cost_inr": [1000.0, 600.0],
            "shipment_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),  # both Mon-week
            "transporter": ["t", "t"],
        }
    )
    rows["route"] = rows["origin"] + "-" + rows["destination"]
    rows["week_of"] = rows["shipment_date"] - pd.to_timedelta(rows["shipment_date"].dt.weekday, unit="D")

    out = weekly_route_metrics(rows)
    assert len(out) == 1
    expected = (1000.0 + 600.0) / (10.0 * 100.0 + 5.0 * 100.0)
    assert abs(out.iloc[0]["cost_per_tonne_km"] - expected) < 1e-9


def test_no_duplicate_route_week_rows():
    weekly = weekly_route_metrics(load_shipments())
    assert not weekly.duplicated(subset=["route", "week_of"]).any()
