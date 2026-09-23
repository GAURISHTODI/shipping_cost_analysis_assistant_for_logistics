"""Load shipment records and compute weekly cost-per-tonne-km per route.

A "route" is the (origin, destination) pair; route_type (Short/Medium/Long)
is taken as-is from the data, not re-derived.
"""
from __future__ import annotations

import pandas as pd

from src.config import SHIPMENTS_CSV


def load_shipments(path=SHIPMENTS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["shipment_date"])
    df["route"] = df["origin"] + "-" + df["destination"]

    # Monday of the ISO week containing shipment_date.
    df["week_of"] = (
        df["shipment_date"] - pd.to_timedelta(df["shipment_date"].dt.weekday, unit="D")
    ).dt.normalize()

    return df


def weekly_route_metrics(shipments: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to one row per (route, route_type, week_of).

    cost_per_tonne_km is computed on the *aggregated* week totals
    (sum(freight_cost) / sum(quantity_tonnes * distance_km)), i.e. a
    tonne-km-weighted average across every shipment in that route's week --
    not a plain average of each shipment's individual ratio. This keeps a
    handful of very small shipments from skewing the week's figure, and it
    matches the formula in the brief ("total freight cost / (quantity x
    distance)") applied at the weekly-total level.
    """
    s = shipments.copy()
    s["tonne_km"] = s["quantity_tonnes"] * s["distance_km"]

    grouped = (
        s.groupby(["route", "route_type", "week_of"], as_index=False)
        .agg(
            total_freight_cost=("freight_cost_inr", "sum"),
            total_tonne_km=("tonne_km", "sum"),
            shipment_count=("shipment_id", "count"),
        )
    )
    grouped["cost_per_tonne_km"] = (
        grouped["total_freight_cost"] / grouped["total_tonne_km"]
    )
    return grouped.sort_values(["route", "week_of"]).reset_index(drop=True)
