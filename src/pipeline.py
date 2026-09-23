"""End-to-end orchestration: shipments -> weekly metrics -> baselines ->
candidate flags -> note grounding -> final output rows.

Column contract (must match data/sample_output_format_v2.csv exactly):
    route, week_of, cost_per_tonne_km, vs_own_history, vs_similar_routes,
    flagged, matched_note_id, reason
"""
from __future__ import annotations

import pandas as pd

from src.baselines import compute_baselines
from src.flagging import add_candidate_flag
from src.grounding import ground
from src.llm import LLMPolisher
from src.notes_parser import load_notes
from src.reasoning import (
    explain_justified,
    explain_unexplained,
    explain_unflagged_or_ordinary,
    format_pct,
)
from src.retrieval import VectorStore
from src.weekly_metrics import load_shipments, weekly_route_metrics

OUTPUT_COLUMNS = [
    "route",
    "week_of",
    "cost_per_tonne_km",
    "vs_own_history",
    "vs_similar_routes",
    "flagged",
    "matched_note_id",
    "reason",
]


def build_output(use_llm: bool = False, llm_model: str = "llama3.2") -> tuple[pd.DataFrame, LLMPolisher]:
    shipments = load_shipments()
    weekly = weekly_route_metrics(shipments)
    weekly = compute_baselines(weekly)
    weekly = add_candidate_flag(weekly)

    notes = load_notes()
    store = VectorStore(notes)
    polisher = LLMPolisher(enabled=use_llm, model=llm_model)

    rows = []
    for row in weekly.itertuples(index=False):
        route, week_of = row.route, row.week_of

        if not row.is_candidate:
            flagged = "No"
            matched_note_id = ""
            reason = explain_unflagged_or_ordinary()
        else:
            match = ground(store, notes, route, week_of)
            if match is None:
                flagged = "Yes"
                matched_note_id = ""
                reason = explain_unexplained(store, notes, route, week_of)
            else:
                flagged = "No (justified)"
                matched_note_id = match.note.note_id
                reason = explain_justified(match, route, week_of)
                reason = polisher.polish(reason, match.note.text, flagged)

        rows.append(
            {
                "route": route,
                "week_of": week_of.date().isoformat(),
                "cost_per_tonne_km": round(row.cost_per_tonne_km, 2),
                "vs_own_history": format_pct(row.vs_own_history_pct, "vs this route's past average"),
                "vs_similar_routes": format_pct(row.vs_similar_routes_pct, "vs similar-length routes this week"),
                "flagged": flagged,
                "matched_note_id": matched_note_id,
                "reason": reason,
            }
        )

    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    return out, polisher
