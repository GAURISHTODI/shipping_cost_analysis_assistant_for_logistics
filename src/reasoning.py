"""Turn a verdict (+ its grounded note, if any) into the output row's text.

Deterministic and template-based by default: zero LLM calls, zero cost,
byte-identical across runs -- which is what the brief's reproducibility and
"run it responsibly" guardrails actually reward. An optional LLM polish
pass exists in src/llm.py and can rewrite `reason` into looser prose, but
it is never allowed to change `flagged`, `matched_note_id`, or the
percentages: those are fixed before the LLM ever sees the row, so a bad
LLM output can make the writing worse but can never fabricate a verdict.
See README "Why no LLM by default" for the full reasoning.
"""
from __future__ import annotations

import pandas as pd

from src.grounding import GroundedMatch
from src.notes_parser import ParsedNote
from src.retrieval import VectorStore

MIN_RELEVANCE_TO_MENTION = 0.05


def format_pct(pct: float | None, suffix: str) -> str:
    if pct is None or pd.isna(pct):
        return "N/A (no prior weeks)" if "past average" in suffix else "N/A (no peer routes)"
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct * 100:.1f}% {suffix}"


def _why_not_justifying(note: ParsedNote, route: str, week_of: pd.Timestamp) -> str:
    if not note.matches_route(route):
        return f"it applies to {note.applies_to_raw}, not {route}"
    if not note.covers_week(week_of):
        return (
            f"its effective window ({note.window_start.date()} to "
            f"{note.window_end.date()}) does not cover the week of {week_of.date()}"
        )
    return "it does not describe a reason for a cost rise on this route"


def explain_unflagged_or_ordinary() -> str:
    return "Cost is within the expected range versus both this route's history and its peer routes; no anomaly detected."


def explain_justified(match: GroundedMatch, route: str, week_of: pd.Timestamp) -> str:
    n = match.note
    return (
        f"Matches note {n.note_id} dated {n.date.date()}: {n.text.strip()} "
        f"This covers the week of {week_of.date()} on {route}, so the cost rise has a clear explanation."
    )


def explain_unexplained(
    store: VectorStore, notes: list[ParsedNote], route: str, week_of: pd.Timestamp
) -> str:
    if not notes:
        return "No matching note found for this route or date range. Cost rise looks unexplained and worth a human review."

    query = f"cost rise on route {route} week of {week_of.date()}"
    best = store.query(query, top_k=1)
    if not best or best[0].score < MIN_RELEVANCE_TO_MENTION:
        return "No matching note found for this route or date range. Cost rise looks unexplained and worth a human review."

    n = best[0].note
    why = _why_not_justifying(n, route, week_of)
    return (
        f"The closest note ({n.note_id}, {n.date.date()}) says: \"{n.text.strip()}\" "
        f"-- but {why}. No genuine justification found; flagged for review."
    )
