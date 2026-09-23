"""Deterministic, rule-based verification of every "justified" verdict.

This module -- not the retriever, not an LLM -- is what decides whether a
route-week is allowed to be marked justified. It re-checks every candidate
note against three hard requirements straight from the brief:
    1. the note must apply to THIS route (exact match, or "All Routes")
    2. the note's window must overlap THIS week
    3. the note must actually describe a cost-INCREASING event (polarity)
Any candidate failing any check is discarded. If nothing survives, the
route-week is "unexplained" -- never a guess.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.notes_parser import ParsedNote
from src.retrieval import VectorStore


@dataclass
class GroundedMatch:
    note: ParsedNote
    retrieval_score: float


def find_candidate_notes(notes: list[ParsedNote], route: str, week_of: pd.Timestamp) -> list[ParsedNote]:
    """Notes that pass all three hard requirements for this route-week."""
    return [
        n
        for n in notes
        if n.matches_route(route) and n.covers_week(week_of) and n.is_justifying()
    ]


def ground(
    store: VectorStore,
    notes: list[ParsedNote],
    route: str,
    week_of: pd.Timestamp,
) -> GroundedMatch | None:
    candidates = find_candidate_notes(notes, route, week_of)
    if not candidates:
        return None

    if len(candidates) == 1:
        return GroundedMatch(candidates[0], retrieval_score=1.0)

    # More than one verified candidate (rare in this dataset): use retrieval
    # to rank by relevance, preferring an exact-route note over an
    # "All Routes" note, then highest similarity, then lowest note_id for a
    # stable tie-break.
    query = f"cost rise on route {route} week of {week_of.date()}"
    scored = {sn.note.note_id: sn.score for sn in store.query(query, top_k=len(notes))}

    def sort_key(n: ParsedNote):
        return (0 if not n.applies_to_all else 1, -scored.get(n.note_id, 0.0), n.note_id)

    best = sorted(candidates, key=sort_key)[0]
    return GroundedMatch(best, retrieval_score=scored.get(best.note_id, 0.0))
