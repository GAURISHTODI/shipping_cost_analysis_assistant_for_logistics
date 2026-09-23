import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.grounding import ground
from src.notes_parser import load_notes
from src.retrieval import VectorStore


def test_wrong_route_never_matches():
    notes = load_notes()
    store = VectorStore(notes)
    # N002 is Ahmedabad-Mumbai only; must not match a different route even
    # in the exact same week.
    match = ground(store, notes, "Mumbai-Pune", pd.Timestamp("2025-01-20"))
    assert match is None


def test_negative_note_never_justifies_even_exact_route_and_date():
    notes = load_notes()
    store = VectorStore(notes)
    match = ground(store, notes, "Mumbai-Delhi", pd.Timestamp("2024-07-29"))
    assert match is None  # N005 says costs were not significantly affected


def test_all_routes_positive_note_can_justify_any_route_in_window():
    notes = load_notes()
    store = VectorStore(notes)
    match = ground(store, notes, "Delhi-Chennai", pd.Timestamp("2025-05-05"))
    assert match is not None
    assert match.note.note_id == "N003"


def test_vague_window_does_not_reach_into_later_weeks():
    notes = load_notes()
    store = VectorStore(notes)
    # N003 has no explicit end date; must not justify a week two weeks later.
    match = ground(store, notes, "Delhi-Chennai", pd.Timestamp("2025-05-19"))
    assert match is None
