import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.notes_parser import load_notes


def _note(notes, note_id):
    return next(n for n in notes if n.note_id == note_id)


def test_all_ten_notes_parse():
    notes = load_notes()
    assert len(notes) == 10


def test_explicit_range_parsed_for_n001():
    n = _note(load_notes(), "N001")
    assert n.window_start == pd.Timestamp("2025-02-24")
    assert n.window_end == pd.Timestamp("2025-03-08")
    assert n.polarity == "positive"
    assert n.applies_to_route == "Chennai-Bangalore"


def test_default_single_week_window_for_n002():
    n = _note(load_notes(), "N002")
    assert n.window_start == pd.Timestamp("2025-01-20")  # already a Monday
    assert n.window_end == pd.Timestamp("2025-01-26")
    assert n.polarity == "positive"


def test_negative_polarity_notes():
    notes = load_notes()
    negative_ids = {"N004", "N005", "N006", "N007", "N008", "N009", "N010"}
    for nid in negative_ids:
        assert _note(notes, nid).polarity == "negative", nid


def test_n004_never_matches_any_route_despite_all_routes_label():
    n = _note(load_notes(), "N004")
    assert n.applies_to_all is False
    assert n.applies_to_route is None


def test_covers_week_overlap_logic():
    n = _note(load_notes(), "N001")
    assert n.covers_week(pd.Timestamp("2025-03-03"))  # inside range
    assert not n.covers_week(pd.Timestamp("2025-03-10"))  # after range end
    assert not n.covers_week(pd.Timestamp("2025-02-17"))  # before range start
