"""Turn context_notes.csv into structured, verifiable facts.

Each note is parsed into:
  - applies_to_all (bool): "All Routes"
  - applies_to_route (str | None): a specific "Origin-Destination" route
  - polarity ("positive" | "negative"): does the note's text actually
    describe a cost-INCREASING event? A note can mention the right route
    and the right dates and still be polarity="negative" (e.g. "costs were
    not significantly affected") -- those must never justify a flag.
  - window (start_date, end_date): the date range the note's effect covers.

Window rule (see config.py for the full rationale): if the note text
contains an explicit "<Month> <day> to <Month> <day>" range, use it exactly.
Otherwise the effect is assumed to last only the ISO week (Mon-Sun)
containing the note's own `date` field -- vague phrases like "starting this
week" do NOT extend the window indefinitely.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from src.config import NEGATIVE_PATTERNS, NOTES_CSV, POSITIVE_PATTERNS

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_RANGE_RE = re.compile(
    r"\b(" + "|".join(_MONTHS) + r")[a-z]*\.?\s+(\d{1,2})\s+to\s+"
    r"(" + "|".join(_MONTHS) + r")[a-z]*\.?\s+(\d{1,2})\b",
    re.IGNORECASE,
)

_NEG_RE = re.compile("|".join(NEGATIVE_PATTERNS), re.IGNORECASE)
_POS_RE = re.compile("|".join(POSITIVE_PATTERNS), re.IGNORECASE)


@dataclass(frozen=True)
class ParsedNote:
    note_id: str
    date: pd.Timestamp
    applies_to_raw: str
    applies_to_all: bool
    applies_to_route: str | None
    polarity: str  # "positive" or "negative"
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    text: str

    def covers_week(self, week_of: pd.Timestamp) -> bool:
        week_end = week_of + pd.Timedelta(days=6)
        return week_of <= self.window_end and week_end >= self.window_start

    def matches_route(self, route: str) -> bool:
        return self.applies_to_all or self.applies_to_route == route

    def is_justifying(self) -> bool:
        return self.polarity == "positive"


def _week_monday(ts: pd.Timestamp) -> pd.Timestamp:
    return (ts - pd.Timedelta(days=ts.weekday())).normalize()


def _parse_window(note_date: pd.Timestamp, text: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    m = _RANGE_RE.search(text)
    if not m:
        start = _week_monday(note_date)
        return start, start + pd.Timedelta(days=6)

    mon1, day1, mon2, day2 = m.groups()
    year = note_date.year
    start = pd.Timestamp(year=year, month=_MONTHS[mon1[:3].lower()], day=int(day1))
    end = pd.Timestamp(year=year, month=_MONTHS[mon2[:3].lower()], day=int(day2))
    if end < start:
        end = end.replace(year=year + 1)
    return start, end


def _parse_polarity(text: str) -> str:
    if _NEG_RE.search(text):
        return "negative"
    if _POS_RE.search(text):
        return "positive"
    # No explicit cost-increasing language and no explicit negation:
    # default to negative (non-justifying). Per the brief's guardrail --
    # "no supporting note -> unexplained; do not guess" -- an ambiguous
    # note must never be treated as justification.
    return "negative"


def _parse_row(row: pd.Series) -> ParsedNote:
    applies_to_raw = str(row["applies_to"]).strip()
    applies_to_all = applies_to_raw.lower() == "all routes"
    text = str(row["note"])

    # A note can nominally say "All Routes" but explicitly disclaim that the
    # affected routes aren't in this dataset -- that must never match.
    if "not part of" in text.lower():
        applies_to_all = False
        applies_to_route = None
    else:
        applies_to_route = None if applies_to_all else applies_to_raw

    note_date = pd.Timestamp(row["date"])
    window_start, window_end = _parse_window(note_date, text)

    return ParsedNote(
        note_id=row["note_id"],
        date=note_date,
        applies_to_raw=applies_to_raw,
        applies_to_all=applies_to_all,
        applies_to_route=applies_to_route,
        polarity=_parse_polarity(text),
        window_start=window_start,
        window_end=window_end,
        text=text,
    )


def load_notes(path=NOTES_CSV) -> list[ParsedNote]:
    df = pd.read_csv(path, parse_dates=["date"])
    return [_parse_row(row) for _, row in df.iterrows()]
