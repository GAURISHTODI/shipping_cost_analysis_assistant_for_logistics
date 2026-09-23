"""Stretch goal: answer plain-English questions about the findings.

Deliberately retrieval-based, not free-form-LLM-based: a question is
parsed for a route name (matched against known routes) and/or a month/year
mention, then answered directly from output/output.csv -- the same
grounded rows the CSV deliverable contains. This means every answer is
already provably grounded (it's quoting a row that itself passed
src/grounding.py), and the agent is usable with zero LLM calls.

If a route can't be identified from the question, falls back to the
TF-IDF VectorStore over the *notes* so at least a relevant note surfaces.
"""
from __future__ import annotations

import re

import pandas as pd

from src.config import OUTPUT_CSV
from src.notes_parser import load_notes
from src.retrieval import VectorStore

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}


class QAAgent:
    def __init__(self, output_csv=OUTPUT_CSV):
        self.output = pd.read_csv(
            output_csv, parse_dates=["week_of"], keep_default_na=False,
            na_values=[], dtype={"matched_note_id": str},
        )
        self.routes = sorted(self.output["route"].unique(), key=len, reverse=True)
        self.notes = load_notes()
        self.note_store = VectorStore(self.notes)

    def _find_route(self, question: str) -> str | None:
        q = question.lower()
        for route in self.routes:
            origin, _, dest = route.partition("-")
            if origin.lower() in q and dest.lower() in q:
                return route
        return None

    def _find_month_year(self, question: str) -> tuple[int, int] | None:
        q = question.lower()
        month = next((m for name, m in _MONTHS.items() if re.search(rf"\b{name}\b", q)), None)
        if month is None:
            return None
        year_match = re.search(r"\b(20\d{2})\b", q)
        year = int(year_match.group(1)) if year_match else None
        return month, year

    def ask(self, question: str) -> str:
        route = self._find_route(question)
        month_year = self._find_month_year(question)

        rows = self.output
        if route:
            rows = rows[rows["route"] == route]
        if month_year:
            month, year = month_year
            rows = rows[rows["week_of"].dt.month == month]
            if year:
                rows = rows[rows["week_of"].dt.year == year]

        if route is None and month_year is None:
            return self._fallback_to_notes(question)

        flagged_rows = rows[rows["flagged"] != "No"]
        if flagged_rows.empty:
            scope = f"on {route}" if route else ""
            when = f" in {month_year}" if month_year else ""
            return f"I found no cost anomalies {scope}{when}. Costs were within the expected range against both history and peer routes."

        lines = []
        for _, r in flagged_rows.iterrows():
            lines.append(
                f"- {r['route']}, week of {r['week_of'].date()}: cost {r['cost_per_tonne_km']} "
                f"INR/tonne-km ({r['vs_own_history']}; {r['vs_similar_routes']}). "
                f"Verdict: {r['flagged']}"
                + (f" [note {r['matched_note_id']}]" if r["matched_note_id"] else "")
                + f". {r['reason']}"
            )
        return "\n".join(lines)

    def _fallback_to_notes(self, question: str) -> str:
        results = self.note_store.query(question, top_k=2)
        if not results:
            return "I couldn't identify a route or time period in that question, and found no relevant note."
        lines = ["I couldn't identify a specific route/date in your question. Closest notes:"]
        for sn in results:
            lines.append(f"- {sn.note.note_id} ({sn.note.date.date()}, {sn.note.applies_to_raw}): {sn.note.text}")
        return "\n".join(lines)


def main() -> None:
    import sys

    agent = QAAgent()
    if len(sys.argv) > 1:
        print(agent.ask(" ".join(sys.argv[1:])))
        return

    print("FreightTiger cost-anomaly Q&A. Type a question, or 'quit' to exit.")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"quit", "exit"}:
            break
        if q:
            print(agent.ask(q))


if __name__ == "__main__":
    main()
