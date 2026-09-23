#!/usr/bin/env python
"""Lightweight eval harness -- run before trusting a submission.

Two independent checks:

1. Official worked examples: the 3 rows given in
   data/sample_output_format_v2.csv. We re-run the pipeline and assert the
   graded numeric fields (cost_per_tonne_km, vs_own_history,
   vs_similar_routes, flagged, matched_note_id) match EXACTLY. Reason
   wording is allowed to differ (the brief says so explicitly).

2. Grounding unit checks (eval/labeled_examples.csv): a hand-built set of
   (route, week_of) -> expected matched_note_id (or blank = "must be
   unexplained"), covering every note's polarity/window/route-match trap
   (see rationale column). This calls src/grounding.ground() directly, so
   it tests the anti-hallucination logic in isolation from whether that
   route-week happens to be a "rising" candidate in the real data.

Exit code is 0 iff everything passes; non-zero otherwise, so this can be
wired into CI.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.grounding import ground  # noqa: E402
from src.notes_parser import load_notes  # noqa: E402
from src.pipeline import build_output  # noqa: E402
from src.retrieval import VectorStore  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
LABELED_CSV = EVAL_DIR / "labeled_examples.csv"

OFFICIAL_EXAMPLES = [
    {
        "route": "Delhi-Jaipur",
        "week_of": "2024-11-11",
        "cost_per_tonne_km": 4.17,
        "vs_own_history": "+35.5% vs this route's past average",
        "vs_similar_routes": "+21.0% vs similar-length routes this week",
        "flagged": "Yes",
        "matched_note_id": "",
    },
    {
        "route": "Ahmedabad-Mumbai",
        "week_of": "2025-01-20",
        "cost_per_tonne_km": 3.29,
        "vs_own_history": "+29.5% vs this route's past average",
        "vs_similar_routes": "+22.5% vs similar-length routes this week",
        "flagged": "No (justified)",
        "matched_note_id": "N002",
    },
    {
        "route": "Mumbai-Pune",
        "week_of": "2025-09-15",
        "cost_per_tonne_km": 3.98,
        "vs_own_history": "+9.2% vs this route's past average",
        "vs_similar_routes": "+23.6% vs similar-length routes this week",
        "flagged": "Yes",
        "matched_note_id": "",
    },
]


def check_official_examples(output_df: pd.DataFrame) -> list[str]:
    failures = []
    graded_cols = [
        "cost_per_tonne_km",
        "vs_own_history",
        "vs_similar_routes",
        "flagged",
        "matched_note_id",
    ]
    for expected in OFFICIAL_EXAMPLES:
        row = output_df[
            (output_df["route"] == expected["route"])
            & (output_df["week_of"] == expected["week_of"])
        ]
        label = f"{expected['route']} / {expected['week_of']}"
        if row.empty:
            failures.append(f"[MISSING] {label}: not found in output")
            continue
        actual = row.iloc[0]
        for col in graded_cols:
            exp_val = expected[col]
            act_val = actual[col]
            act_val = "" if pd.isna(act_val) else act_val
            if str(act_val) != str(exp_val):
                failures.append(
                    f"[MISMATCH] {label} .{col}: expected {exp_val!r}, got {act_val!r}"
                )
    return failures


def check_grounding_examples() -> tuple[list[str], int]:
    failures = []
    notes = load_notes()
    store = VectorStore(notes)
    labeled = pd.read_csv(LABELED_CSV, dtype=str, keep_default_na=False, parse_dates=["week_of"])

    for _, row in labeled.iterrows():
        route = row["route"]
        week_of = pd.Timestamp(row["week_of"])
        expected = row["expected_matched_note_id"] or None

        match = ground(store, notes, route, week_of)
        actual = match.note.note_id if match else None

        label = f"{route} / {week_of.date()}"
        if actual != expected:
            failures.append(
                f"[GROUNDING MISMATCH] {label}: expected {expected!r}, got {actual!r} "
                f"-- {row['rationale']}"
            )
    return failures, len(labeled)


def main() -> int:
    print("Running pipeline for evaluation...")
    output_df, _ = build_output(use_llm=False)

    print("\n=== 1. Official worked examples (sample_output_format_v2.csv) ===")
    official_failures = check_official_examples(output_df)
    if official_failures:
        for f in official_failures:
            print(" FAIL:", f)
    else:
        print(f" PASS: all {len(OFFICIAL_EXAMPLES)} official examples match exactly.")

    print("\n=== 2. Grounding / anti-hallucination unit checks ===")
    grounding_failures, n = check_grounding_examples()
    if grounding_failures:
        for f in grounding_failures:
            print(" FAIL:", f)
    else:
        print(f" PASS: all {n} labeled grounding cases match exactly.")

    total_failures = len(official_failures) + len(grounding_failures)
    print(f"\n=== Summary: {total_failures} failure(s) ===")
    return 1 if total_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
