#!/usr/bin/env python
"""Run the pipeline 3 independent times and diff the results.

Writes output/reproducibility_check.md with the verdict. The pipeline has
no LLM calls by default (deterministic template), so this should be a
byte-identical diff; run with --use-llm to additionally prove the optional
LLM path doesn't move the verdict/numbers (only `reason` wording may
differ when the LLM is on).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import OUTPUT_DIR  # noqa: E402
from src.pipeline import build_output  # noqa: E402

GRADED_COLUMNS = [
    "route",
    "week_of",
    "cost_per_tonne_km",
    "vs_own_history",
    "vs_similar_routes",
    "flagged",
    "matched_note_id",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    runs = []
    for i in range(args.runs):
        df, _ = build_output(use_llm=args.use_llm)
        runs.append(df.reset_index(drop=True))
        print(f"Run {i + 1}/{args.runs}: {len(df)} rows generated.")

    baseline = runs[0][GRADED_COLUMNS]
    all_identical = True
    report_lines = [f"# Reproducibility check ({args.runs} runs)\n"]
    report_lines.append(f"LLM polishing: {'ON' if args.use_llm else 'OFF (deterministic template)'}\n")

    for i, run in enumerate(runs[1:], start=2):
        comparison = run[GRADED_COLUMNS]
        diff_mask = ~(comparison.eq(baseline) | (comparison.isna() & baseline.isna()))
        n_diff = diff_mask.any(axis=1).sum()
        if n_diff == 0:
            report_lines.append(f"- Run 1 vs Run {i}: IDENTICAL on all graded columns ({len(baseline)} rows).")
        else:
            all_identical = False
            report_lines.append(f"- Run 1 vs Run {i}: {n_diff} row(s) DIFFER.")
            diffs = comparison[diff_mask.any(axis=1)]
            report_lines.append("```\n" + diffs.to_string() + "\n```")

    if not args.use_llm:
        reason_identical = all(runs[0]["reason"].equals(r["reason"]) for r in runs[1:])
        report_lines.append(
            f"\nReason text (template, no LLM): identical across all runs = {reason_identical}."
        )

    verdict = "PASS" if all_identical else "FAIL"
    report_lines.insert(1, f"**Verdict: {verdict}**\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "reproducibility_check.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\nVerdict: {verdict}")
    print(f"Report written to {report_path}")
    return 0 if all_identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
