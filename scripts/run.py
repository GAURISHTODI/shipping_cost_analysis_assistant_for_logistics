#!/usr/bin/env python
"""Main entrypoint: python scripts/run.py [--use-llm] [--model llama3.2] [--out path.csv]

Runs the full pipeline over data/shipment_records.csv and writes the
output CSV matching data/sample_output_format_v2.csv's column contract.
Also writes output/token_cost_log.md for the run (zero cost unless
--use-llm is passed and an Ollama server is reachable).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import OUTPUT_CSV, OUTPUT_DIR  # noqa: E402
from src.pipeline import build_output  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--use-llm", action="store_true", help="Enable optional LLM reason-polishing via a local Ollama server (off by default).")
    parser.add_argument("--model", default="qwen2.5:0.5b", help="Ollama model name to use if --use-llm is set.")
    parser.add_argument("--out", default=str(OUTPUT_CSV), help="Output CSV path.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df, polisher = build_output(use_llm=args.use_llm, llm_model=args.model)
    df.to_csv(out_path, index=False)

    n_total = len(df)
    n_flagged = (df["flagged"] == "Yes").sum()
    n_justified = (df["flagged"] == "No (justified)").sum()
    n_ordinary = (df["flagged"] == "No").sum()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_filename = "token_cost_log_llm.md" if args.use_llm else "token_cost_log.md"
    log_path = OUTPUT_DIR / log_filename
    log_path.write_text(
        "# Token / cost log — one full run\n\n"
        f"- Total route-week rows evaluated: {n_total}\n"
        f"- Flagged unexplained (`Yes`): {n_flagged}\n"
        f"- Flagged & justified (`No (justified)`): {n_justified}\n"
        f"- Ordinary / not anomalous (`No`): {n_ordinary}\n\n"
        "## LLM usage\n\n"
        f"```json\n{json.dumps(polisher.usage.as_dict(), indent=2)}\n```\n\n"
        + (
            "LLM polishing was OFF for this run: reason text was generated entirely by\n"
            "the deterministic template in src/reasoning.py. 0 LLM calls, 0 tokens, $0 cost.\n"
            if not args.use_llm
            else "LLM polishing was ON via a local Ollama model (open-source, no per-token\n"
            "billing) — see calls/tokens above. Even so, the verdict, matched_note_id, and\n"
            "percentages were already fixed by the deterministic grounding step before the\n"
            "LLM ever ran, so estimated_cost_usd is $0 and the LLM cannot change the verdict.\n"
        ),
        encoding="utf-8",
    )

    if not args.quiet:
        print(f"Wrote {n_total} rows to {out_path}")
        print(f"  flagged Yes: {n_flagged} | justified: {n_justified} | ordinary: {n_ordinary}")
        print(f"Token/cost log: {log_path}")


if __name__ == "__main__":
    main()
