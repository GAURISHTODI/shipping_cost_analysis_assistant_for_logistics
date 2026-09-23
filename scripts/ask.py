#!/usr/bin/env python
"""Stretch-goal Q&A CLI: python scripts/ask.py "why did X get pricier in March?"

Run with no arguments for an interactive prompt. Requires output/output.csv
to already exist (run scripts/run.py first).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.qa_agent import main  # noqa: E402

if __name__ == "__main__":
    main()
