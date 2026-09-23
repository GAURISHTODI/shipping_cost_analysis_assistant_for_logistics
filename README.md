# FreightTiger Cost Watch — a grounded shipping-cost anomaly assistant

Built for the FreightTiger SDE Intern (AI) 24-hour case study: *"Build a smart assistant that watches shipping costs."*

It groups shipment records by route and week, computes cost-per-tonne-km, compares each week against two exactly-specified baselines, flags anomalous rises, and — before flagging anything as unexplained — checks a small notes corpus via retrieval (RAG) to see if there's a real, provable reason. Every "justified" verdict is independently re-verified by a deterministic rule engine, so the assistant cannot fabricate a justification even if the retriever surfaces an irrelevant note.

## 1. Quickstart

```bash
pip install -r requirements.txt

python scripts/run.py                    # writes output/output.csv + output/token_cost_log.md
python eval/eval_harness.py              # scores the pipeline against known-correct examples
python scripts/check_reproducibility.py  # runs the pipeline 3x, diffs the results
python -m pytest tests -q                # 18 unit tests

python scripts/ask.py "why did Chennai-Bangalore get pricier in March?"   # stretch-goal Q&A
python scripts/ask.py                    # interactive mode
```

No API key, no internet connection, and no paid service is required for any of the above — see [§6 Cost of running it](#6-cost-of-running-it).

## 2. What it does, precisely

**Grouping & weekly metric.** Shipments are grouped by `route` (`origin-destination`) and the given `route_type`. Each `shipment_date` is bucketed into its Monday-anchored week (`week_of`). For each `(route, week)`:

```
cost_per_tonne_km = sum(freight_cost) / sum(quantity_tonnes × distance_km)
```

computed over the *week's total*, not averaged per-shipment — this is a tonne-km-weighted average so a handful of small shipments can't skew the week's figure. ([src/weekly_metrics.py](src/weekly_metrics.py))

**Baseline 1 — vs. own history.** Trailing 8-week rolling average `cost_per_tonne_km` for the *same route*, using only weeks strictly before the current one (`shift(1)` before the rolling window — no look-ahead). Routes with fewer than 8 prior weeks use whatever prior weeks exist; `own_history_weeks_used` records exactly how many. A route's very first observed week has zero prior weeks, so its baseline is `N/A (no prior weeks)`, never padded or guessed. ([src/baselines.py](src/baselines.py))

**Baseline 2 — vs. similar routes.** Same-week average `cost_per_tonne_km` across every *other* route sharing the same `route_type`, with the route itself excluded from its own peer average (computed via a group-sum-minus-self, not a second self-join).

**Flagging.** A route-week is a rising/anomalous *candidate* if it is more than **+15%** above either baseline (`ANOMALY_THRESHOLD_PCT` in [src/config.py](src/config.py)). This single documented threshold was checked against all three worked examples in `sample_output_format_v2.csv` — including the Mumbai-Pune row, which was flagged almost entirely on the peer comparison (+23.6%) despite a modest own-history rise (+9.2%), confirming the flag condition is an **OR**, not an **AND**, across the two baselines.

## 3. Note grounding — the anti-hallucination core

This is the part the brief weights most heavily ("can we trust it?"), so it's the part I spent the most design effort on, and it is **not an LLM decision** — it's plain rule-based code, independently testable.

Each of the 10 notes in `context_notes.csv` is parsed into three structured facts ([src/notes_parser.py](src/notes_parser.py)):

1. **Route match** — `applies_to == route` exactly, or `"All Routes"`.
2. **Time window** — if the note text contains an explicit range (e.g. *"Feb 24 to Mar 8"*), that exact range is used. Otherwise, the note's effect is assumed to last only the **single ISO week containing the note's own `date`** — a vague phrase like *"starting this week"* does **not** extend the window indefinitely.
3. **Polarity** — does the note actually describe a cost-*increasing* event? A note can mention the right route and the right date and still be `negative` if it contains phrases like *"not significantly affected"*, *"remained normal"*, *"returned to normal"*, *"improved road conditions"*, or *"without a rate change"*. Ambiguous notes (no clear positive language) default to `negative` — per the brief: *"no supporting note → unexplained; do not guess."*

A verdict is only `justified` if **all three** checks pass ([src/grounding.py](src/grounding.py)). This module is called independently of retrieval and is covered by its own unit tests ([tests/test_grounding.py](tests/test_grounding.py)) and a 12-case labeled eval set ([eval/labeled_examples.csv](eval/labeled_examples.csv)) that specifically targets every trap in the 10 notes — e.g. N005/N007/N008/N010 all mention the *exact* right route and date but are negative-polarity and must never justify; N004 says "All Routes" but its text says the affected routes aren't in this dataset; N003 (diesel, no explicit end date) must justify the exact week it was issued but must **not** reach forward to justify a rise two weeks later.

### Why this design (the Mumbai-Pune trap)

The worked example for Mumbai-Pune (week of 2025-09-15) is flagged *unexplained*, even though a diesel-price note (N003, all routes, dated 2025-05-05) exists four months earlier with no stated end date. My first instinct was to treat "starting this week" as open-ended (diesel prices don't just revert). That would have wrongly justified this row. Reproducing the sample forced the stricter rule above: **default to the narrowest defensible window, and let a rise go unexplained rather than guess forward.** This single design choice is what makes N001–N010 grade correctly across all three worked examples and all twelve labeled traps.

**Retrieval's actual role.** A local TF-IDF vector store ([src/retrieval.py](src/retrieval.py)) over the 10 notes ranks candidates when more than one verified note could apply (rare here), and powers the stretch-goal Q&A agent. TF-IDF, not a neural embedding model, was chosen deliberately: the corpus is 10 short notes, a sentence-transformer buys semantic recall at the cost of a ~90MB download and a source of run-to-run drift — a bad trade for a reproducibility-graded, 24-hour, possibly-offline submission. The interface is small enough (`.query(text, k)`) that swapping in `sentence-transformers` + FAISS/Chroma is a one-function change if a much larger note corpus ever needed real semantic recall — I said as much in the code comment rather than building it speculatively.

Retrieval **finds** candidates; grounding **proves** them. An LLM is never asked "is this justified?" — that question is answered by code that can be unit-tested and is either right or wrong, not persuasive or not.

## 4. Explanation text

`reason` is generated by a deterministic template ([src/reasoning.py](src/reasoning.py)) that quotes the matched note's own text and dates — zero LLM calls by default. An optional LLM polish pass exists ([src/llm.py](src/llm.py), `--use-llm`, via a local Ollama model) to loosen the phrasing, but it is structurally prevented from ever changing the verdict, `matched_note_id`, or the percentages — those are computed and fixed *before* the LLM is invoked, so at worst a bad LLM call makes the prose clunkier, never wrong. See §6 for why this is off by default.

## 5. Reproducibility

`scripts/check_reproducibility.py` runs the full pipeline 3 times and diffs every graded column. Result (`output/reproducibility_check.md`, regenerate anytime):

```
Verdict: PASS
Run 1 vs Run 2: IDENTICAL on all graded columns (728 rows).
Run 1 vs Run 3: IDENTICAL on all graded columns (728 rows).
Reason text (template, no LLM): identical across all runs = True.
```

This is expected, not lucky: there is no randomness anywhere in the default path (no LLM sampling, no random seeds, no dict-ordering-dependent logic). If `--use-llm` is passed, `temperature=0` is set on the Ollama call and the same reproducibility script can be re-run with that flag to confirm wording may vary while the verdict/numbers still don't — that's the guardrail the brief actually asks for.

## 6. Cost of running it

Default run (`python scripts/run.py`, no flags): **0 LLM calls, 0 tokens, $0.00.** Every `reason` string comes from the deterministic template in §4; retrieval is local TF-IDF (CPU, milliseconds, no external call). This is logged automatically to `output/token_cost_log.md` after every run.

If `--use-llm` is passed, only the ~27 candidate rows (out of 728 route-weeks — see §7) that reach the reasoning step incur an LLM call, using a free, open-source, locally-hosted Ollama model (no per-token billing). The log still reports `estimated_cost_usd: 0.0` because there is no metered API in play; call/token counts are logged anyway for honesty (`prompt_eval_count` / `eval_count` from Ollama's response).

I chose to make the LLM optional and default-off rather than "on with a cheap free-tier model" because: the deterministic template already satisfies the grading bar ("grounded, not hallucinated") without it; every LLM call is a reproducibility risk even at temperature 0 (model/version drift between runs); and the brief explicitly rewards "sensible LLM call volume" — zero calls is the most sensible volume for a sub-problem (734 short template strings) that doesn't need one.

## 7. Knowing the system is right

Two independent checks, run before every submission (`python eval/eval_harness.py`):

1. **Official worked examples** — re-derives the exact 3 rows given in `sample_output_format_v2.csv` and asserts `cost_per_tonne_km`, `vs_own_history`, `vs_similar_routes`, `flagged`, and `matched_note_id` match **exactly** (reason wording is allowed to differ per the brief). All 3 pass.
2. **Grounding unit checks** — 12 hand-labeled `(route, week_of) → expected matched_note_id` cases (`eval/labeled_examples.csv`) built to specifically exercise every note's polarity/window/route trap, called directly against `src/grounding.ground()` so the result doesn't depend on whether that route-week happens to be a rising candidate that week. All 12 pass.

Plus 18 `pytest` unit tests covering the rolling-window no-look-ahead property, the peer-average self-exclusion, note parsing per note, and grounding edge cases (`tests/`).

Of the 728 route-week rows in the current output: 701 are ordinary (`No`), 24 are flagged unexplained (`Yes`), 3 are flagged-and-justified (`No (justified)`) — i.e. the vast majority of the 10-note corpus's positive-polarity notes (N001, N002, N003) were each actually put to use, and no negative-polarity note was ever allowed to justify a row, which I spot-checked directly against the full output.

## 8. Stretch goal — Q&A over the findings

```bash
python scripts/ask.py "why did Delhi-Jaipur get pricier?"
python scripts/ask.py "any anomalies on Mumbai-Pune in October?"
```

The agent (`src/qa_agent.py`) extracts a route name and/or month from the question via simple pattern matching, then answers directly from `output/output.csv` — i.e. every answer is a grounded row that already passed the same verification in §3, not a fresh LLM guess. If no route/date can be identified, it falls back to the notes vector store and surfaces the most relevant note(s) instead of hallucinating an answer.

## 9. Project layout

```
src/
  weekly_metrics.py   shipment loading + weekly cost_per_tonne_km aggregation
  baselines.py        own-history rolling avg + peer route_type avg
  flagging.py         rising/anomalous candidate threshold
  notes_parser.py     notes -> structured (route, window, polarity) facts
  retrieval.py         TF-IDF vector store (RAG) over notes
  grounding.py         deterministic justification verifier (anti-hallucination)
  reasoning.py         template-based `reason` text
  llm.py               optional, off-by-default LLM polish + usage accounting
  qa_agent.py           stretch-goal Q&A over the generated output
  pipeline.py           orchestrates the above into the output rows
eval/
  labeled_examples.csv  12 hand-labeled grounding traps
  eval_harness.py        scores the pipeline against §7's two checks
tests/                   18 pytest unit tests
scripts/
  run.py                 main entrypoint -> output/output.csv
  check_reproducibility.py
  ask.py                 stretch-goal Q&A CLI
data/                    the three provided input files
output/                  output.csv, token_cost_log.md, reproducibility_check.md
```

## 10. Known limitations / explicit trade-offs

- **Threshold is a fixed constant (15%)**, not a statistical control-chart bound. Documented and justified against the worked examples in §2, but a production system would likely use a per-route/per-route_type variance-aware bound instead of one global percentage.
- **Note-window inference is rule-based regex over 10 short notes.** It generalizes (it isn't hardcoded per note_id), but it was validated by hand against exactly these 10 notes; a much larger, messier notes corpus would need a more robust date-range parser (e.g. `dateutil` fuzzy parsing) or an LLM-based extraction step *with the same downstream deterministic re-verification* this design already has.
- **Multiple verified candidate notes for one row** is handled by preferring an exact-route note over "All Routes", then by TF-IDF relevance — this path exists but is untested against real data because it never actually triggers in this dataset (documented, not hidden).
