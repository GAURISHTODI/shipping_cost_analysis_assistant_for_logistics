# Token / cost log — one full run

- Total route-week rows evaluated: 728
- Flagged unexplained (`Yes`): 24
- Flagged & justified (`No (justified)`): 3
- Ordinary / not anomalous (`No`): 701

## LLM usage

```json
{
  "model": "qwen2.5:0.5b",
  "calls": 3,
  "input_tokens": 580,
  "output_tokens": 117,
  "failed_calls": 0,
  "rejected_low_quality": 1,
  "estimated_cost_usd": 0.0
}
```

LLM polishing was ON via a local Ollama model (open-source, no per-token
billing) — see calls/tokens above. Even so, the verdict, matched_note_id, and
percentages were already fixed by the deterministic grounding step before the
LLM ever ran, so estimated_cost_usd is $0 and the LLM cannot change the verdict.
