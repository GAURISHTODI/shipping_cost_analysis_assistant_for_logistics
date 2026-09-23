"""Optional, pluggable LLM polish pass. OFF by default.

Design intent: the pipeline's correctness (verdict, matched_note_id,
percentages) never depends on this module -- those are fixed by
src/grounding.py before any LLM is involved. This module, if enabled, only
rewrites the `reason` sentence into looser prose using the *same* grounded
note text it's handed; it cannot introduce a new note or flip a verdict.

Backend: a local Ollama server (free, open-source models, no API key, no
per-token billing) via its HTTP API. If Ollama isn't running, or the
--use-llm flag isn't passed, every row falls back to the deterministic
template in src/reasoning.py with zero LLM calls and zero cost. Swapping in
a hosted API (OpenAI/Anthropic/etc.) instead of Ollama only requires
replacing `_call_ollama` -- the token/cost accounting interface stays the
same.

We default to OFF because: (a) the deterministic template already satisfies
"grounded, not hallucinated" -- an LLM can only make the wording more
natural, not more correct; (b) every LLM call is a reproducibility risk
even at temperature 0 (model/version drift); (c) the brief explicitly
rewards "sensible LLM call volume" -- zero is the most sensible volume for
a task the template already solves deterministically.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2"
TEMPERATURE = 0.0


@dataclass
class UsageLog:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    failures: int = 0
    model: str = DEFAULT_MODEL

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "failed_calls": self.failures,
            "estimated_cost_usd": 0.0,  # local open-source model: no per-token billing
        }


@dataclass
class LLMPolisher:
    enabled: bool = False
    model: str = DEFAULT_MODEL
    usage: UsageLog = field(default_factory=UsageLog)

    def __post_init__(self):
        self.usage.model = self.model

    def polish(self, template_reason: str, note_text: str, verdict: str) -> str:
        if not self.enabled:
            return template_reason

        prompt = (
            "Rewrite the following explanation in one or two plain-English "
            "sentences. Do not invent any fact not already present. Keep the "
            f"verdict '{verdict}' unchanged.\n\n"
            f"Explanation: {template_reason}\n\nSource note: {note_text}"
        )
        try:
            text, prompt_tokens, response_tokens = self._call_ollama(prompt)
        except (urllib.error.URLError, OSError, TimeoutError):
            self.usage.failures += 1
            return template_reason

        self.usage.calls += 1
        self.usage.input_tokens += prompt_tokens
        self.usage.output_tokens += response_tokens
        return text.strip() or template_reason

    def _call_ollama(self, prompt: str) -> tuple[str, int, int]:
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": TEMPERATURE},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return (
            body.get("response", ""),
            int(body.get("prompt_eval_count", 0)),
            int(body.get("eval_count", 0)),
        )
