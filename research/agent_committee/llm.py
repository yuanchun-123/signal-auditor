"""Backends that turn (system prompt, user prompt) requests into JSON answers.

* :class:`ClaudeBackend` -- Claude API, synchronous (with server-side refusal
  fallbacks) or Message Batches (50% cheaper, results within ~1 hour). Every
  answer is cached in a JSONL file keyed by request ID, so reruns are free and
  an interrupted run resumes where it stopped.
* The rule-based backend lives in ``committee.py``: it needs no API key and is
  used for tests, plumbing checks and as the "twin" baseline.

Credentials: the Anthropic SDK reads ``ANTHROPIC_API_KEY`` (or an ``ant auth
login`` profile). Nothing here stores a key.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PRICES = {  # USD per million tokens (input, output), first-party API, Nov 2026
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}


class ClaudeBackend:
    def __init__(self, schema: dict, cache_path: str | Path, model: str = "claude-opus-5-5",
                 effort: str = "low", max_tokens: int = 16000, concurrency: int = 4):
        import anthropic

        self.client = anthropic.Anthropic()
        self.schema = schema
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.concurrency = concurrency
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.cache: dict[str, dict] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                rec = json.loads(line)
                self.cache[rec["id"]] = rec
        self.usage = {"input": 0, "output": 0, "cache_read": 0}
        self.usd = 0.0

    # ------------------------------------------------------------------ #
    def _params(self, system: str, user: str) -> dict:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user}],
            "output_config": {"effort": self.effort,
                              "format": {"type": "json_schema", "schema": self.schema}},
        }

    def _store(self, rid: str, answer: dict | None, stop: str, usage,
               price_mult: float = 1.0) -> None:
        rec = {"id": rid, "answer": answer, "stop_reason": stop}
        if usage is not None:
            rec["usage"] = {"input": usage.input_tokens, "output": usage.output_tokens,
                            "cache_read": getattr(usage, "cache_read_input_tokens", 0) or 0}
            pin, pout = PRICES.get(self.model, (0.0, 0.0))
            with self._lock:
                for k in self.usage:
                    self.usage[k] += rec["usage"][k]
                self.usd += price_mult * (rec["usage"]["input"] * pin
                                          + rec["usage"]["cache_read"] * pin * 0.1
                                          + rec["usage"]["output"] * pout) / 1e6
        with self._lock:
            self.cache[rid] = rec
            with self.cache_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")

    @staticmethod
    def _parse(message) -> dict | None:
        if message.stop_reason == "refusal":
            return None
        text = next((b.text for b in message.content if b.type == "text"), "")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def cost(self) -> float:
        """Approximate USD spent in this process (cache writes priced as plain input)."""
        return self.usd

    # ------------------------------------------------------------------ #
    def run(self, requests: dict[str, tuple[str, str]], mode: str = "sync") -> dict[str, dict | None]:
        todo = {k: v for k, v in requests.items() if k not in self.cache}
        if todo:
            (self._run_sync if mode == "sync" else self._run_batch)(todo)
        return {k: self.cache.get(k, {}).get("answer") for k in requests}

    def _run_sync(self, todo: dict[str, tuple[str, str]]) -> None:
        def one(item):
            rid, (system, user) = item
            # server-side fallback re-runs a refused request on another model in the same call
            msg = self.client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                **self._params(system, user))
            self._store(rid, self._parse(msg), msg.stop_reason, msg.usage)

        with ThreadPoolExecutor(self.concurrency) as ex:
            for i, _ in enumerate(ex.map(one, todo.items()), 1):
                if i % 25 == 0:
                    print(f"  {i}/{len(todo)} requests, ~${self.cost():.2f} so far", flush=True)

    def _run_batch(self, todo: dict[str, tuple[str, str]], poll: int = 60) -> None:
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        reqs = [Request(custom_id=rid, params=MessageCreateParamsNonStreaming(**self._params(s, u)))
                for rid, (s, u) in todo.items()]
        batch = self.client.messages.batches.create(requests=reqs)
        print(f"  batch {batch.id}: {len(reqs)} requests submitted", flush=True)
        while True:
            b = self.client.messages.batches.retrieve(batch.id)
            if b.processing_status == "ended":
                break
            print(f"  batch {batch.id}: {b.request_counts.processing} processing", flush=True)
            time.sleep(poll)
        for res in self.client.messages.batches.results(batch.id):
            if res.result.type == "succeeded":
                m = res.result.message
                self._store(res.custom_id, self._parse(m), m.stop_reason, m.usage, price_mult=0.5)
            else:
                print(f"  {res.custom_id}: {res.result.type}")
