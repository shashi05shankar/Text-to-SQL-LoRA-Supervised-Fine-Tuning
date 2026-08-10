"""A deterministic fake generator.

Exists so the API contract and the whole eval path can be tested on CPU in milliseconds,
with no torch, no transformers, no download, and no GPU. Tests that need "a model" get
this; nothing in the serving path can then be excused as untested because a real model was
too expensive to load.

It is not a language model and does not pretend to be. It returns canned SQL, which is
exactly what a harness test needs -- the harness's job is to score whatever it is given.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence


class StubGenerator:
    """Returns canned completions, cycling through `responses`.

    Mirrors `SqlGenerator`'s full interface -- including `base_only()` and `has_adapter` --
    because the serving layer depends on both. A stub that implemented only `generate()`
    would let a bug in the adapter-switching path pass every test.

    With the adapter disabled it returns `base_responses` instead, so a test can prove the
    two paths genuinely differ rather than merely both returning something.

    Records every prompt it was given, so a test can assert the prompt actually reached
    the model rather than being dropped somewhere in the serving layer.
    """

    def __init__(
        self,
        responses: Sequence[str] | None = None,
        base_responses: Sequence[str] | None = None,
        has_adapter: bool = True,
    ) -> None:
        self.responses = list(responses) if responses else ["SELECT 1"]
        self.base_responses = list(base_responses) if base_responses else ["SELECT 0"]
        self.has_adapter = has_adapter
        self.seen_prompts: list[str] = []
        self.adapter_enabled = True

    @contextlib.contextmanager
    def base_only(self):
        previous = self.adapter_enabled
        self.adapter_enabled = False
        try:
            yield
        finally:
            self.adapter_enabled = previous

    def generate(self, prompts: Sequence[str]) -> list[str]:
        self.seen_prompts.extend(prompts)
        pool = self.responses if self.adapter_enabled else self.base_responses
        return [pool[i % len(pool)] for i in range(len(prompts))]

    # The serving layer calls one prompt at a time.
    def generate_one(self, prompt: str) -> str:
        return self.generate([prompt])[0]


class EchoGoldGenerator:
    """Returns the gold query for each example -- a system that is correct by construction.

    Used to prove the harness reports ~100% when handed perfect predictions. A harness
    that cannot score a perfect system correctly cannot be trusted to score a real one.
    """

    def __init__(self, gold_queries: Sequence[str]) -> None:
        self.gold_queries = list(gold_queries)
        self.calls = 0

    def generate(self, prompts: Sequence[str]) -> list[str]:
        if len(prompts) != len(self.gold_queries):
            raise ValueError("prompt count does not match the gold queries this stub holds")
        self.calls += 1
        return list(self.gold_queries)
