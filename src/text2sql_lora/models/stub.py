"""A deterministic fake generator.

Exists so the API contract and the whole eval path can be tested on CPU in milliseconds,
with no torch, no transformers, no download, and no GPU. Tests that need "a model" get
this; nothing in the serving path can then be excused as untested because a real model was
too expensive to load.

It is not a language model and does not pretend to be. It returns canned SQL, which is
exactly what a harness test needs -- the harness's job is to score whatever it is given.
"""

from __future__ import annotations

from collections.abc import Sequence


class StubGenerator:
    """Returns canned completions, cycling through `responses`.

    Records every prompt it was given, so a test can assert the prompt actually reached
    the model rather than being dropped somewhere in the serving layer.
    """

    def __init__(self, responses: Sequence[str] | None = None) -> None:
        self.responses = list(responses) if responses else ["SELECT 1"]
        self.seen_prompts: list[str] = []

    def generate(self, prompts: Sequence[str]) -> list[str]:
        self.seen_prompts.extend(prompts)
        return [self.responses[i % len(self.responses)] for i in range(len(prompts))]

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
