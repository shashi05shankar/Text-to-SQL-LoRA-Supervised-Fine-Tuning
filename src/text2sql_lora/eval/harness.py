"""Score a system's predictions against Spider, and compare systems.

The harness is deliberately model-agnostic: it takes a `generate(prompts) -> list[str]`
callable. The real model supplies one on the GPU; a stub supplies one in CPU tests; the
gold queries themselves supply one in the self-consistency check. That is what lets the
*same* scoring code produce the numbers in the README, run in the test suite, and run
inside the Kaggle notebook -- there is exactly one implementation of the metric.

Gold execution results are cached per example, because the three systems being compared
all score against the same 1,034 gold queries and executing them three times would triple
the eval cost for no information.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from text2sql_lora.config import EvalConfig, Paths
from text2sql_lora.data.spider import SpiderData
from text2sql_lora.eval.execute import ExecutionResult, execute_query
from text2sql_lora.eval.metrics import (
    ExampleScore,
    SystemScores,
    exact_match,
    extract_sql,
    has_order_by,
    results_match,
    sql_hardness,
)

GenerateFn = Callable[[Sequence[str]], list[str]]


@dataclass
class GoldCache:
    """Executed gold results, keyed by example index."""

    results: dict[int, ExecutionResult]

    @classmethod
    def build(
        cls,
        data: SpiderData,
        paths: Paths | None = None,
        config: EvalConfig | None = None,
        progress_every: int = 200,
    ) -> GoldCache:
        paths = paths or Paths()
        config = config or EvalConfig()
        results: dict[int, ExecutionResult] = {}
        for index, example in enumerate(data.examples):
            results[index] = execute_query(
                paths.db_path(example.db_id),
                example.query,
                timeout_seconds=config.timeout_seconds,
            )
            if progress_every and index and index % progress_every == 0:
                print(f"  gold {index}/{len(data.examples)}")
        return cls(results=results)

    def failures(self) -> list[int]:
        """Indices whose GOLD query does not execute.

        Any such example is unscoreable -- every system would be marked wrong on it
        regardless of what it predicted. A non-empty result here means the download is
        broken, not that the models are.
        """
        return [index for index, result in self.results.items() if not result.ok]


def score_predictions(
    name: str,
    data: SpiderData,
    predictions: Sequence[str],
    paths: Paths | None = None,
    config: EvalConfig | None = None,
    gold_cache: GoldCache | None = None,
    progress_every: int = 200,
) -> SystemScores:
    """Score raw model completions for one system."""
    if len(predictions) != len(data.examples):
        raise ValueError(
            f"got {len(predictions)} predictions for {len(data.examples)} examples"
        )

    paths = paths or Paths()
    config = config or EvalConfig()
    gold_cache = gold_cache or GoldCache.build(data, paths, config, progress_every)

    scores: list[ExampleScore] = []
    for index, (example, raw) in enumerate(zip(data.examples, predictions, strict=True)):
        predicted_sql = extract_sql(raw)
        db_path = paths.db_path(example.db_id)

        gold_result = gold_cache.results[index]
        pred_result = execute_query(
            db_path, predicted_sql, timeout_seconds=config.timeout_seconds
        )

        correct = results_match(
            gold_result.rows,
            pred_result.rows,
            order_matters=has_order_by(example.query),
            float_decimals=config.float_decimals,
        )

        scores.append(
            ExampleScore(
                index=index,
                db_id=example.db_id,
                question=example.question,
                gold=example.query,
                prediction=predicted_sql,
                execution_correct=bool(correct),
                exact_match=exact_match(example.query, predicted_sql),
                invalid_sql=not pred_result.ok,
                error=pred_result.error,
                hardness=sql_hardness(example.query),
                elapsed_seconds=pred_result.elapsed_seconds,
            )
        )

        if progress_every and index and index % progress_every == 0:
            running = sum(s.execution_correct for s in scores) / len(scores)
            print(f"  {name} {index}/{len(data.examples)}  running EX {running:.3f}")

    return SystemScores(name=name, scores=scores)


def evaluate_system(
    name: str,
    data: SpiderData,
    prompts: Sequence[str],
    generate: GenerateFn,
    paths: Paths | None = None,
    config: EvalConfig | None = None,
    gold_cache: GoldCache | None = None,
) -> SystemScores:
    """Generate then score, in one call."""
    if len(prompts) != len(data.examples):
        raise ValueError(f"got {len(prompts)} prompts for {len(data.examples)} examples")
    predictions = generate(prompts)
    return score_predictions(
        name, data, predictions, paths=paths, config=config, gold_cache=gold_cache
    )


def gold_self_consistency(
    data: SpiderData,
    paths: Paths | None = None,
    config: EvalConfig | None = None,
    gold_cache: GoldCache | None = None,
) -> SystemScores:
    """Score the gold queries AS IF they were predictions.

    This must come out at ~100% execution accuracy. It is the harness's self-test: if
    scoring the right answer against itself does not say "correct", then the comparison
    rules, the database paths, or the execution layer are broken -- and every model number
    produced afterwards would be meaningless.

    It is run BEFORE any training, deliberately. Discovering a broken harness after
    spending a GPU session is the expensive way to find out.

    It will not always be exactly 100%: a handful of Spider gold queries are genuinely
    non-deterministic (an unordered LIMIT over ties can return different rows on different
    executions). Those are reported rather than hidden.
    """
    return score_predictions(
        "gold_self_consistency",
        data,
        [example.query for example in data.examples],
        paths=paths,
        config=config,
        gold_cache=gold_cache,
    )


def compare(systems: Sequence[SystemScores]) -> dict:
    """Assemble the base-vs-tuned comparison payload."""
    if not systems:
        raise ValueError("nothing to compare")

    summaries = {system.name: system.summary() for system in systems}
    by_hardness = [row for system in systems for row in system.by_hardness()]

    payload: dict = {"results": summaries, "by_hardness": by_hardness}

    # The headline delta, computed here rather than typed into the README by hand.
    if "base_zero_shot" in summaries and "lora_zero_shot" in summaries:
        base = summaries["base_zero_shot"]["execution_accuracy"]
        tuned = summaries["lora_zero_shot"]["execution_accuracy"]
        payload["headline"] = {
            "base_zero_shot_execution_accuracy": base,
            "lora_zero_shot_execution_accuracy": tuned,
            "absolute_gain": tuned - base,
            "relative_gain": (tuned - base) / base if base > 0 else None,
        }
        if "base_few_shot" in summaries:
            few = summaries["base_few_shot"]["execution_accuracy"]
            payload["headline"]["base_few_shot_execution_accuracy"] = few
            payload["headline"]["gain_over_few_shot"] = tuned - few

    return payload
