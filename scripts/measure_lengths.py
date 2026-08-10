"""Measure the real prompt-length distribution, and set `max_seq_len` from it.

`max_seq_len` is a genuine tradeoff: too small and schemas get truncated (the model is
asked to write SQL against tables it cannot see), too large and VRAM and training time
both climb for no benefit. Spider's databases vary enormously -- a few have dozens of
tables -- so the right value is an empirical question about this corpus, not a number to
copy from a blog post.

This script tokenizes every training example with the real base-model tokenizer and
reports the distribution, plus what fraction would truncate at each candidate window.
Its output is what justifies the `DataConfig.max_seq_len` value in the README.

    python scripts/measure_lengths.py
    python scripts/measure_lengths.py --few-shot 3    # the baseline's longer prompts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql_lora.config import DataConfig, Paths
from text2sql_lora.data.prompts import (
    build_inference_prompt,
    select_few_shot,
)
from text2sql_lora.data.spider import load_schemas, load_split

CANDIDATE_WINDOWS = (512, 768, 1024, 1536, 2048)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    parser.add_argument("--few-shot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 = all examples")
    parser.add_argument("--out", type=Path, default=None, help="optional JSON output path")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    config = DataConfig()
    paths = Paths()

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    schemas = load_schemas(paths)
    data = load_split(args.split, paths=paths, schemas=schemas)

    few_shot = []
    if args.few_shot:
        pool = load_split("train", paths=paths, schemas=schemas).examples
        few_shot = select_few_shot(pool, n=args.few_shot, seed=config.seed)

    examples = data.examples[: args.limit] if args.limit else data.examples
    print(f"tokenizing {len(examples)} {args.split} examples with {config.base_model}")

    prompt_lengths: list[int] = []
    target_lengths: list[int] = []
    for index, example in enumerate(examples):
        prompt = build_inference_prompt(tokenizer, example, few_shot=few_shot or None)
        prompt_lengths.append(len(tokenizer(prompt, add_special_tokens=False)["input_ids"]))
        target = example.query.strip() + tokenizer.eos_token
        target_lengths.append(len(tokenizer(target, add_special_tokens=False)["input_ids"]))
        if index and index % 1000 == 0:
            print(f"  {index}/{len(examples)}")

    prompts = np.array(prompt_lengths)
    targets = np.array(target_lengths)
    totals = prompts + targets

    def describe(name: str, values: np.ndarray) -> dict:
        stats = {
            "mean": float(values.mean()),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "p99": float(np.percentile(values, 99)),
            "max": int(values.max()),
        }
        print(
            f"{name:>8}: mean {stats['mean']:7.1f}  p50 {stats['p50']:6.0f}  "
            f"p90 {stats['p90']:6.0f}  p99 {stats['p99']:6.0f}  max {stats['max']:6d}"
        )
        return stats

    print()
    report = {
        "split": args.split,
        "n_examples": len(examples),
        "few_shot": args.few_shot,
        "base_model": config.base_model,
        "prompt_tokens": describe("prompt", prompts),
        "target_tokens": describe("target", targets),
        "total_tokens": describe("total", totals),
        "truncation_rate_by_window": {},
    }

    print("\ntruncation rate by candidate window (fraction of examples that would truncate):")
    for window in CANDIDATE_WINDOWS:
        rate = float((totals > window).mean())
        report["truncation_rate_by_window"][str(window)] = rate
        verdict = "OK" if rate <= config.max_truncation_rate else "over budget"
        marker = " <- configured" if window == config.max_seq_len else ""
        print(f"  {window:5d}: {rate:7.2%}  {verdict}{marker}")

    chosen_rate = report["truncation_rate_by_window"][str(config.max_seq_len)]
    print(
        f"\nconfigured max_seq_len={config.max_seq_len} truncates {chosen_rate:.2%} "
        f"(budget {config.max_truncation_rate:.0%})"
    )
    if chosen_rate > config.max_truncation_rate:
        acceptable = [w for w in CANDIDATE_WINDOWS if report["truncation_rate_by_window"][str(w)]
                      <= config.max_truncation_rate]
        suggestion = acceptable[0] if acceptable else max(CANDIDATE_WINDOWS)
        print(f"ACTION: raise DataConfig.max_seq_len to {suggestion}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
