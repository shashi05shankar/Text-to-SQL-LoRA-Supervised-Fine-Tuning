"""Harness self-test: score Spider's gold queries against themselves.

This must come out at ~100% execution accuracy. If it does not, the metric, the database
paths, or the execution layer is broken -- and every base-vs-tuned number produced
afterwards would be meaningless.

It runs BEFORE any training, on purpose. Finding out that the harness was broken *after*
burning a GPU session is the expensive way to learn it.

A perfect 100% is not guaranteed and a small shortfall is not necessarily a bug: a few
Spider gold queries are genuinely non-deterministic (an unordered `LIMIT` over tied values
can return different rows on different executions). Anything that fails is printed in full
so it can be judged rather than waved away.

    python scripts/check_harness.py
    python scripts/check_harness.py --limit 100     # quick pass
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql_lora.config import EvalConfig, Paths
from text2sql_lora.data.spider import load_split, missing_databases
from text2sql_lora.eval.harness import GoldCache, gold_self_consistency

# Below this, something is structurally wrong rather than merely noisy.
PASS_THRESHOLD = 0.99


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--limit", type=int, default=0, help="0 = all examples")
    parser.add_argument("--out", type=Path, default=Path("artifacts/harness_check.json"))
    args = parser.parse_args()

    paths = Paths()
    config = EvalConfig()

    data = load_split(args.split, paths=paths)
    if args.limit:
        data = data.subset(list(range(min(args.limit, len(data)))))

    print(f"harness self-test on {len(data)} {args.split} examples")
    print(f"databases: {len(set(data.db_ids()))}, timeout {config.timeout_seconds}s\n")

    missing = missing_databases(data, paths=paths)
    if missing:
        print(f"FAIL: {len(missing)} databases have no .sqlite file: {missing}")
        return 1

    print("executing gold queries...")
    gold_cache = GoldCache.build(data, paths=paths, config=config, progress_every=200)

    unexecutable = gold_cache.failures()
    if unexecutable:
        print(f"\n{len(unexecutable)} gold queries do not execute at all:")
        for index in unexecutable[:10]:
            print(f"  [{index}] {data.examples[index].db_id}: {data.examples[index].query}")
            print(f"        -> {gold_cache.results[index].error}")

    print("\nscoring gold against itself...")
    system = gold_self_consistency(data, paths=paths, config=config, gold_cache=gold_cache)
    summary = system.summary()

    failures = [s for s in system.scores if not s.execution_correct]
    print(f"\nexecution accuracy: {summary['execution_accuracy']:.4f} over {summary['n']}")
    print(f"exact match:        {summary['exact_match']:.4f}  (sanity: should be 1.0)")
    print(f"invalid SQL rate:   {summary['invalid_sql_rate']:.4f}")

    if failures:
        print(f"\n{len(failures)} gold queries did not score as correct:")
        for score in failures[:15]:
            print(f"  [{score.index}] {score.db_id}")
            print(f"      {score.gold}")
            print(f"      error: {score.error}")

    report = {
        "split": args.split,
        "summary": summary,
        "n_unexecutable_gold": len(unexecutable),
        "n_scoring_failures": len(failures),
        "failures": [
            {"index": s.index, "db_id": s.db_id, "gold": s.gold, "error": s.error}
            for s in failures
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")

    passed = summary["execution_accuracy"] >= PASS_THRESHOLD
    print(
        f"\n{'PASS' if passed else 'FAIL'}: gold self-consistency "
        f"{summary['execution_accuracy']:.4f} vs threshold {PASS_THRESHOLD}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
