"""Deep evaluation of a finished run. Writes `artifacts/analysis.json`.

`metrics.json` says *how often* each system was right. This says whether the difference
is real, *how* each system fails, and *what* it gets structurally right — the things an
accuracy number cannot tell you.

Everything here runs on CPU from artifacts the training run already produced. Predictions
are re-executed locally against the real Spider databases, which is cheap (SQLite) and is
what makes the failure taxonomy possible: `predictions.jsonl` records *that* a query was
wrong, not what it returned.

    python scripts/analyze.py
    python scripts/analyze.py --bootstrap 2000     # faster, slightly wider intervals
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql_lora.config import EvalConfig, Paths
from text2sql_lora.data.schema import parse_tables_json
from text2sql_lora.eval.errors import FailureKind, build_profile, classify
from text2sql_lora.eval.execute import execute_query
from text2sql_lora.eval.stats import compare_systems, wilson_interval
from text2sql_lora.eval.structure import (
    SchemaIndex,
    check_grounding,
    compare_shapes,
    query_shape,
    summarize_components,
    summarize_grounding,
)

SYSTEMS = ("base_zero_shot", "base_few_shot", "lora_zero_shot")
COMPARISONS = (
    ("base_zero_shot", "lora_zero_shot"),
    ("base_few_shot", "lora_zero_shot"),
    ("base_zero_shot", "base_few_shot"),
)


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"No {path}. Run the training notebook and unzip its artifacts first — "
            "see docs/kaggle_run.md."
        )
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--out", type=Path, default=Path("artifacts/analysis.json"))
    args = parser.parse_args()

    paths = Paths()
    eval_config = EvalConfig()
    rows = load_rows(paths.predictions_jsonl())
    schemas = parse_tables_json(paths.tables_json())
    n = len(rows)
    print(f"analysing {n} dev questions across {len(SYSTEMS)} systems\n")

    correct = {s: [bool(r[s]["execution_correct"]) for r in rows] for s in SYSTEMS}

    # --- 1. accuracy with intervals -----------------------------------------
    accuracy = {}
    for system in SYSTEMS:
        hits = sum(correct[system])
        low, high = wilson_interval(hits, n)
        accuracy[system] = {
            "n": n, "correct": hits, "accuracy": hits / n, "ci_low": low, "ci_high": high,
        }
        print(f"  {system:16} {hits / n:.4f}  [{low:.4f}, {high:.4f}]")

    # --- 2. is the difference real? -----------------------------------------
    print("\npaired significance tests...")
    comparisons = {}
    for baseline, candidate in COMPARISONS:
        comparison = compare_systems(
            baseline, correct[baseline], candidate, correct[candidate],
            n_resamples=args.bootstrap,
        )
        comparisons[f"{candidate}_vs_{baseline}"] = comparison.as_dict()
        mc = comparison.mcnemar
        print(
            f"  {candidate} vs {baseline}: +{mc.b_gained}/-{mc.b_lost}  "
            f"p={mc.p_value:.3g}  {'SIGNIFICANT' if mc.significant else 'not significant'}"
        )

    # --- 3. re-execute to build the failure taxonomy -------------------------
    print("\nre-executing predictions to classify failures...")
    gold_cache: dict[int, object] = {}
    kinds: dict[str, list[FailureKind]] = {s: [] for s in SYSTEMS}

    for i, row in enumerate(rows):
        db_path = paths.db_path(row["db_id"])
        if i not in gold_cache:
            gold_cache[i] = execute_query(
                db_path, row["gold"], timeout_seconds=eval_config.timeout_seconds
            )
        gold_result = gold_cache[i]

        for system in SYSTEMS:
            pred_result = execute_query(
                db_path, row[system]["prediction"], timeout_seconds=eval_config.timeout_seconds
            )
            kinds[system].append(
                classify(
                    gold_result, pred_result, row[system]["execution_correct"],
                    float_decimals=eval_config.float_decimals,
                )
            )
        if i and i % 250 == 0:
            print(f"  {i}/{n}")

    failures = {s: build_profile(s, kinds[s]).as_dict() for s in SYSTEMS}
    print()
    for system in SYSTEMS:
        f = failures[system]
        print(
            f"  {system:16} invalid {f['invalid_rate']:.3f} | "
            f"semantic {f['semantic_error_rate']:.3f} | "
            f"semantic share of errors {f['semantic_share_of_errors']:.3f}"
        )

    # --- 4. schema grounding + component matching ----------------------------
    print("\nparsing SQL for grounding and structure...")
    indexes = {db_id: SchemaIndex.from_schema(s) for db_id, s in schemas.items()}
    gold_shapes = [query_shape(r["gold"]) for r in rows]

    grounding = {}
    components = {}
    for system in SYSTEMS:
        ground_results = [
            check_grounding(r[system]["prediction"], indexes[r["db_id"]]) for r in rows
        ]
        matches = [
            compare_shapes(gold_shape, query_shape(r[system]["prediction"]))
            for gold_shape, r in zip(gold_shapes, rows, strict=True)
        ]
        grounding[system] = summarize_grounding(ground_results)
        components[system] = summarize_components(matches)

        # The identifiers most often invented — concrete evidence, not a rate.
        invented: Counter = Counter()
        for result in ground_results:
            invented.update(result.hallucinated_columns)
        grounding[system]["top_hallucinated_columns"] = [
            {"identifier": ident, "n": count} for ident, count in invented.most_common(10)
        ]

        print(
            f"  {system:16} parse {grounding[system]['parse_rate']:.3f} | "
            f"grounded {grounding[system]['grounded_rate']:.3f} | "
            f"clause F1 {components[system]['clause_f1']:.3f} | "
            f"shape match {components[system]['exact_shape_match_rate']:.3f}"
        )

    # --- 5. per-database ------------------------------------------------------
    per_db: dict[str, dict] = {}
    for i, row in enumerate(rows):
        entry = per_db.setdefault(
            row["db_id"], {"db_id": row["db_id"], "n": 0, **dict.fromkeys(SYSTEMS, 0)}
        )
        entry["n"] += 1
        for system in SYSTEMS:
            entry[system] += int(correct[system][i])
    per_database = sorted(
        (
            {
                "db_id": e["db_id"],
                "n": e["n"],
                **{s: e[s] / e["n"] for s in SYSTEMS},
                "delta": (e["lora_zero_shot"] - e["base_zero_shot"]) / e["n"],
            }
            for e in per_db.values()
        ),
        key=lambda e: e["lora_zero_shot"],
    )

    # --- 6. efficiency --------------------------------------------------------
    adapter_bytes = sum(
        f.stat().st_size for f in paths.adapter_dir().rglob("*") if f.is_file()
    ) if paths.adapter_dir().exists() else 0
    metrics = json.loads(paths.metrics_json().read_text(encoding="utf-8"))
    manifest = metrics.get("manifest", {})
    wall = manifest.get("wall_clock_seconds", 0.0)
    efficiency = {
        "adapter_bytes": adapter_bytes,
        "adapter_mb": adapter_bytes / 1e6,
        "trainable_params": manifest.get("trainable_params"),
        "wall_clock_seconds": wall,
        "wall_clock_hours": wall / 3600 if wall else None,
        "gpu": manifest.get("gpu"),
        "n_train": manifest.get("n_train"),
        "examples_per_second": (
            manifest["n_train"] * manifest.get("train", {}).get("epochs", 0) / wall
            if wall and manifest.get("n_train")
            else None
        ),
    }

    payload = {
        "n": n,
        "accuracy": accuracy,
        "comparisons": comparisons,
        "failures": failures,
        "grounding": grounding,
        "components": components,
        "per_database": per_database,
        "efficiency": efficiency,
        "source_metrics": str(paths.metrics_json()),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
