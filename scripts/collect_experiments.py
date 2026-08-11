"""Gather experiment arms into one `artifacts/experiments.json`.

Each arm is a full run written to `artifacts/experiments/<name>/metrics.json` by the
Kaggle notebook. This walks those, plus the original top-level `artifacts/metrics.json`
(the published QLoRA run), and collects the fields the LoRA-vs-QLoRA comparison needs:
execution accuracy, wall clock, peak VRAM, adapter size, quantization.

The original run is included automatically and labelled `qlora_nf4`, so the fp16 arm is
the only one that ever needs re-running. No number is recomputed here — everything is read
straight from each run's own manifest and results.

    python scripts/collect_experiments.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql_lora.config import Paths
from text2sql_lora.eval.stats import compare_systems


def adapter_size(adapter_dir: Path) -> int:
    if not adapter_dir.exists():
        return 0
    return sum(f.stat().st_size for f in adapter_dir.rglob("*") if f.is_file())


def infer_quantization(manifest: dict) -> str:
    """Determine an arm's precision, even for artifacts written before the field existed.

    Runs from before `quantization` was added to the manifest still record the training
    config, and `load_in_4bit` there says the same thing -- so the published QLoRA run is
    correctly labelled `nf4` rather than `unknown`.
    """
    if "quantization" in manifest:
        return manifest["quantization"]
    load_in_4bit = manifest.get("train", {}).get("load_in_4bit")
    if load_in_4bit is True:
        return "nf4"
    if load_in_4bit is False:
        return "none"
    return "unknown"


def load_predictions(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def correctness(rows: list[dict], system: str) -> list[bool]:
    return [bool(r[system]["execution_correct"]) for r in rows]


def cross_arm_significance(arms: list[dict], predictions: dict[str, list[dict]]) -> list[dict]:
    """Paired McNemar + bootstrap between the first arm and each other arm.

    This is the part that makes a small headline difference defensible. The fp16 arm ended
    ~3 points below QLoRA -- worth nothing as a claim until a paired test on the identical
    1,034 questions says the gap is real and not resampling noise. Both `base_zero_shot`
    (the base under each arm's own precision) and `lora_zero_shot` (the tuned model) are
    compared, because the interesting story is that quantization changes them in *opposite*
    directions.
    """
    usable = [a["name"] for a in arms if predictions.get(a["name"])]
    if len(usable) < 2:
        return []

    reference = usable[0]
    ref_rows = predictions[reference]
    comparisons: list[dict] = []

    for other in usable[1:]:
        other_rows = predictions[other]
        if len(other_rows) != len(ref_rows):
            print(f"  skip {other} vs {reference}: different prediction counts")
            continue
        # The paired test is only valid if the rows line up question-for-question.
        aligned = all(
            a["db_id"] == b["db_id"] and a["question"] == b["question"]
            for a, b in zip(ref_rows, other_rows, strict=True)
        )
        if not aligned:
            print(f"  skip {other} vs {reference}: questions not aligned")
            continue

        for system, label in (("base_zero_shot", "base"), ("lora_zero_shot", "tuned")):
            comparison = compare_systems(
                reference, correctness(ref_rows, system),
                other, correctness(other_rows, system),
                n_resamples=10_000,
            )
            payload = comparison.as_dict()
            payload["system"] = label
            comparisons.append(payload)
            mc = comparison.mcnemar
            print(
                f"  {label}: {other} vs {reference}  "
                f"{comparison.candidate_accuracy:.4f} vs {comparison.baseline_accuracy:.4f}  "
                f"p={mc.p_value:.3g}  {'significant' if mc.significant else 'ns'}"
            )
    return comparisons


def summarise(name: str, metrics: dict, adapter_dir: Path) -> dict:
    """Pull the comparison fields out of one run's payload."""
    manifest = metrics.get("manifest", {})
    results = metrics.get("results", {})
    headline = metrics.get("headline", {})

    lora = results.get("lora_zero_shot", {})
    base = results.get("base_zero_shot", {})

    return {
        "name": name,
        "quantization": infer_quantization(manifest),
        "base_model": manifest.get("base_model"),
        "seed": manifest.get("seed"),
        "epochs": manifest.get("train", {}).get("epochs"),
        "lora_rank": manifest.get("lora", {}).get("r"),
        "base_execution_accuracy": base.get("execution_accuracy"),
        "tuned_execution_accuracy": lora.get("execution_accuracy"),
        "absolute_gain": headline.get("absolute_gain"),
        "tuned_invalid_sql_rate": lora.get("invalid_sql_rate"),
        "wall_clock_seconds": manifest.get("wall_clock_seconds"),
        "wall_clock_hours": (
            manifest.get("wall_clock_seconds", 0) / 3600
            if manifest.get("wall_clock_seconds")
            else None
        ),
        "peak_vram_gb": manifest.get("peak_vram_gb"),
        "adapter_mb": adapter_size(adapter_dir) / 1e6 if adapter_size(adapter_dir) else None,
        "gpu": manifest.get("gpu"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/experiments.json"))
    args = parser.parse_args()

    paths = Paths()
    arms: list[dict] = []
    predictions: dict[str, list[dict]] = {}

    # 1. The published run at the top level -- the QLoRA arm, no re-run needed. It is the
    #    reference arm (listed first), so cross-arm tests read "<other> vs published".
    if paths.metrics_json().exists():
        metrics = json.loads(paths.metrics_json().read_text(encoding="utf-8"))
        quant = infer_quantization(metrics.get("manifest", {}))
        name = "lora_fp16_published" if quant == "none" else f"{quant}_published"
        arms.append(summarise(name, metrics, paths.adapter_dir()))
        rows = load_predictions(paths.predictions_jsonl())
        if rows:
            predictions[name] = rows
        print(f"  {name}: from {paths.metrics_json()}")

    # 2. Every arm under artifacts/experiments/.
    experiments_dir = paths.artifacts_dir / "experiments"
    if experiments_dir.exists():
        for arm_dir in sorted(experiments_dir.iterdir()):
            arm_metrics = arm_dir / "metrics.json"
            if not arm_metrics.exists():
                continue
            metrics = json.loads(arm_metrics.read_text(encoding="utf-8"))
            arms.append(summarise(arm_dir.name, metrics, arm_dir / "adapter"))
            rows = load_predictions(arm_dir / "predictions.jsonl")
            if rows:
                predictions[arm_dir.name] = rows
            print(f"  {arm_dir.name}: from {arm_metrics}")

    if not arms:
        print("No runs found. Train an arm first (see notebooks/kaggle_lora_vs_qlora.ipynb).")
        return 1

    print("\ncross-arm paired significance (vs the reference arm)...")
    significance = cross_arm_significance(arms, predictions)

    payload = {"arms": arms, "n_arms": len(arms), "cross_arm_significance": significance}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out} ({len(arms)} arms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
