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

    # 1. The published run at the top level -- the QLoRA arm, no re-run needed.
    if paths.metrics_json().exists():
        metrics = json.loads(paths.metrics_json().read_text(encoding="utf-8"))
        quant = infer_quantization(metrics.get("manifest", {}))
        name = "lora_published" if quant == "none" else f"{quant}_published"
        arms.append(summarise(name, metrics, paths.adapter_dir()))
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
            print(f"  {arm_dir.name}: from {arm_metrics}")

    if not arms:
        print("No runs found. Train an arm first (see notebooks/kaggle_lora_vs_qlora.ipynb).")
        return 1

    payload = {"arms": arms, "n_arms": len(arms)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out} ({len(arms)} arms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
