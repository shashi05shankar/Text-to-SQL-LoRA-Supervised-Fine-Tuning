"""Fold the two arms' metrics.json files into the LoRA-vs-QLoRA comparison.

    python scripts/compare_arms.py artifacts/finqa/lora_fp16 artifacts/finqa/qlora_nf4

Prints (and writes `comparison.json` next to the first arm) the ROUGE-1 of each system,
the training-time difference, and a paired bootstrap on tuned ROUGE-1 between the arms.
Every figure is read from the arms' artifacts; nothing is typed in.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finqa_lora.compare import paired_bootstrap, relative_speedup


def load_arm(directory: Path) -> tuple[dict, list[dict]]:
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    with open(directory / "predictions.jsonl", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    return metrics, rows


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    lora_dir, qlora_dir = Path(argv[1]), Path(argv[2])
    lora, lora_rows = load_arm(lora_dir)
    qlora, qlora_rows = load_arm(qlora_dir)

    if [r["question"] for r in lora_rows] != [r["question"] for r in qlora_rows]:
        print("arms were evaluated on different test questions; refusing to compare")
        return 1

    t_lora, t_qlora = lora["train_seconds"], qlora["train_seconds"]
    quality = paired_bootstrap(
        [r["tuned_rouge1"] for r in lora_rows], [r["tuned_rouge1"] for r in qlora_rows]
    )
    result = {
        "rouge1": {
            "base_zero_shot_lora_arm": lora["base_zero_shot"]["rouge1"],
            "base_zero_shot_qlora_arm": qlora["base_zero_shot"]["rouge1"],
            "lora_tuned": lora["tuned"]["rouge1"],
            "qlora_tuned": qlora["tuned"]["rouge1"],
        },
        "train_seconds": {"lora": t_lora, "qlora": t_qlora},
        # Positive = QLoRA saved that fraction of LoRA's time. Negative = QLoRA was slower.
        "qlora_time_saved_vs_lora": relative_speedup(t_lora, t_qlora),
        "lora_time_saved_vs_qlora": relative_speedup(t_qlora, t_lora),
        "qlora_minus_lora_rouge1": quality,
        "peak_vram_gb": {
            "lora": lora["manifest"].get("peak_vram_gb"),
            "qlora": qlora["manifest"].get("peak_vram_gb"),
        },
    }
    out = lora_dir.parent / "comparison.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    r = result["rouge1"]
    print(f"ROUGE-1  zero-shot {r['base_zero_shot_lora_arm']:.3f} (fp16 base) / "
          f"{r['base_zero_shot_qlora_arm']:.3f} (4-bit base)")
    print(f"         LoRA {r['lora_tuned']:.3f} | QLoRA {r['qlora_tuned']:.3f}  "
          f"(QLoRA-LoRA {quality['delta']:+.4f}, 95% CI "
          f"[{quality['ci_low']:+.4f}, {quality['ci_high']:+.4f}])")
    print(f"train    LoRA {t_lora / 60:.1f} min | QLoRA {t_qlora / 60:.1f} min  "
          f"(QLoRA time saved vs LoRA: {result['qlora_time_saved_vs_lora']:+.1%})")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
