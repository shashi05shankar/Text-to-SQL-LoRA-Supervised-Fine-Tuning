"""Run the ENTIRE Kaggle pipeline locally on CPU with a toy model.

`train.sft.run()` is what the notebook calls. This script calls exactly that function --
same code path, same artifacts, same metric -- but swaps the 1.5B base model for a
2-layer randomly-initialized one and shrinks everything else. It finishes in a couple of
minutes on a laptop with no GPU.

**Its accuracy numbers are meaningless and must never be reported.** A random model knows
no SQL. What this proves is that the pipeline runs to completion and produces
well-formed artifacts -- which is the thing you want to know *before* spending a Kaggle
session, not an hour into one.

    python scripts/smoke_local.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql_lora.config import DataConfig, EvalConfig, LoraConfig, Paths, TrainConfig
from text2sql_lora.models.generator import load_tokenizer


def build_tiny_model(target: Path, base_model: str) -> Path:
    """A 2-layer Qwen2 with random weights, saved where `from_pretrained` can find it."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    tokenizer = load_tokenizer(base_model)
    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        tie_word_embeddings=True,
    )
    model = Qwen2ForCausalLM(config)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(target))
    tokenizer.save_pretrained(str(target))
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-limit", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--keep", action="store_true", help="keep the scratch artifacts")
    args = parser.parse_args()

    from text2sql_lora.train.sft import run

    workspace = Path(tempfile.mkdtemp(prefix="t2s_smoke_"))
    print(f"scratch workspace: {workspace}\n")

    try:
        tiny = build_tiny_model(workspace / "tiny_model", DataConfig().base_model)

        paths = Paths(artifacts_dir=workspace / "artifacts")
        run(
            device="cpu",
            paths=paths,
            data_config=replace(
                DataConfig(), base_model=str(tiny), max_seq_len=256, max_new_tokens=16
            ),
            lora_config=LoraConfig(r=4, alpha=8),
            train_config=replace(
                TrainConfig(),
                epochs=1,
                per_device_batch_size=2,
                gradient_accumulation_steps=1,
                logging_steps=1,
                eval_steps=2,
                gradient_checkpointing=False,
                load_in_4bit=False,
            ),
            eval_config=EvalConfig(timeout_seconds=10.0),
            dev_limit=args.dev_limit,
            max_steps=args.max_steps,
            generation_batch_size=4,
        )

        print("\n--- artifact check ---")
        metrics_path = paths.metrics_json()
        predictions_path = paths.predictions_jsonl()

        written = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows = [
            json.loads(line)
            for line in predictions_path.read_text(encoding="utf-8").splitlines()
        ]

        problems: list[str] = []
        for key in ("results", "by_hardness", "headline", "manifest", "training"):
            if key not in written:
                problems.append(f"metrics.json is missing {key!r}")
        for system in ("base_zero_shot", "base_few_shot", "lora_zero_shot"):
            if system not in written.get("results", {}):
                problems.append(f"no results for {system!r}")
        if len(rows) != args.dev_limit:
            problems.append(f"predictions.jsonl has {len(rows)} rows, expected {args.dev_limit}")
        if not written.get("training", {}).get("log_history"):
            problems.append("no training log_history -- the loss curve would be empty")
        if not paths.adapter_dir().joinpath("adapter_config.json").exists():
            problems.append("no adapter_config.json in the adapter directory")

        adapter_bytes = sum(
            f.stat().st_size for f in paths.adapter_dir().rglob("*") if f.is_file()
        )
        print(f"metrics.json      : {metrics_path.stat().st_size:,} bytes")
        print(f"predictions.jsonl : {len(rows)} rows")
        print(f"adapter/          : {adapter_bytes:,} bytes")
        print(f"systems evaluated : {sorted(written.get('results', {}))}")
        print(f"trainable params  : {written['manifest']['trainable_params']:,} "
              f"({written['manifest']['trainable_percent']:.3f}%)")

        if problems:
            print("\nFAIL:")
            for problem in problems:
                print(f"  - {problem}")
            return 1

        print("\nPASS: the pipeline runs end to end and writes well-formed artifacts.")
        print("(The accuracy numbers above are from a random model. They mean nothing.)")
        return 0

    finally:
        if args.keep:
            print(f"\nkept: {workspace}")
        else:
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
