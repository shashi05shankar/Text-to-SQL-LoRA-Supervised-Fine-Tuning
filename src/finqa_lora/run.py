"""The GPU run: split, train one arm (LoRA or QLoRA), score base vs tuned, write artifacts.

One call = one arm. Run it twice (`quantization="none"` and `"nf4"`) and let
`scripts/compare_arms.py` fold the two `metrics.json` files into the LoRA-vs-QLoRA table.
Base and tuned are scored in the same session, from the same loaded weights (adapter on/off),
so the ROUGE delta cannot come from the environment changing between runs.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from finqa_lora import __version__, rouge
from finqa_lora.compare import paired_bootstrap
from finqa_lora.data import download, load_examples, split_by_context
from finqa_lora.prompts import QADataset, build_inference_prompt
from text2sql_lora.config import DataConfig, LoraConfig, TrainConfig
from text2sql_lora.models.generator import (
    GenerationSettings,
    SqlGenerator,
    load_tokenizer,
)
from text2sql_lora.train.sft import train_lora


@dataclass(frozen=True)
class FinQAConfig:
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    max_seq_len: int = 768
    max_new_tokens: int = 128


def train_runtime_seconds(log_history: list[dict]) -> float | None:
    """The Trainer's own timer for the training loop (model load and eval generation excluded)."""
    for entry in reversed(log_history):
        if "train_runtime" in entry:
            return float(entry["train_runtime"])
    return None


def run(
    out_dir: Path,
    data_dir: Path,
    quantization: str = "nf4",
    device: str = "cuda",
    finqa_config: FinQAConfig | None = None,
    data_config: DataConfig | None = None,
    lora_config: LoraConfig | None = None,
    train_config: TrainConfig | None = None,
    test_limit: int = 0,
    max_steps: int = -1,
    generation_batch_size: int = 16,
) -> dict:
    """Train one arm and evaluate. `quantization` is "nf4" (QLoRA) or "none" (LoRA, fp16)."""
    if quantization not in ("nf4", "none"):
        raise ValueError("quantization must be 'nf4' or 'none'")
    import torch

    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    finqa_config = finqa_config or FinQAConfig()
    data_config = data_config or DataConfig()
    lora_config = lora_config or LoraConfig()
    # The ONLY difference between arms is the base precision.
    train_config = replace(train_config or TrainConfig(), load_in_4bit=quantization == "nf4")

    examples = load_examples(download(data_dir))
    train_idx, val_idx, test_idx = split_by_context(
        [e.context for e in examples],
        finqa_config.val_fraction,
        finqa_config.test_fraction,
        data_config.seed,
    )
    train = [examples[i] for i in train_idx]
    val = [examples[i] for i in val_idx]
    test = [examples[i] for i in test_idx]
    if test_limit:
        test = test[:test_limit]
    print(f"train {len(train)} / val {len(val)} / test {len(test)} (contexts disjoint)")

    tokenizer = load_tokenizer(data_config.base_model)
    train_ds = QADataset(train, tokenizer, finqa_config.max_seq_len)
    val_ds = QADataset(val, tokenizer, finqa_config.max_seq_len)
    print(f"{train_ds.truncation_rate():.2%} of training examples truncated")

    out_dir.mkdir(parents=True, exist_ok=True)
    model, trainer, params = train_lora(
        train_ds,
        val_ds,
        tokenizer,
        data_config,
        lora_config,
        train_config,
        output_dir=out_dir / "training",
        device=device,
        max_steps=max_steps,
    )
    train_seconds = train_runtime_seconds(trainer.state.log_history)

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    generator = SqlGenerator(
        model,
        tokenizer,
        settings=GenerationSettings(
            max_new_tokens=finqa_config.max_new_tokens, batch_size=generation_batch_size
        ),
        has_adapter=True,
    )
    prompts = [build_inference_prompt(tokenizer, e) for e in test]
    references = [e.answer for e in test]

    print("generating: base zero-shot (adapter off)")
    with generator.base_only():
        base_out = generator.generate(prompts)
    print("generating: tuned")
    tuned_out = generator.generate(prompts)

    base_rows = rouge.per_example(base_out, references)
    tuned_rows = rouge.per_example(tuned_out, references)
    base_scores, tuned_scores = rouge.mean_scores(base_rows), rouge.mean_scores(tuned_rows)
    gain = paired_bootstrap(
        [r["rouge1"] for r in base_rows], [r["rouge1"] for r in tuned_rows]
    )

    payload = {
        "arm": "qlora_nf4" if quantization == "nf4" else "lora_fp16",
        "quantization": quantization,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "base_zero_shot": base_scores,
        "tuned": tuned_scores,
        "rouge1_gain_bootstrap": gain,
        "train_seconds": train_seconds,
        "total_wall_clock_seconds": time.perf_counter() - started,
        "manifest": {
            "package_version": __version__,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "base_model": data_config.base_model,
            "seed": data_config.seed,
            "max_seq_len": finqa_config.max_seq_len,
            "train_truncation_rate": train_ds.truncation_rate(),
            "finqa": asdict(finqa_config),
            "lora": asdict(lora_config),
            "train": asdict(train_config),
            "params": params,
            "rouge_impl": "finqa_lora.rouge (lowercase alnum tokens, no stemming)",
        },
        "training": {"log_history": trainer.state.log_history},
    }
    if torch.cuda.is_available():
        payload["manifest"]["gpu"] = torch.cuda.get_device_name(0)
        payload["manifest"]["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9

    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as handle:
        for i, example in enumerate(test):
            handle.write(
                json.dumps(
                    {
                        "question": example.question,
                        "reference": example.answer,
                        "base_zero_shot": base_out[i],
                        "tuned": tuned_out[i],
                        "base_rouge1": base_rows[i]["rouge1"],
                        "tuned_rouge1": tuned_rows[i]["rouge1"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        f"\nROUGE-1  base {base_scores['rouge1']:.3f} -> tuned {tuned_scores['rouge1']:.3f}"
        f" | train {train_seconds and train_seconds / 60:.1f} min"
    )
    return payload
