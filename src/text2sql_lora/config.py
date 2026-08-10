"""Paths and configuration.

Every path is overridable by environment variable so the same code runs unchanged in
three places: this repo on a laptop, a Kaggle session (`/kaggle/working`), and a Docker
container (`/artifacts`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


@dataclass(frozen=True)
class Paths:
    """Where data and artifacts live."""

    data_dir: Path = field(default_factory=lambda: _env_path("T2S_DATA_DIR", REPO_ROOT / "data"))
    artifacts_dir: Path = field(
        default_factory=lambda: _env_path("T2S_ARTIFACTS_DIR", REPO_ROOT / "artifacts")
    )

    def __post_init__(self) -> None:
        """Accept strings as well as `Path`s.

        `Paths(artifacts_dir="/kaggle/working/smoke")` is the natural thing to write, and
        a dataclass will happily store the `str` -- the annotation is not enforced. The
        failure then surfaces much later and far away, as
        `TypeError: unsupported operand type(s) for /: 'str' and 'str'` inside a training
        call, which says nothing about where the bad value came from. Coerce at the
        boundary instead.

        `object.__setattr__` because the dataclass is frozen.
        """
        for name in ("data_dir", "artifacts_dir"):
            value = getattr(self, name)
            if not isinstance(value, Path):
                object.__setattr__(self, name, Path(value))

    # --- data ---------------------------------------------------------------
    def spider_dir(self) -> Path:
        """Root of the unpacked `spider_data.zip` (holds `database/` and `tables.json`)."""
        return self.data_dir / "spider_data"

    def database_dir(self) -> Path:
        return self.spider_dir() / "database"

    def tables_json(self) -> Path:
        return self.spider_dir() / "tables.json"

    def parquet(self, split: str) -> Path:
        """Question/SQL pairs for `train` or `validation`."""
        return self.data_dir / "spider_pairs" / f"{split}.parquet"

    def db_path(self, db_id: str) -> Path:
        return self.database_dir() / db_id / f"{db_id}.sqlite"

    # --- artifacts ----------------------------------------------------------
    def adapter_dir(self) -> Path:
        return self.artifacts_dir / "adapter"

    def metrics_json(self) -> Path:
        return self.artifacts_dir / "metrics.json"

    def predictions_jsonl(self) -> Path:
        return self.artifacts_dir / "predictions.jsonl"

    def loss_curve_png(self) -> Path:
        return self.artifacts_dir / "loss_curve.png"


@dataclass(frozen=True)
class DataConfig:
    """Prompt construction and the train/val split."""

    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    seed: int = 42

    # Fraction of the 7,000 training examples held out for training-time monitoring.
    # Split BY db_id, so no schema appears on both sides.
    val_fraction: float = 0.1

    # Number of in-context examples for the few-shot baseline. Fixed and seeded, so the
    # baseline is as reproducible as the fine-tuned model.
    n_few_shot: int = 3

    # Truncation budget, chosen from measurement rather than habit. Over all 7,000
    # training examples tokenized with the real Qwen2.5 tokenizer (see
    # artifacts/length_report.json, produced by scripts/measure_lengths.py):
    #   total tokens: mean 400.5, p50 308, p90 687, p99 2095, max 2232
    #   truncation at 512 -> 23.10%, 768 -> 7.83%, 1024 -> 2.34%, 1536 -> 1.17%
    # 1024 sits inside the 5% budget. Going to 1536 would recover only another 1.17% of
    # examples for 50% more sequence length -- the tail is a handful of very wide schemas,
    # not a broad problem.
    max_seq_len: int = 1024
    max_truncation_rate: float = 0.05

    # Generation budget at eval time. Spider gold queries are short; 256 is generous.
    max_new_tokens: int = 256


@dataclass(frozen=True)
class LoraConfig:
    """LoRA hyperparameters. Base weights stay frozen; only these matrices train."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


@dataclass(frozen=True)
class TrainConfig:
    """SFT hyperparameters, sized for a <=8 GB VRAM budget.

    Defaults assume a Turing-class GPU (Kaggle T4, compute capability 7.5), which does
    NOT support bfloat16 -- hence fp16 throughout. On Ampere or newer the notebook
    switches to bf16 automatically.
    """

    epochs: float = 3.0
    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8  # effective batch 16
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    # 6,282 training examples at an effective batch of 16 is ~1,178 steps over 3 epochs.
    # At eval_steps=100 that is ~11 passes over the 718-example validation split, which on
    # a T4 costs more wall clock than it buys: the curve only needs enough points to show
    # the shape. 200 gives ~6.
    eval_steps: int = 200
    gradient_checkpointing: bool = True

    # 4-bit NF4 quantization of the frozen base weights (QLoRA).
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass(frozen=True)
class EvalConfig:
    """Execution-accuracy comparison rules. Changing these changes the metric."""

    # Wall-clock budget for a single SQL execution. A timeout counts as INCORRECT,
    # never as skipped -- otherwise a model could inflate its score by emitting
    # queries that hang.
    timeout_seconds: float = 30.0

    # Floats are compared after rounding to this many decimals. SQLite returns
    # platform-dependent trailing digits for AVG/SUM that neither model controls.
    float_decimals: int = 4
