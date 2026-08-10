"""LoRA SFT: build the dataset, train, evaluate three systems, write the artifacts.

This module is the whole GPU run. The Kaggle notebook is thin orchestration around it --
it does not contain a second copy of anything, and in particular it does not contain a
second copy of the metric. The numbers in the README come out of the same `eval` package
that the local test suite exercises.

Run order, and why it is that order:

1. Build train/val from the 7,000 training examples, split by database.
2. `run_smoke()` -- a handful of steps on a few examples, to prove the pipeline moves
   before committing a session to it.
3. Train LoRA adapters on the frozen 4-bit base.
4. Evaluate THREE systems on Spider dev: base zero-shot, base 3-shot, tuned zero-shot.
   Dev is touched here and nowhere else.
5. Write `metrics.json`, `predictions.jsonl`, and the adapter.

Evaluation happens after training in the same session on purpose: the base numbers and the
tuned numbers then come from one machine, one library stack, and one decoding path, so the
delta cannot be an artifact of the environment changing between runs.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

from text2sql_lora import __version__
from text2sql_lora.config import (
    DataConfig,
    EvalConfig,
    LoraConfig,
    Paths,
    TrainConfig,
)
from text2sql_lora.data.prompts import (
    build_inference_prompt,
    build_supervised_example,
    select_few_shot,
)
from text2sql_lora.data.spider import SpiderData, load_schemas, load_split, train_val_split
from text2sql_lora.eval.harness import GoldCache, compare, score_predictions
from text2sql_lora.models.generator import (
    GenerationSettings,
    SqlGenerator,
    attach_lora,
    count_parameters,
    load_base_model,
    load_tokenizer,
    resolve_dtype,
)

# --- dataset -----------------------------------------------------------------


class SupervisedDataset:
    """Tokenized (prompt -> SQL) pairs with completion-only labels.

    A plain list-backed dataset rather than a `datasets.Dataset`: the corpus is 7,000 short
    examples and fits in memory many times over, and one fewer library version to pin on
    Kaggle is worth more than lazy loading here.
    """

    def __init__(self, data: SpiderData, tokenizer, max_seq_len: int) -> None:
        self.records: list[dict] = []
        self.n_truncated = 0

        for example in data.examples:
            record = build_supervised_example(tokenizer, example, max_seq_len=max_seq_len)
            self.n_truncated += int(record.pop("truncated"))
            record.pop("n_prompt_tokens")
            record.pop("n_target_tokens")
            self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        return self.records[index]

    def truncation_rate(self) -> float:
        return self.n_truncated / len(self.records) if self.records else 0.0


class PadCollator:
    """Pad a batch to its longest member.

    `input_ids` pad with the pad token; `labels` pad with -100 so padding contributes
    nothing to the loss. Padding labels with the pad token id instead would train the model
    to emit padding, which is the sort of bug that shows up only as mediocre output.
    """

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        import torch

        longest = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}

        for feature in features:
            pad_length = longest - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * pad_length)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_length)
            batch["labels"].append(feature["labels"] + [-100] * pad_length)

        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


# --- training ----------------------------------------------------------------


def build_training_arguments(
    output_dir: Path,
    config: TrainConfig,
    device: str,
    max_steps: int = -1,
):
    import torch
    from transformers import TrainingArguments

    use_cuda = device == "cuda" and torch.cuda.is_available()
    dtype = resolve_dtype(device)

    # Paged 8-bit Adam keeps optimizer state off the critical VRAM path and is the
    # standard QLoRA choice -- but it needs bitsandbytes, and a broken bnb install would
    # otherwise take the run down *after* the model has loaded. Degrade to plain AdamW
    # instead: on a 16 GB T4 with a 1.5B 4-bit base there is ample headroom for it.
    optimizer = "adamw_torch"
    if use_cuda:
        try:
            import bitsandbytes  # noqa: F401

            optimizer = "paged_adamw_8bit"
        except Exception as exc:
            print(f"bitsandbytes unavailable ({exc}); using adamw_torch")

    arguments = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.epochs,
        max_steps=max_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_grad_norm=config.max_grad_norm,
        gradient_checkpointing=config.gradient_checkpointing and use_cuda,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # T4 is Turing: fp16 only. bf16 needs compute capability 8.0+.
        fp16=use_cuda and dtype is torch.float16,
        bf16=use_cuda and dtype is torch.bfloat16,
        optim=optimizer,
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="no",  # only the final adapter is wanted; checkpoints eat the disk
        report_to=[],
        seed=DataConfig().seed,
        remove_unused_columns=False,
    )

    # QLoRA and DataParallel are incompatible, and the failure is ugly.
    #
    # `Trainer` wraps the model in `nn.DataParallel` whenever it sees more than one GPU
    # -- which a Kaggle "T4 x2" session always does. DataParallel *replicates* the model
    # onto each device every step, and a bitsandbytes 4-bit layer does not survive
    # replication: its quantization state does not come along, so the replica's weights
    # are garbage pointers. The result is `CUDA error: an illegal memory access` raised
    # from inside `gemm_4bit`, several frames deep in bnb, saying nothing about the
    # actual cause.
    #
    # Pinning to one GPU is the fix, not a workaround: this run is sized for a single
    # <=8 GB device by design, so the second GPU was never going to be used. Real
    # multi-GPU QLoRA needs DDP (one process per GPU), which is out of scope here.
    if use_cuda and torch.cuda.device_count() > 1:
        _ = arguments.device  # force device setup; this is what populates _n_gpu
        arguments._n_gpu = 1
        print(
            f"{torch.cuda.device_count()} GPUs visible; pinning to 1 "
            "(DataParallel corrupts 4-bit quantized weights)"
        )

    return arguments


def train_lora(
    train_dataset: SupervisedDataset,
    val_dataset: SupervisedDataset,
    tokenizer,
    data_config: DataConfig,
    lora_config: LoraConfig,
    train_config: TrainConfig,
    output_dir: Path,
    device: str = "cuda",
    max_steps: int = -1,
):
    """Attach LoRA to the frozen base and train. Returns `(model, trainer, param_counts)`."""
    from transformers import Trainer

    base = load_base_model(
        data_config.base_model, device=device, quantize=train_config.load_in_4bit
    )
    model = attach_lora(base, lora_config, for_training=True)

    params = count_parameters(model)
    print(
        f"trainable {params['trainable_params']:,} / {params['total_params']:,} "
        f"({params['trainable_percent']:.3f}%)"
    )

    trainer = Trainer(
        model=model,
        args=build_training_arguments(output_dir, train_config, device, max_steps=max_steps),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )

    model.config.use_cache = False  # incompatible with gradient checkpointing
    trainer.train()
    model.config.use_cache = True

    return model, trainer, params


# --- evaluation --------------------------------------------------------------


def build_prompts(tokenizer, data: SpiderData, few_shot=None) -> list[str]:
    return [
        build_inference_prompt(tokenizer, example, few_shot=few_shot)
        for example in data.examples
    ]


def evaluate_three_systems(
    generator: SqlGenerator,
    tokenizer,
    dev: SpiderData,
    train_pool: SpiderData,
    data_config: DataConfig,
    paths: Paths,
    eval_config: EvalConfig,
) -> tuple[dict, list]:
    """Run base zero-shot, base few-shot, and tuned zero-shot on the SAME dev set.

    The gold queries are executed once and reused across all three, since all three are
    scored against the same answers.
    """
    print("caching gold execution results...")
    gold_cache = GoldCache.build(dev, paths=paths, config=eval_config, progress_every=200)
    unexecutable = gold_cache.failures()
    if unexecutable:
        print(f"WARNING: {len(unexecutable)} gold queries do not execute; see metrics.json")

    zero_shot_prompts = build_prompts(tokenizer, dev)
    few_shot = select_few_shot(
        train_pool.examples, n=data_config.n_few_shot, seed=data_config.seed
    )
    few_shot_prompts = build_prompts(tokenizer, dev, few_shot=few_shot)

    systems = []

    print("\n[1/3] base, zero-shot (adapter disabled)")
    with generator.base_only():
        base_zero = generator.generate(zero_shot_prompts)
    systems.append(
        score_predictions("base_zero_shot", dev, base_zero, paths, eval_config, gold_cache)
    )

    print(f"\n[2/3] base, {data_config.n_few_shot}-shot (adapter disabled)")
    with generator.base_only():
        base_few = generator.generate(few_shot_prompts)
    systems.append(
        score_predictions("base_few_shot", dev, base_few, paths, eval_config, gold_cache)
    )

    print("\n[3/3] LoRA fine-tuned, zero-shot")
    tuned = generator.generate(zero_shot_prompts)
    systems.append(
        score_predictions("lora_zero_shot", dev, tuned, paths, eval_config, gold_cache)
    )

    payload = compare(systems)
    payload["n_unexecutable_gold"] = len(unexecutable)
    return payload, systems


def write_predictions(path: Path, dev: SpiderData, systems: list) -> None:
    """One JSONL row per dev question, carrying every system's prediction.

    This is the audit trail behind the headline: any number in the README can be traced to
    the individual predictions that produced it. It also lets the Streamlit demo show real
    base-vs-tuned comparisons instantly, without CPU generation.
    """
    by_name = {system.name: {s.index: s for s in system.scores} for system in systems}
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as handle:
        for index, example in enumerate(dev.examples):
            row = {
                "index": index,
                "db_id": example.db_id,
                "question": example.question,
                "gold": example.query,
                "hardness": next(
                    scores[index].hardness for scores in by_name.values() if index in scores
                ),
            }
            for name, scores in by_name.items():
                score = scores.get(index)
                if score is None:
                    continue
                row[name] = {
                    "prediction": score.prediction,
                    "execution_correct": score.execution_correct,
                    "exact_match": score.exact_match,
                    "invalid_sql": score.invalid_sql,
                    "error": score.error,
                }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_manifest(
    data_config: DataConfig,
    lora_config: LoraConfig,
    train_config: TrainConfig,
    eval_config: EvalConfig,
    params: dict,
    n_train: int,
    n_val: int,
    n_dev: int,
    truncation_rate: float,
    wall_clock_seconds: float,
    device: str,
) -> dict:
    """Everything needed to reproduce or challenge the run."""
    import torch
    import transformers

    manifest = {
        "package_version": __version__,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": device,
        "base_model": data_config.base_model,
        "seed": data_config.seed,
        "n_train": n_train,
        "n_val": n_val,
        "n_dev": n_dev,
        "max_seq_len": data_config.max_seq_len,
        "train_truncation_rate": truncation_rate,
        "wall_clock_seconds": wall_clock_seconds,
        "lora": asdict(lora_config),
        "train": asdict(train_config),
        "eval": asdict(eval_config),
        **params,
    }
    try:
        import peft

        manifest["peft"] = peft.__version__
    except ImportError:
        pass
    if torch.cuda.is_available():
        manifest["gpu"] = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        manifest["gpu_compute_capability"] = f"{major}.{minor}"
    return manifest


# --- entry points ------------------------------------------------------------


def run(
    device: str = "cuda",
    paths: Paths | None = None,
    data_config: DataConfig | None = None,
    lora_config: LoraConfig | None = None,
    train_config: TrainConfig | None = None,
    eval_config: EvalConfig | None = None,
    dev_limit: int = 0,
    max_steps: int = -1,
    generation_batch_size: int = 8,
) -> dict:
    """The full run: train, evaluate three systems, write artifacts. Returns the payload."""
    started = time.perf_counter()

    paths = paths or Paths()
    data_config = data_config or DataConfig()
    lora_config = lora_config or LoraConfig()
    train_config = train_config or TrainConfig()
    eval_config = eval_config or EvalConfig()

    schemas = load_schemas(paths)
    full_train = load_split("train", paths=paths, schemas=schemas)
    dev = load_split("validation", paths=paths, schemas=schemas)
    if dev_limit:
        dev = dev.subset(list(range(min(dev_limit, len(dev)))))

    train_data, val_data = train_val_split(full_train, data_config)
    print(
        f"train {len(train_data)} / val {len(val_data)} "
        f"({len(set(train_data.db_ids()))} vs {len(set(val_data.db_ids()))} databases, disjoint) "
        f"| dev {len(dev)}"
    )

    tokenizer = load_tokenizer(data_config.base_model)
    train_dataset = SupervisedDataset(train_data, tokenizer, data_config.max_seq_len)
    val_dataset = SupervisedDataset(val_data, tokenizer, data_config.max_seq_len)
    print(f"tokenized; {train_dataset.truncation_rate():.2%} of training examples truncated")

    model, trainer, params = train_lora(
        train_dataset,
        val_dataset,
        tokenizer,
        data_config,
        lora_config,
        train_config,
        output_dir=paths.artifacts_dir / "training",
        device=device,
        max_steps=max_steps,
    )

    adapter_dir = paths.adapter_dir()
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"saved adapter to {adapter_dir}")

    generator = SqlGenerator(
        model,
        tokenizer,
        settings=GenerationSettings(
            max_new_tokens=data_config.max_new_tokens, batch_size=generation_batch_size
        ),
        has_adapter=True,
    )

    payload, systems = evaluate_three_systems(
        generator, tokenizer, dev, train_data, data_config, paths, eval_config
    )

    payload["manifest"] = build_manifest(
        data_config,
        lora_config,
        train_config,
        eval_config,
        params,
        n_train=len(train_data),
        n_val=len(val_data),
        n_dev=len(dev),
        truncation_rate=train_dataset.truncation_rate(),
        wall_clock_seconds=time.perf_counter() - started,
        device=device,
    )
    payload["training"] = {"log_history": trainer.state.log_history}

    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    paths.metrics_json().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_predictions(paths.predictions_jsonl(), dev, systems)
    print(f"\nwrote {paths.metrics_json()} and {paths.predictions_jsonl()}")

    headline = payload.get("headline", {})
    if headline:
        print(
            f"\nEXECUTION ACCURACY  base {headline['base_zero_shot_execution_accuracy']:.4f} "
            f"-> tuned {headline['lora_zero_shot_execution_accuracy']:.4f} "
            f"(+{headline['absolute_gain']:.4f})"
        )
    return payload


def run_smoke(device: str = "cuda", paths: Paths | None = None) -> dict:
    """A few steps on a few examples. Proves the pipeline runs before a session is spent.

    Its numbers are meaningless and must never be reported -- it exists to surface shape
    errors, dtype errors, and OOM in two minutes rather than two hours.
    """
    print("SMOKE RUN -- the accuracy numbers below are meaningless by construction\n")
    return run(
        device=device,
        paths=paths,
        train_config=TrainConfig(epochs=1, eval_steps=5, logging_steps=1),
        dev_limit=8,
        max_steps=5,
        generation_batch_size=4,
    )
