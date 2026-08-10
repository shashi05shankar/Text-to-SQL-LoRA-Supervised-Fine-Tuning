"""End-to-end SFT pipeline test on a randomly-initialized tiny model.

The point is NOT accuracy -- a 2-layer random model knows no SQL and its numbers are
meaningless. The point is that every moving part actually moves: the dataset builds,
the collator pads labels correctly, LoRA attaches and freezes the base, `Trainer` steps,
the adapter saves and reloads, generation returns text, the harness scores it, and the
artifacts get written.

Those are the failures that would otherwise surface an hour into a Kaggle session. The
model is constructed from a config rather than downloaded, so this stays offline and fast.

Skipped when torch/peft are not installed -- the default local install is CPU-serving
only, and this suite must never require them.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch", reason="training deps not installed")
pytest.importorskip("peft", reason="training deps not installed")

from text2sql_lora.config import (  # noqa: E402
    DataConfig,
    EvalConfig,
    LoraConfig,
    Paths,
    TrainConfig,
)
from text2sql_lora.data.prompts import Example  # noqa: E402
from text2sql_lora.data.spider import SpiderData  # noqa: E402
from text2sql_lora.models.generator import (  # noqa: E402
    GenerationSettings,
    SqlGenerator,
    attach_lora,
    count_parameters,
    load_tokenizer,
)
from text2sql_lora.train.sft import PadCollator, SupervisedDataset  # noqa: E402

pytestmark = pytest.mark.integration

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return load_tokenizer(BASE_MODEL)
    except Exception as exc:
        pytest.skip(f"tokenizer unavailable offline: {exc}")


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory, tokenizer):
    """A 2-layer randomly-initialized Qwen2, saved to disk.

    Built from a config, not downloaded: the architecture is what is under test, not the
    weights.
    """
    from transformers import Qwen2Config, Qwen2ForCausalLM

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
    path = tmp_path_factory.mktemp("tiny_model")
    model.save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))
    return path


@pytest.fixture
def tiny_data():
    examples = [
        Example(
            "mini_school",
            f"How many students are aged {age}?",
            f"SELECT COUNT(*) FROM student WHERE age = {age}",
            "CREATE TABLE student (\n  student_id NUMBER PRIMARY KEY,\n  age NUMBER\n);",
        )
        for age in range(20, 26)
    ]
    return SpiderData(split="train", examples=examples, schemas={})


# --- dataset and collator ----------------------------------------------------


def test_dataset_builds_with_masked_prompts(tiny_data, tokenizer):
    dataset = SupervisedDataset(tiny_data, tokenizer, max_seq_len=512)

    assert len(dataset) == len(tiny_data.examples)
    record = dataset[0]
    assert len(record["input_ids"]) == len(record["labels"])
    assert -100 in record["labels"]  # the prompt is masked
    assert any(label != -100 for label in record["labels"])  # the target is not
    assert dataset.truncation_rate() == 0.0


def test_collator_pads_labels_with_ignore_index(tiny_data, tokenizer):
    """Padding labels with the pad token id instead of -100 would train the model to
    emit padding -- a bug that shows up only as mediocre output."""
    dataset = SupervisedDataset(tiny_data, tokenizer, max_seq_len=512)
    collator = PadCollator(tokenizer.pad_token_id)

    batch = collator([dataset[0], dataset[1]])

    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    # Wherever attention_mask is 0 (padding), the label must be ignored.
    padding = batch["attention_mask"] == 0
    assert torch.all(batch["labels"][padding] == -100)


def test_collator_pads_to_the_longest_member(tiny_data, tokenizer):
    dataset = SupervisedDataset(tiny_data, tokenizer, max_seq_len=512)
    collator = PadCollator(tokenizer.pad_token_id)
    records = [dataset[i] for i in range(3)]

    batch = collator(records)
    assert batch["input_ids"].shape[1] == max(len(r["input_ids"]) for r in records)


# --- LoRA attachment ---------------------------------------------------------


def test_lora_trains_a_small_fraction_and_freezes_the_base(tiny_model_dir):
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(str(tiny_model_dir))
    model = attach_lora(base, LoraConfig(r=4, alpha=8), for_training=False)

    params = count_parameters(model)
    assert 0 < params["trainable_params"] < params["total_params"]
    assert params["trainable_percent"] < 50.0

    # Every trainable tensor must be a LoRA tensor; nothing in the base may be trainable.
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable_names
    assert all("lora_" in name for name in trainable_names)


# --- the full loop -----------------------------------------------------------


def test_train_generate_score_and_persist(tiny_model_dir, tiny_data, tokenizer, tmp_path):
    """One pass through everything the Kaggle notebook does, at toy scale."""
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

    dataset = SupervisedDataset(tiny_data, tokenizer, max_seq_len=256)
    base = AutoModelForCausalLM.from_pretrained(str(tiny_model_dir))
    model = attach_lora(base, LoraConfig(r=4, alpha=8), for_training=False)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=2,
            per_device_train_batch_size=2,
            learning_rate=1e-3,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            remove_unused_columns=False,
            use_cpu=True,
        ),
        train_dataset=dataset,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )
    trainer.train()

    # The loss was actually recorded -- a silent no-op run would leave this empty.
    losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    assert losses
    assert all(loss == loss for loss in losses)  # no NaN

    # The adapter round-trips through disk.
    adapter_dir = tmp_path / "adapter"
    model.save_pretrained(str(adapter_dir))
    assert (adapter_dir / "adapter_config.json").exists()

    reloaded = AutoModelForCausalLM.from_pretrained(str(tiny_model_dir))
    from peft import PeftModel

    reloaded = PeftModel.from_pretrained(reloaded, str(adapter_dir))

    # Generation returns text, and the adapter can be switched off in place.
    generator = SqlGenerator(
        reloaded,
        tokenizer,
        settings=GenerationSettings(max_new_tokens=4, batch_size=2),
        has_adapter=True,
    )
    prompts = ["SELECT", "SELECT"]
    tuned = generator.generate(prompts)
    with generator.base_only():
        base_out = generator.generate(prompts)

    assert len(tuned) == len(base_out) == 2
    assert all(isinstance(text, str) for text in tuned + base_out)


def test_scoring_and_artifact_writing(tiny_data, mini_db, monkeypatch, tmp_path):
    """The scoring and persistence half of the run, with predictions supplied directly."""
    from text2sql_lora.eval.harness import compare, score_predictions
    from text2sql_lora.train.sft import write_predictions

    monkeypatch.setattr(Paths, "db_path", lambda self, db_id: mini_db)
    dev = SpiderData(
        split="validation",
        examples=[
            Example("mini_school", "How many students?", "SELECT COUNT(*) FROM student", "s"),
            Example("mini_school", "Names?", "SELECT name FROM student", "s"),
        ],
        schemas={},
    )

    base = score_predictions(
        "base_zero_shot", dev, ["SELCT bad", "SELCT bad"], config=EvalConfig(), progress_every=0
    )
    tuned = score_predictions(
        "lora_zero_shot",
        dev,
        [e.query for e in dev.examples],
        config=EvalConfig(),
        progress_every=0,
    )

    payload = compare([base, tuned])
    assert payload["headline"]["absolute_gain"] == 1.0

    predictions_path = tmp_path / "predictions.jsonl"
    write_predictions(predictions_path, dev, [base, tuned])

    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["base_zero_shot"]["execution_correct"] is False
    assert rows[0]["lora_zero_shot"]["execution_correct"] is True
    assert rows[0]["gold"] and rows[0]["question"] and rows[0]["hardness"]


def test_manifest_records_the_run(tmp_path):
    """The manifest is what makes a reported number auditable."""
    from text2sql_lora.train.sft import build_manifest

    manifest = build_manifest(
        DataConfig(),
        LoraConfig(),
        TrainConfig(),
        EvalConfig(),
        {"trainable_params": 100, "total_params": 10_000, "trainable_percent": 1.0},
        n_train=6300,
        n_val=700,
        n_dev=1034,
        truncation_rate=0.0234,
        wall_clock_seconds=123.4,
        device="cpu",
    )

    assert manifest["base_model"] == DataConfig().base_model
    assert manifest["n_dev"] == 1034
    assert manifest["trainable_percent"] == 1.0
    assert manifest["torch"] and manifest["transformers"] and manifest["created_utc"]
    assert manifest["lora"]["r"] == LoraConfig().r
