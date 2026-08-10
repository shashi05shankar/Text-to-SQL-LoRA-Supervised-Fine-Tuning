"""Path configuration.

The string-coercion tests exist because of a real failure: passing
`Paths(artifacts_dir="/kaggle/working/smoke")` -- a perfectly natural thing to write --
stored a `str`, and the run died much later inside the trainer with
`TypeError: unsupported operand type(s) for /: 'str' and 'str'`, an error that points
nowhere near the actual cause.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from text2sql_lora.config import DataConfig, EvalConfig, LoraConfig, Paths, TrainConfig

pytestmark = pytest.mark.unit


def test_string_paths_are_coerced():
    paths = Paths(data_dir="/tmp/data", artifacts_dir="/tmp/artifacts")

    assert isinstance(paths.data_dir, Path)
    assert isinstance(paths.artifacts_dir, Path)


def test_derived_paths_work_from_string_inputs():
    """The original bug: every derived path uses `/`, which a str does not support."""
    paths = Paths(data_dir="/tmp/data", artifacts_dir="/tmp/artifacts")

    assert paths.artifacts_dir / "training"
    for derived in (
        paths.spider_dir(),
        paths.database_dir(),
        paths.tables_json(),
        paths.parquet("train"),
        paths.db_path("concert_singer"),
        paths.adapter_dir(),
        paths.metrics_json(),
        paths.predictions_jsonl(),
        paths.loss_curve_png(),
    ):
        assert isinstance(derived, Path)


def test_path_objects_pass_through_unchanged(tmp_path):
    paths = Paths(data_dir=tmp_path / "d", artifacts_dir=tmp_path / "a")
    assert paths.data_dir == tmp_path / "d"


def test_env_overrides_are_honoured(monkeypatch, tmp_path):
    monkeypatch.setenv("T2S_DATA_DIR", str(tmp_path / "envdata"))
    monkeypatch.setenv("T2S_ARTIFACTS_DIR", str(tmp_path / "envartifacts"))

    paths = Paths()

    assert paths.data_dir == tmp_path / "envdata"
    assert paths.artifacts_dir == tmp_path / "envartifacts"


def test_db_path_layout_matches_spider(tmp_path):
    paths = Paths(data_dir=tmp_path)
    expected = tmp_path / "spider_data" / "database" / "concert_singer" / "concert_singer.sqlite"
    assert paths.db_path("concert_singer") == expected


def test_configs_are_frozen_so_a_run_cannot_mutate_them():
    """The manifest records these values; if they could drift mid-run it would misreport."""
    import dataclasses

    for config in (DataConfig(), LoraConfig(), TrainConfig(), EvalConfig()):
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.seed = 1  # type: ignore[misc]


def test_effective_batch_size_is_what_the_readme_claims():
    config = TrainConfig()
    assert config.per_device_batch_size * config.gradient_accumulation_steps == 16


def test_lora_targets_cover_attention_and_mlp():
    targets = set(LoraConfig().target_modules)
    assert {"q_proj", "k_proj", "v_proj", "o_proj"} <= targets
    assert {"gate_proj", "up_proj", "down_proj"} <= targets
