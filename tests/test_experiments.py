"""The LoRA-vs-QLoRA collector and its README rendering."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from text2sql_lora.config import Paths

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def collect():
    return load_script("collect_experiments")


@pytest.fixture
def report():
    return load_script("report")


def make_metrics(quant: str, tuned_ex: float, wall_s: float, vram_gb: float) -> dict:
    return {
        "results": {
            "base_zero_shot": {"execution_accuracy": 0.41, "invalid_sql_rate": 0.32},
            "lora_zero_shot": {"execution_accuracy": tuned_ex, "invalid_sql_rate": 0.15},
        },
        "headline": {"absolute_gain": tuned_ex - 0.41},
        "manifest": {
            "quantization": quant,
            "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
            "seed": 42,
            "wall_clock_seconds": wall_s,
            "peak_vram_gb": vram_gb,
            "peak_vram_bytes": int(vram_gb * 1e9),
            "gpu": "Tesla T4",
            "lora": {"r": 16},
            "train": {"epochs": 3.0},
        },
    }


def write_arm(base: Path, name: str, metrics: dict, adapter_bytes: int = 1_000_000) -> None:
    arm_dir = base / "experiments" / name
    (arm_dir / "adapter").mkdir(parents=True, exist_ok=True)
    (arm_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (arm_dir / "adapter" / "adapter_model.safetensors").write_bytes(b"0" * adapter_bytes)


# --- collector ---------------------------------------------------------------


def test_collector_includes_the_published_run_as_an_arm(collect, tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    (artifacts / "adapter").mkdir(parents=True)
    (artifacts / "metrics.json").write_text(
        json.dumps(make_metrics("nf4", 0.6547, 15221.0, 2.6)), encoding="utf-8"
    )
    (artifacts / "adapter" / "w.safetensors").write_bytes(b"0" * 85_000_000)

    monkeypatch.setattr(collect, "Paths", lambda: Paths(artifacts_dir=artifacts))
    monkeypatch.setattr(
        "sys.argv",
        ["collect_experiments.py", "--out", str(artifacts / "experiments.json")],
    )
    assert collect.main() == 0

    payload = json.loads((artifacts / "experiments.json").read_text(encoding="utf-8"))
    assert payload["n_arms"] == 1
    arm = payload["arms"][0]
    assert arm["quantization"] == "nf4"
    assert arm["tuned_execution_accuracy"] == 0.6547
    assert arm["adapter_mb"] == pytest.approx(85.0, abs=1.0)


def test_collector_merges_published_and_experiment_arms(collect, tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    (artifacts / "adapter").mkdir(parents=True)
    (artifacts / "metrics.json").write_text(
        json.dumps(make_metrics("nf4", 0.6547, 15221.0, 2.6)), encoding="utf-8"
    )
    write_arm(artifacts, "lora_fp16", make_metrics("none", 0.66, 11000.0, 4.8))

    monkeypatch.setattr(collect, "Paths", lambda: Paths(artifacts_dir=artifacts))
    monkeypatch.setattr(
        "sys.argv",
        ["collect_experiments.py", "--out", str(artifacts / "experiments.json")],
    )
    assert collect.main() == 0

    arms = json.loads((artifacts / "experiments.json").read_text(encoding="utf-8"))["arms"]
    quants = {a["quantization"] for a in arms}
    assert quants == {"nf4", "none"}


def test_quantization_inferred_from_train_config_for_old_manifests(collect):
    """The published run predates the `quantization` field; its load_in_4bit flag must
    still identify it as the QLoRA arm rather than 'unknown'."""
    assert collect.infer_quantization({"quantization": "none"}) == "none"
    assert collect.infer_quantization({"train": {"load_in_4bit": True}}) == "nf4"
    assert collect.infer_quantization({"train": {"load_in_4bit": False}}) == "none"
    assert collect.infer_quantization({}) == "unknown"


def test_collector_reports_failure_when_no_runs_exist(collect, tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "Paths", lambda: Paths(artifacts_dir=tmp_path / "empty"))
    monkeypatch.setattr("sys.argv", ["collect_experiments.py", "--out", str(tmp_path / "x.json")])
    assert collect.main() == 1


# --- rendering ---------------------------------------------------------------


# NOTE: rendering of the LoRA-vs-QLoRA section is tested in tests/test_report.py, against
# a fixture shaped like the real collector output (base + tuned accuracy per arm, plus
# cross-arm significance). This file covers the collector that *produces* that structure.


def test_render_needs_at_least_two_arms(report):
    one_arm = {"arms": [{"name": "solo", "quantization": "nf4"}], "n_arms": 1}
    assert report.render_experiments(one_arm) == ""
    assert report.render_experiments({}) == ""


def test_render_tolerates_a_partially_populated_arm(report):
    """The published arm has no recorded VRAM; render must degrade, not crash."""
    experiments = {
        "arms": [
            {"name": "a", "quantization": "nf4", "base_execution_accuracy": 0.41,
             "tuned_execution_accuracy": 0.65, "wall_clock_hours": 4.2,
             "peak_vram_gb": None, "adapter_mb": 85.0},
            {"name": "b", "quantization": "none", "base_execution_accuracy": 0.48,
             "tuned_execution_accuracy": 0.62, "wall_clock_hours": 3.0,
             "peak_vram_gb": 8.13, "adapter_mb": 85.0},
        ],
        "n_arms": 2,
        "cross_arm_significance": [],
    }
    block = report.render_experiments(experiments)
    assert "not recorded" in block
    assert "None GB" not in block
