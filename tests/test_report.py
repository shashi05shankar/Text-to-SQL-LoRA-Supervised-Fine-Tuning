"""The README generator and its drift check.

The drift check is the enforcement mechanism behind this project's central rule -- no
number in the README that did not come out of a real run. So it is tested that it actually
fails when a number is tampered with, rather than merely that it passes when nothing is.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_report_module():
    spec = importlib.util.spec_from_file_location(
        "report_script", REPO_ROOT / "scripts" / "report.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def report():
    return load_report_module()


@pytest.fixture
def metrics() -> dict:
    """A metrics.json shaped exactly like the real one, with invented numbers.

    Invented is fine here: what is under test is the rendering, not the values.
    """
    return {
        "results": {
            "base_zero_shot": {
                "system": "base_zero_shot",
                "n": 1034,
                "execution_accuracy": 0.25,
                "exact_match": 0.05,
                "invalid_sql_rate": 0.30,
            },
            "base_few_shot": {
                "system": "base_few_shot",
                "n": 1034,
                "execution_accuracy": 0.40,
                "exact_match": 0.10,
                "invalid_sql_rate": 0.15,
            },
            "lora_zero_shot": {
                "system": "lora_zero_shot",
                "n": 1034,
                "execution_accuracy": 0.60,
                "exact_match": 0.30,
                "invalid_sql_rate": 0.02,
            },
        },
        "by_hardness": [
            {"system": "base_zero_shot", "hardness": "easy", "n": 200,
             "execution_accuracy": 0.5, "exact_match": 0.1},
            {"system": "lora_zero_shot", "hardness": "easy", "n": 200,
             "execution_accuracy": 0.8, "exact_match": 0.4},
        ],
        "headline": {
            "base_zero_shot_execution_accuracy": 0.25,
            "lora_zero_shot_execution_accuracy": 0.60,
            "absolute_gain": 0.35,
            "relative_gain": 1.4,
            "base_few_shot_execution_accuracy": 0.40,
            "gain_over_few_shot": 0.20,
        },
        "manifest": {
            "package_version": "0.1.0",
            "created_utc": "2026-08-10T12:00:00+00:00",
            "device": "cuda",
            "gpu": "Tesla T4",
            "torch": "2.4.0",
            "transformers": "4.44.0",
            "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
            "seed": 42,
            "n_train": 6300,
            "n_val": 700,
            "n_dev": 1034,
            "max_seq_len": 1024,
            "wall_clock_seconds": 7200.0,
            # As a real 4-bit run reports them: `total_params` sums numel() over the
            # quantized model, so it is roughly HALF the true parameter count.
            "trainable_params": 18_464_768,
            "total_params": 907_081_216,
            "trainable_percent": 2.036,
            "lora": {"r": 16, "alpha": 32},
        },
    }


def test_headline_delta_is_rendered(report, metrics):
    block = report.render(metrics)

    assert "25.0% -> 60.0%" in block
    assert "+35.0% absolute" in block
    assert "1,034 held-out questions" in block


def test_all_three_systems_appear(report, metrics):
    block = report.render(metrics)

    for label in ("Base, zero-shot", "Base, 3-shot", "LoRA fine-tuned"):
        assert label in block
    assert "0.6000" in block  # tuned execution accuracy, four decimals


def test_hardness_bucketing_is_labelled_as_ours(report, metrics):
    """It must never read as though it were Yale's official hardness definition."""
    block = report.render(metrics)

    assert "our own" in block.lower()
    assert "not Yale's official" in block


def test_provenance_reports_measured_parameter_counts(report, metrics):
    block = report.render(metrics)

    assert "18,464,768 trainable parameters" in block
    assert "the base weights never moved" in block
    assert "Tesla T4" in block


def test_lora_share_is_quoted_against_true_params_not_packed_ones(report, metrics):
    """A 4-bit model reports about half its real parameter count, so dividing by it
    overstates LoRA's share (2.04% instead of 1.20%). The README must quote the smaller,
    less flattering, correct figure -- and say why."""
    block = report.render(metrics)

    assert "1.20% of the base model's 1,543,714,304 parameters" in block
    assert "2.04%" in block  # the packed figure is disclosed...
    assert "not the one quoted here" in block  # ...and explicitly disowned
    assert "packs two weights per byte" in block


def test_unknown_base_model_still_explains_the_packed_count(report, metrics):
    """No hardcoded true-parameter count for this model: the note must degrade to the
    explanation rather than inventing a number."""
    metrics["manifest"]["base_model"] = "some/unknown-model"
    note = report.trainable_share_note(metrics["manifest"])

    assert "1,543,714,304" not in note
    assert "packs two weights per byte" in note


def test_share_note_is_empty_without_counts(report):
    assert report.trainable_share_note({}) == ""


def test_unexecutable_gold_is_disclosed_when_present(report, metrics):
    metrics["n_unexecutable_gold"] = 3
    block = report.render(metrics)
    assert "3 gold queries did not execute" in block


def test_splice_replaces_only_between_the_markers(report, metrics):
    readme = f"# Title\n\nintro\n\n{report.BEGIN}\nold content\n{report.END}\n\nfooter\n"
    updated = report.splice(readme, report.render(metrics))

    assert "old content" not in updated
    assert updated.startswith("# Title")
    assert updated.rstrip().endswith("footer")
    assert "25.0% -> 60.0%" in updated


def test_splice_without_markers_is_an_error(report, metrics):
    with pytest.raises(SystemExit):
        report.splice("# Title\n\nno markers here\n", report.render(metrics))


def test_check_mode_catches_a_hand_edited_number(report, metrics, tmp_path, monkeypatch):
    """The whole point: if someone types a nicer number into the README, this must fail."""
    from text2sql_lora.config import Paths

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    monkeypatch.setattr(report, "Paths", lambda: Paths(artifacts_dir=artifacts))

    readme = tmp_path / "README.md"
    readme.write_text(f"# T\n\n{report.BEGIN}\n{report.END}\n", encoding="utf-8")

    # Generate honestly -> check passes.
    monkeypatch.setattr("sys.argv", ["report.py", "--readme", str(readme)])
    assert report.main() == 0
    monkeypatch.setattr("sys.argv", ["report.py", "--check", str(readme)])
    assert report.main() == 0

    # Now inflate the headline by hand -> check must fail.
    tampered = readme.read_text(encoding="utf-8").replace("60.0%", "95.0%")
    readme.write_text(tampered, encoding="utf-8")
    assert report.main() == 1


@pytest.fixture
def analysis() -> dict:
    return {
        "n": 1034,
        "accuracy": {
            "base_zero_shot": {"accuracy": 0.41, "ci_low": 0.38, "ci_high": 0.44},
            "lora_zero_shot": {"accuracy": 0.65, "ci_low": 0.62, "ci_high": 0.68},
        },
        "comparisons": {
            "lora_zero_shot_vs_base_zero_shot": {
                "baseline": "base_zero_shot",
                "candidate": "lora_zero_shot",
                "mcnemar": {"gained": 308, "lost": 55, "p_value": 6.16e-40,
                            "significant_at_05": True},
                "bootstrap": {"ci_low": 0.21, "ci_high": 0.28},
            },
            "base_few_shot_vs_base_zero_shot": {
                "baseline": "base_zero_shot",
                "candidate": "base_few_shot",
                "mcnemar": {"gained": 134, "lost": 132, "p_value": 0.951,
                            "significant_at_05": False},
                "bootstrap": {"ci_low": -0.029, "ci_high": 0.034},
            },
        },
        "failures": {
            "base_zero_shot": {
                "n": 1034, "n_correct": 424, "invalid_rate": 0.325,
                "semantic_error_rate": 0.265, "semantic_share_of_errors": 0.449,
                "by_kind": {"correct": {"n": 424, "rate": 0.41},
                            "invalid: no such column": {"n": 257, "rate": 0.2485}},
            },
            "lora_zero_shot": {
                "n": 1034, "n_correct": 677, "invalid_rate": 0.150,
                "semantic_error_rate": 0.195, "semantic_share_of_errors": 0.566,
                "by_kind": {"correct": {"n": 677, "rate": 0.65},
                            "invalid: no such column": {"n": 141, "rate": 0.1364}},
            },
        },
        "grounding": {
            "base_zero_shot": {"parse_rate": 0.9942, "grounded_rate": 0.7311},
            "lora_zero_shot": {"parse_rate": 1.0, "grounded_rate": 0.8849},
        },
        "components": {
            "base_zero_shot": {"clause_f1": 0.6986, "join_count_match_rate": 0.5619,
                               "aggregate_match_rate": 0.8723,
                               "exact_shape_match_rate": 0.3066},
            "lora_zero_shot": {"clause_f1": 0.9164, "join_count_match_rate": 0.8182,
                               "aggregate_match_rate": 0.9381,
                               "exact_shape_match_rate": 0.7166},
        },
        "efficiency": {
            "adapter_mb": 85.3, "trainable_params": 18_464_768,
            "gpu": "Tesla T4", "wall_clock_hours": 4.23, "examples_per_second": 1.24,
        },
    }


def test_analysis_reports_significance_verdicts(report, analysis):
    block = report.render_analysis(analysis)

    assert "6.16e-40" in block
    assert "**significant**" in block
    assert "not significant" in block  # the few-shot comparison


def test_analysis_states_the_few_shot_null_result_plainly(report, analysis):
    """The most important sentence in the whole report: prompting did not do this."""
    block = report.render_analysis(analysis)
    assert "statistically indistinguishable" in block


def test_analysis_includes_failure_taxonomy_and_structure(report, analysis):
    block = report.render_analysis(analysis)

    assert "invalid: no such column" in block
    assert "Semantic share of errors" in block
    assert "Fully grounded" in block
    assert "0.8849" in block  # tuned grounding rate


def test_analysis_reports_cost(report, analysis):
    block = report.render_analysis(analysis)

    assert "85 MB" in block
    assert "Tesla T4" in block
    assert "18,464,768" in block


def test_analysis_block_is_empty_without_analysis(report):
    """report.py must still work from metrics.json alone."""
    assert report.render_analysis({}) == ""


def test_full_render_composes_both_blocks(report, metrics, analysis):
    combined = report.render(metrics) + report.render_analysis(analysis)

    assert "### Headline" in combined
    assert "### Is the difference real?" in combined
    assert "### What it got structurally right" in combined


def test_missing_metrics_is_a_failure_not_an_empty_report(report, tmp_path, monkeypatch):
    from text2sql_lora.config import Paths

    monkeypatch.setattr(report, "Paths", lambda: Paths(artifacts_dir=tmp_path / "nothing"))
    monkeypatch.setattr("sys.argv", ["report.py"])

    assert report.main() == 1
