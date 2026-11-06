import pytest

from finqa_lora import rouge
from finqa_lora.compare import paired_bootstrap, relative_speedup
from finqa_lora.data import QAExample, split_by_context
from finqa_lora.prompts import build_messages
from finqa_lora.run import train_runtime_seconds

pytestmark = pytest.mark.unit


def test_rouge1_known_values():
    assert rouge.rouge_n("the cat sat", "the cat sat", 1) == 1.0
    assert rouge.rouge_n("dog", "cat", 1) == 0.0
    # pred 4 tokens, ref 2, overlap 2 -> P=.5 R=1 F=2/3
    assert rouge.rouge_n("the cat sat down", "the cat", 1) == pytest.approx(2 / 3)


def test_rouge_clips_repeated_tokens():
    # overlap counts each token at most as often as the reference has it
    assert rouge.rouge_n("a a a a", "a b", 1) == pytest.approx(2 * 0.25 * 0.5 / 0.75)


def test_rouge_l_uses_subsequence_not_bag():
    assert rouge.rouge_l("a b c", "c b a") < 1.0
    assert rouge.rouge_l("a b c", "a b c") == 1.0


def test_rouge_matches_reference_package_when_installed():
    scorer_mod = pytest.importorskip("rouge_score.rouge_scorer")
    scorer = scorer_mod.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    pred = "Revenue grew 12% to $4.5 billion in fiscal 2023."
    ref = "Revenue increased 12% to $4.5 billion during fiscal 2023."
    ours, theirs = rouge.score(pred, ref), scorer.score(ref, pred)
    for key in ours:
        assert ours[key] == pytest.approx(theirs[key].fmeasure)


def test_split_keeps_contexts_disjoint_and_complete():
    contexts = [f"c{i % 40}" for i in range(400)]
    train, val, test = split_by_context(contexts, 0.1, 0.1, seed=1)
    assert sorted(train + val + test) == list(range(400))
    sets = [{contexts[i] for i in part} for part in (train, val, test)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert split_by_context(contexts, 0.1, 0.1, seed=1) == (train, val, test)


def test_split_rejects_bad_fractions():
    with pytest.raises(ValueError):
        split_by_context(["a", "b"], 0.6, 0.5, seed=0)


def test_prompt_contains_context_and_question_but_not_answer():
    ex = QAExample("What grew?", "Revenue grew.", "Revenue.")
    text = " ".join(m["content"] for m in build_messages(ex))
    assert "Revenue grew." in text and "What grew?" in text
    assert "\nAnswer: Revenue." not in text


def test_relative_speedup_sign():
    assert relative_speedup(100, 87.6) == pytest.approx(0.124)
    assert relative_speedup(100, 141) < 0  # slower candidate is negative, not clipped


def test_paired_bootstrap_detects_consistent_gain_and_null():
    base = [0.2, 0.3, 0.25, 0.28] * 25
    assert paired_bootstrap(base, [x + 0.1 for x in base])["excludes_zero"]
    assert not paired_bootstrap(base, list(base))["excludes_zero"]


def test_train_runtime_read_from_last_entry():
    log = [{"loss": 1.0}, {"train_runtime": 12.5, "train_loss": 0.4}]
    assert train_runtime_seconds(log) == 12.5
    assert train_runtime_seconds([{"loss": 1.0}]) is None
