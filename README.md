# Financial QA by LoRA / QLoRA fine-tuning

Fine-tunes **Qwen2.5-1.5B-Instruct** on financial question answering over SEC 10-K filing
passages ([`virattt/financial-qa-10K`](https://huggingface.co/datasets/virattt/financial-qa-10K),
7,000 question/passage/answer rows) with **LoRA** and **QLoRA** (4-bit NF4), scores answer
quality with **ROUGE** against the zero-shot base, and benchmarks the two methods against
each other on **training time**, **answer quality** and **peak VRAM**.

## Status: code complete, GPU runs not yet done

No financial-QA result exists yet. Every figure in this project must come out of
`metrics.json` files written by the notebook below; none are quoted here until they do.

## Method

- **Task**: context-grounded QA. Prompt = passage + question; target = reference answer
  (`src/finqa_lora/prompts.py`). Same prompt for base and tuned model.
- **Split**: 80/10/10 train/val/test, **grouped by passage** so no passage appears on both
  sides (`data.split_by_context`). Test is 700 questions, fixed by seed 42.
- **Training**: shared with the earlier project — LoRA r=16, α=32 on all seven projections,
  completion-only loss masking, 3 epochs, effective batch 16, lr 2e-4 cosine, fp16 on a T4.
  The two arms differ in **one** thing: base precision (fp16 vs 4-bit NF4 + double quant).
- **Quality metric**: ROUGE-1/2/L F-measure (`src/finqa_lora/rouge.py`; lowercase alphanumeric
  tokens, no stemming — cross-checked against `rouge-score` in the tests when installed).
  Headline is ROUGE-1; a paired bootstrap gives a CI on every difference.
- **Speed metric**: the Trainer's `train_runtime` (training loop only — not model loading or
  evaluation generation). Both arms must run on the same GPU type.
- **Baseline context**: copying the passage verbatim scores ROUGE-1 ≈ 0.54 against the
  references (measured on all rows), because answers largely restate the passage. ROUGE is
  therefore a lexical-overlap measure here, not a correctness measure; read it with that in
  mind.

## Reproduce

1. Push this branch, then open `notebooks/kaggle_finqa_lora_qlora.ipynb` on Kaggle (GPU T4,
   Internet on). Run it with `ARM = "lora_fp16"`, then again with `ARM = "qlora_nf4"`.
2. Unzip both outputs to `artifacts/finqa/lora_fp16/` and `artifacts/finqa/qlora_nf4/`.
3. ```bash
   python scripts/compare_arms.py artifacts/finqa/lora_fp16 artifacts/finqa/qlora_nf4
   ```
   prints and writes `artifacts/finqa/comparison.json` (ROUGE-1 per system, training-time
   difference, bootstrap CI on the quality gap, peak VRAM).

Local checks (CPU only): `pip install -e ".[dev]"` then `pytest tests/finqa`.

## Layout

| Path | What |
| --- | --- |
| `src/finqa_lora/` | data, prompts, ROUGE, comparison stats, the one-arm GPU run |
| `scripts/compare_arms.py` | folds two arms into the LoRA-vs-QLoRA comparison |
| `notebooks/kaggle_finqa_lora_qlora.ipynb` | the Kaggle run (one arm per execution) |
| `src/text2sql_lora/` | earlier text-to-SQL project; its model loading, LoRA and trainer code is reused here |
| `docs/text2sql_spider_README.md` | the earlier project's full write-up (Spider, execution accuracy 41.0% → 65.5%) |
