# Running the training on Kaggle

Everything here happens once. When it finishes you download one zip, drop it into
`artifacts/`, and the rest of the project runs locally on CPU forever after.

**Assumed budget: ≤ 8 GB VRAM.** The configured run (4-bit NF4 + gradient checkpointing,
seq 1024, batch 2 × 8 accumulation) is sized to fit an 8 GB laptop GPU unchanged; a
Kaggle T4's 16 GB leaves plenty of headroom.

---

## 1. Get the repo onto Kaggle

**Route A — GitHub (preferred).** Push this repo to a public GitHub repo, then set
`REPO_URL` in the notebook's first code cell. The install cell clones it.

**Route B — attached dataset (no remote needed).** Zip the repo *without* `.venv/` and
`data/`, upload it at *Datasets → New Dataset → Upload*, and attach it to the notebook via
*Add Input*. The install cell finds it automatically by looking for a `pyproject.toml` that
mentions `text2sql-lora`.

```powershell
# From D:\Projects\text2sql-lora-sft
Compress-Archive -Path src,scripts,tests,notebooks,pyproject.toml,README.md -DestinationPath text2sql-lora-sft.zip -Force
```

## 2. Notebook settings

Three things in the right-hand sidebar, before running anything:

| Setting | Value | Why |
| --- | --- | --- |
| Accelerator | **GPU T4 x2** | P100 works too; only one GPU is used |
| Internet | **On** | needed for the base model and both Spider downloads |
| Persistence | off | everything is written to `/kaggle/working` and downloaded |

## 3. Run it

Open `notebooks/kaggle_lora_sft.ipynb` and run cells in order. The notebook has three
checkpoints that fail fast, in increasing order of cost:

1. **GPU check** — raises immediately if no accelerator is attached.
2. **Harness self-test** — scores 200 gold queries against themselves. Must be ~1.0. If it
   is not, the metric is broken and nothing after it would mean anything.
3. **Smoke run** — 5 optimizer steps on 8 dev examples, roughly two minutes. Its accuracy
   numbers are meaningless by construction; it exists to surface OOM, dtype, and shape
   errors before the real run.

Only then does the real run start. Budget **2–3.5 hours** end to end, inside Kaggle's 9 h
session limit.

## 4. What to download

The last cell writes `/kaggle/working/artifacts.zip`. Download it from the **Output** panel
on the right.

Unzip it into the repo's `artifacts/` directory so you end up with:

```
artifacts/
  adapter/              LoRA weights + adapter_config.json + tokenizer
  metrics.json          every number the README is allowed to quote
  predictions.jsonl     per-question audit trail; drives the Streamlit demo
  loss_curve.png
```

The base model is **not** in the zip and does not need to be — it is pulled from the Hub on
first local use. That is the whole point of LoRA: the trained part is tens of MB, not 3 GB.

```powershell
Expand-Archive -Path artifacts.zip -DestinationPath artifacts -Force
```

## 5. Then, locally

```bash
pip install -e ".[serve,ui]"
python scripts/report.py            # writes the real numbers into the README
pytest
```

---

## If something goes wrong

**CUDA out of memory.** Drop `TrainConfig.per_device_batch_size` to 1 and raise
`gradient_accumulation_steps` to 16 — the effective batch stays 16, so the run is
comparable. If it persists, lower `DataConfig.max_seq_len` to 768 (measured cost: 7.83% of
training examples truncate instead of 2.34%).

**`bitsandbytes` import error.** Usually a torch/bnb mismatch from Kaggle's preinstalled
stack. The install cell deliberately does not upgrade torch; if bnb still fails, set
`TrainConfig.load_in_4bit=False` — a 1.5B model in fp16 is about 3 GB and still fits a T4
without quantization. Record that you did this, because it changes the manifest.

**bf16 error.** Should not happen — the code selects fp16 on compute capability < 8.0 — but
if you see it, confirm the GPU check cell reported `float16 (Turing: no bf16)`.

**`ImportError: Found an incompatible version of torchao` inside `get_peft_model`.**
Kaggle preinstalls torchao 0.10.0, and PEFT's LoRA dispatcher *raises* on a version below
its minimum instead of skipping the optional backend. The install cell now uninstalls
torchao (we don't use it) and prints `torchao available to peft: False`. If you hit this
on an older clone, run this once, then re-run the training cell:

```python
import importlib, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)
importlib.invalidate_caches()
for m in [m for m in sys.modules if m == "torchao" or m.startswith("torchao.")]:
    del sys.modules[m]
import peft.import_utils as iu
iu.is_torchao_available.cache_clear()
print("torchao available to peft:", iu.is_torchao_available())  # must be False
```

**Session died mid-run.** Nothing is salvageable before the adapter is saved; rerun. The
smoke checkpoint exists specifically so that a rerun costs minutes of diagnosis, not hours.

**The gain is small.** Report it anyway. A modest honest number you can defend beats a
large one you cannot, and the three-system comparison (base zero-shot / base few-shot /
tuned) is what makes even a small delta interpretable.

---

## Optional: the LoRA vs QLoRA ablation

`notebooks/kaggle_lora_vs_qlora.ipynb` runs the identical recipe with **only** the base
precision changed, to measure what 4-bit quantization actually costs (accuracy) and saves
(VRAM).

1. Set `ARM = "lora_fp16"` in the arm-selection cell (the second cell). The default QLoRA
   run is already published, so this fp16 arm is the only one that needs GPU time.
2. Run all cells. It writes `artifacts/experiments/lora_fp16/`, and the last cell zips it.
3. Locally: unzip into `artifacts/experiments/lora_fp16/`, then

   ```bash
   python scripts/collect_experiments.py     # merges the fp16 arm with the published QLoRA one
   python scripts/report.py                   # renders the LoRA-vs-QLoRA table into the README
   python scripts/report.py --check README.md
   ```

The comparison table (accuracy, wall clock, peak VRAM, adapter size) appears only once two
arms exist. Expected: fp16 trains somewhat faster (no per-step dequantization on Turing)
and uses more VRAM (~4.8 GB vs ~2.6 GB), with accuracy close — but the point is to
**measure** that, not assume it.

Running the `qlora_nf4` arm here too is optional. It reproduces the published number on
the same hardware, which makes the VRAM and wall-clock figures exactly like-for-like
rather than comparing against a run from a different session.
