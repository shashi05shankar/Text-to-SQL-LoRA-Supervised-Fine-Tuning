"""Load the base model (optionally with a LoRA adapter) and generate SQL.

One loader serves three callers -- Kaggle training, Kaggle evaluation, and local CPU
serving -- so the model that produces the README's numbers is the model the API serves.

The base-vs-tuned comparison uses peft's `disable_adapter()` context manager rather than
loading two models. That is not just a memory saving: it is a direct demonstration of what
LoRA is. There is one set of base weights in memory, untouched, and turning the adapter off
returns the original model exactly. Loading a separate "base model" would leave room to
wonder whether the two runs differed in some other way.

Generation is greedy (`do_sample=False`) everywhere. A sampled metric would change between
runs, and "execution accuracy 41%" would mean "41%, plus or minus whatever the seed did."
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from text2sql_lora.config import DataConfig, LoraConfig, Paths


@dataclass
class GenerationSettings:
    max_new_tokens: int = 256
    batch_size: int = 8


def resolve_dtype(device: str):
    """Pick a compute dtype the hardware actually supports.

    bfloat16 needs compute capability 8.0+ (Ampere). The Kaggle T4 is Turing (7.5), so it
    must use float16 -- asking for bf16 there raises rather than falling back.
    """
    import torch

    if device != "cuda" or not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability(0)
    return torch.bfloat16 if major >= 8 else torch.float16


def resolve_serving_dtype():
    """Dtype for CPU inference, where MEMORY is the binding constraint, not arithmetic.

    Measured on an 8 GB laptop: at float32 the 1.5B base model needs ~6.2 GB of weights,
    the machine pages to disk, and a single query took **369 seconds**. bfloat16 halves
    that to ~3.1 GB, which fits in RAM.

    So the usual advice -- "CPU fp16/bf16 is slower than fp32" -- is about arithmetic
    throughput and only holds when the weights fit. When they do not, halving the memory
    traffic dominates everything else.

    Override with `T2S_SERVING_DTYPE=float32` on a machine with RAM to spare.
    """
    import os

    import torch

    requested = os.environ.get("T2S_SERVING_DTYPE", "bfloat16").lower()
    return {"float32": torch.float32, "float16": torch.float16}.get(
        requested, torch.bfloat16
    )


def dtype_kwarg(compute_dtype) -> dict:
    """Name the dtype argument the way the installed transformers expects.

    transformers 5.x renamed `torch_dtype` to `dtype` and warns or errors on the old name;
    4.x only knows `torch_dtype`. Kaggle's preinstalled version is not under our control
    and moves without notice, so this is resolved by inspection rather than assumption.
    """
    import inspect

    from transformers import AutoModelForCausalLM

    try:
        signature = inspect.signature(AutoModelForCausalLM.from_pretrained)
        if "dtype" in signature.parameters:
            return {"dtype": compute_dtype}
    except (TypeError, ValueError):
        pass

    import transformers

    major = int(transformers.__version__.split(".")[0])
    return {"dtype": compute_dtype} if major >= 5 else {"torch_dtype": compute_dtype}


def build_quantization_config(compute_dtype):
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only generation must pad on the LEFT. With right padding, the model
    # continues from pad tokens instead of from the prompt and emits garbage -- and it
    # does so silently, producing a plausible-looking but meaningless accuracy number.
    tokenizer.padding_side = "left"
    return tokenizer


def load_base_model(
    model_name: str,
    device: str = "cuda",
    quantize: bool = True,
    dtype=None,
):
    """Load the frozen base model. `dtype=None` picks one from the hardware."""
    import torch
    from transformers import AutoModelForCausalLM

    compute_dtype = dtype if dtype is not None else resolve_dtype(device)
    kwargs: dict = {**dtype_kwarg(compute_dtype), "attn_implementation": "sdpa"}

    if quantize and device == "cuda" and torch.cuda.is_available():
        kwargs["quantization_config"] = build_quantization_config(compute_dtype)
        kwargs["device_map"] = {"": 0}
    else:
        # CPU serving: 4-bit needs bitsandbytes and a GPU, so load dense fp32.
        kwargs["device_map"] = None

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if kwargs.get("device_map") is None:
        model = model.to("cpu" if device != "cuda" else device)
    return model


def attach_lora(
    model,
    lora_config: LoraConfig | None = None,
    for_training: bool = True,
    gradient_checkpointing: bool = True,
):
    """Wrap the frozen base model in freshly initialized LoRA adapters.

    Two paths, because a 4-bit base needs preparation an fp16 one does not:

    * **4-bit base** -- `prepare_model_for_kbit_training` casts layer norms to fp32 and
      enables input grads. Quantized training needs the fp32 norms.
    * **Unquantized base** -- `enable_input_require_grads()` is called defensively.

    On the versions pinned here (torch 2.13, peft 0.20) that second call is a **no-op**:
    gradient flow to the LoRA matrices was measured identical with and without it, under
    both `use_reentrant=True` and `False`. It is kept because it is the documented pattern
    for LoRA under gradient checkpointing and costs nothing -- older stacks *did* need it,
    where the checkpointed segment saw no differentiable input, silently produced no
    gradient, and left the loss flat with an untrained adapter.

    `tests/train/test_sft_smoke.py` asserts the contract directly rather than trusting
    any of this: LoRA parameters must receive non-zero gradients, and base weights must
    receive none.
    """
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model, prepare_model_for_kbit_training

    lora_config = lora_config or LoraConfig()

    if for_training and gradient_checkpointing:
        if getattr(model, "is_loaded_in_4bit", False):
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        elif hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    peft_config = PeftLoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout,
        target_modules=list(lora_config.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, peft_config)


def load_adapter(model, adapter_dir: Path | str):
    """Attach trained LoRA weights to a base model for inference."""
    from peft import PeftModel

    adapter_dir = Path(adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"no LoRA adapter at {adapter_dir}. Download it from the Kaggle run into "
            "artifacts/adapter/ (see the README)."
        )
    return PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)


def count_parameters(model) -> dict:
    """Report exactly how few parameters LoRA trains.

    Returned rather than printed so it lands in metrics.json as a measured fact. The
    README quotes this number; it is never estimated.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_params": int(trainable),
        "total_params": int(total),
        "trainable_percent": 100.0 * trainable / total if total else 0.0,
    }


class SqlGenerator:
    """Greedy batched generation, with the adapter switchable on and off."""

    def __init__(
        self,
        model,
        tokenizer,
        settings: GenerationSettings | None = None,
        has_adapter: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.settings = settings or GenerationSettings()
        self.has_adapter = has_adapter
        self.model.eval()

    @classmethod
    def load(
        cls,
        adapter_dir: Path | str | None = None,
        model_name: str | None = None,
        device: str = "cpu",
        quantize: bool = False,
        settings: GenerationSettings | None = None,
        dtype=None,
    ) -> SqlGenerator:
        model_name = model_name or DataConfig().base_model
        tokenizer = load_tokenizer(model_name)
        if dtype is None and device != "cuda":
            dtype = resolve_serving_dtype()
        model = load_base_model(model_name, device=device, quantize=quantize, dtype=dtype)

        has_adapter = False
        if adapter_dir is not None:
            model = load_adapter(model, adapter_dir)
            has_adapter = True

        return cls(model, tokenizer, settings=settings, has_adapter=has_adapter)

    @contextlib.contextmanager
    def base_only(self):
        """Temporarily switch the adapter off, exposing the untouched base model."""
        if not self.has_adapter:
            yield
            return
        with self.model.disable_adapter():
            yield

    def generate(self, prompts: Sequence[str]) -> list[str]:
        """Greedy-decode a batch of prompts, returning only the newly generated text."""
        import torch

        outputs: list[str] = []
        device = next(self.model.parameters()).device

        for start in range(0, len(prompts), self.settings.batch_size):
            batch = list(prompts[start : start + self.settings.batch_size])
            encoded = self.tokenizer(
                batch, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(device)

            with torch.no_grad():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=self.settings.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Slice off the prompt so only the completion is decoded. Left padding makes
            # the prompt length uniform across the batch, so one offset works for all.
            prompt_length = encoded["input_ids"].shape[1]
            for sequence in generated:
                text = self.tokenizer.decode(
                    sequence[prompt_length:], skip_special_tokens=True
                )
                outputs.append(text.strip())

        return outputs

    def generate_both(self, prompts: Sequence[str]) -> tuple[list[str], list[str]]:
        """Generate with the adapter off and on, from one loaded set of weights."""
        with self.base_only():
            base = self.generate(prompts)
        tuned = self.generate(prompts)
        return base, tuned


def default_adapter_dir(paths: Paths | None = None) -> Path:
    return (paths or Paths()).adapter_dir()
