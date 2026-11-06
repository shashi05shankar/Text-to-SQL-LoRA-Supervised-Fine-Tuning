"""Financial question answering by LoRA / QLoRA fine-tuning.

Reuses the generic pieces of `text2sql_lora` (model loading, LoRA attachment, the SFT
trainer wiring, completion-only loss masking) and adds the financial-QA specifics: data,
prompts, ROUGE scoring, and the LoRA-vs-QLoRA timing comparison.
"""

__version__ = "0.1.0"
