"""The serving layer: load once, generate on demand.

Model loading is **lazy**. A 1.5B model takes tens of seconds to load on CPU, and doing it
at import time would mean the container fails its health check before it ever finishes
starting. So `/health` answers immediately and reports whether the model is up yet.

Everything here is behind an injectable interface, so the API tests drive a stub and never
touch torch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from text2sql_lora.config import DataConfig, EvalConfig, Paths
from text2sql_lora.data.prompts import Example, build_messages
from text2sql_lora.data.schema import DatabaseSchema, parse_tables_json, render_schema
from text2sql_lora.eval.execute import execute_query
from text2sql_lora.eval.metrics import extract_sql


@dataclass
class GenerationOutcome:
    sql: str
    raw_output: str
    used_adapter: bool
    generation_seconds: float


class Text2SqlService:
    """Holds the schemas, the model, and the rules for turning a question into SQL."""

    def __init__(
        self,
        paths: Paths | None = None,
        data_config: DataConfig | None = None,
        eval_config: EvalConfig | None = None,
        generator=None,
        tokenizer=None,
    ) -> None:
        self.paths = paths or Paths()
        self.data_config = data_config or DataConfig()
        self.eval_config = eval_config or EvalConfig()

        self._generator = generator
        self._tokenizer = tokenizer
        self._schemas: dict[str, DatabaseSchema] | None = None

    # --- schemas ------------------------------------------------------------

    @property
    def schemas(self) -> dict[str, DatabaseSchema]:
        if self._schemas is None:
            tables_path = self.paths.tables_json()
            self._schemas = parse_tables_json(tables_path) if tables_path.exists() else {}
        return self._schemas

    def list_databases(self) -> list[str]:
        return sorted(self.schemas)

    def schema_text(self, db_id: str) -> str:
        if db_id not in self.schemas:
            raise KeyError(db_id)
        return render_schema(self.schemas[db_id])

    # --- model --------------------------------------------------------------

    @property
    def model_loaded(self) -> bool:
        return self._generator is not None

    @property
    def adapter_loaded(self) -> bool:
        return bool(getattr(self._generator, "has_adapter", False))

    def ensure_model(self):
        """Load base + adapter on first use.

        The adapter is optional: without it the service still serves the base model, which
        is a legitimate (if much worse) system and keeps the container useful before the
        training run has happened.
        """
        if self._generator is not None:
            return self._generator

        from text2sql_lora.models.generator import GenerationSettings, SqlGenerator

        adapter_dir: Path | None = self.paths.adapter_dir()
        if adapter_dir is not None and not adapter_dir.exists():
            print(f"[service] no adapter at {adapter_dir}; serving the BASE model only")
            adapter_dir = None

        self._generator = SqlGenerator.load(
            adapter_dir=adapter_dir,
            model_name=self.data_config.base_model,
            device="cpu",
            quantize=False,
            settings=GenerationSettings(
                max_new_tokens=self.data_config.max_new_tokens, batch_size=1
            ),
        )
        self._tokenizer = self._generator.tokenizer
        return self._generator

    # --- generation ---------------------------------------------------------

    def build_prompt(self, question: str, schema_text: str) -> str:
        example = Example(db_id="", question=question, query="", schema_text=schema_text)
        messages = build_messages(example)
        tokenizer = self._tokenizer or self.ensure_model().tokenizer
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def resolve_schema(self, db_id: str | None, schema_text: str | None) -> str:
        if schema_text:
            return schema_text
        if db_id is None:
            raise ValueError("provide either db_id or schema_text")
        return self.schema_text(db_id)

    def generate(
        self,
        question: str,
        db_id: str | None = None,
        schema_text: str | None = None,
        use_adapter: bool = True,
    ) -> GenerationOutcome:
        resolved_schema = self.resolve_schema(db_id, schema_text)
        generator = self.ensure_model()
        prompt = self.build_prompt(question, resolved_schema)

        started = time.perf_counter()
        if use_adapter or not generator.has_adapter:
            raw = generator.generate([prompt])[0]
        else:
            with generator.base_only():
                raw = generator.generate([prompt])[0]
        elapsed = time.perf_counter() - started

        return GenerationOutcome(
            sql=extract_sql(raw),
            raw_output=raw,
            used_adapter=use_adapter and generator.has_adapter,
            generation_seconds=elapsed,
        )

    def compare(
        self, question: str, db_id: str | None = None, schema_text: str | None = None
    ) -> tuple[GenerationOutcome, GenerationOutcome]:
        """Base and tuned for the same input, from one set of loaded weights."""
        base = self.generate(question, db_id, schema_text, use_adapter=False)
        tuned = self.generate(question, db_id, schema_text, use_adapter=True)
        return base, tuned

    # --- execution ----------------------------------------------------------

    def execute(self, db_id: str, sql: str):
        """Run generated SQL read-only against the real database."""
        return execute_query(
            self.paths.db_path(db_id), sql, timeout_seconds=self.eval_config.timeout_seconds
        )

    # --- artifacts ----------------------------------------------------------

    def metrics(self) -> dict | None:
        """The persisted results from the training run, or None before it has happened."""
        import json

        path = self.paths.metrics_json()
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
