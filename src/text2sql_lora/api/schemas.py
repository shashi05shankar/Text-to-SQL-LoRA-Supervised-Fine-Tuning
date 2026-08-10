"""Request and response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class GenerateRequest(BaseModel):
    """Ask for SQL.

    Supply either a `db_id` (the schema is looked up from Spider's `tables.json`) or a
    `schema` string directly. The `schema` route is what makes this useful beyond Spider:
    the model was fine-tuned on serialized DDL, so any DDL works.
    """

    question: str = Field(..., min_length=1, description="Natural-language question")
    db_id: str | None = Field(None, description="Spider database id; schema is looked up")
    schema_text: str | None = Field(
        None, description="Raw CREATE TABLE text, if not using a db_id"
    )
    use_adapter: bool = Field(
        True,
        description="False disables the LoRA adapter, exposing the untouched base model",
    )
    execute: bool = Field(
        False, description="Also run the generated SQL (only possible with a db_id)"
    )

    @model_validator(mode="after")
    def require_a_schema_source(self) -> GenerateRequest:
        if not self.db_id and not self.schema_text:
            raise ValueError("provide either db_id or schema_text")
        if self.execute and not self.db_id:
            raise ValueError("execute=True requires a db_id -- there is nothing to run against")
        return self


class ExecutionPayload(BaseModel):
    ok: bool
    row_count: int | None = None
    rows: list[list] | None = None
    error: str | None = None
    elapsed_seconds: float


class GenerateResponse(BaseModel):
    sql: str
    raw_output: str
    db_id: str | None
    used_adapter: bool
    generation_seconds: float
    execution: ExecutionPayload | None = None


class CompareResponse(BaseModel):
    """Base vs fine-tuned for the same input, from one loaded set of weights."""

    question: str
    db_id: str | None
    base_sql: str
    tuned_sql: str
    generation_seconds: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    adapter_loaded: bool
    base_model: str
    n_databases: int
    metrics_available: bool


class SchemaResponse(BaseModel):
    db_id: str
    schema_text: str
    n_tables: int
