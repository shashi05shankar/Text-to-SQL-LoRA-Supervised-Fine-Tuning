"""FastAPI app.

The service is provided through a dependency override point, so tests inject a stub and
never load a real model. `/health` deliberately does not force the model to load -- it
answers instantly and reports whether the model is up, which is what a container health
check needs.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException

from text2sql_lora import __version__
from text2sql_lora.api.schemas import (
    CompareResponse,
    ExecutionPayload,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    SchemaResponse,
)
from text2sql_lora.api.service import Text2SqlService

app = FastAPI(
    title="Text-to-SQL (LoRA fine-tuned Qwen2.5-1.5B)",
    version=__version__,
    description=(
        "Generate SQL from a question and a schema. The same endpoint serves the "
        "fine-tuned model and, with use_adapter=false, the untouched base model -- from "
        "one set of loaded weights."
    ),
)

_service: Text2SqlService | None = None


def get_service() -> Text2SqlService:
    global _service
    if _service is None:
        _service = Text2SqlService()
    return _service


@app.get("/health", response_model=HealthResponse)
def health(service: Text2SqlService = Depends(get_service)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=service.model_loaded,
        adapter_loaded=service.adapter_loaded,
        base_model=service.data_config.base_model,
        n_databases=len(service.schemas),
        metrics_available=service.paths.metrics_json().exists(),
    )


@app.get("/databases", response_model=list[str])
def list_databases(service: Text2SqlService = Depends(get_service)) -> list[str]:
    return service.list_databases()


@app.get("/schema/{db_id}", response_model=SchemaResponse)
def get_schema(db_id: str, service: Text2SqlService = Depends(get_service)) -> SchemaResponse:
    try:
        text = service.schema_text(db_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown database {db_id!r}") from None
    return SchemaResponse(
        db_id=db_id, schema_text=text, n_tables=service.schemas[db_id].n_tables()
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(
    request: GenerateRequest, service: Text2SqlService = Depends(get_service)
) -> GenerateResponse:
    try:
        outcome = service.generate(
            question=request.question,
            db_id=request.db_id,
            schema_text=request.schema_text,
            use_adapter=request.use_adapter,
        )
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"unknown database {request.db_id!r}"
        ) from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None

    execution = None
    if request.execute and request.db_id:
        result = service.execute(request.db_id, outcome.sql)
        execution = ExecutionPayload(
            ok=result.ok,
            row_count=len(result.rows) if result.rows is not None else None,
            # Cap the payload: a query returning thousands of rows should not be
            # serialized into an HTTP response.
            rows=[list(row) for row in result.rows[:50]] if result.rows is not None else None,
            error=result.error,
            elapsed_seconds=result.elapsed_seconds,
        )

    return GenerateResponse(
        sql=outcome.sql,
        raw_output=outcome.raw_output,
        db_id=request.db_id,
        used_adapter=outcome.used_adapter,
        generation_seconds=outcome.generation_seconds,
        execution=execution,
    )


@app.post("/compare", response_model=CompareResponse)
def compare(
    request: GenerateRequest, service: Text2SqlService = Depends(get_service)
) -> CompareResponse:
    """Base vs fine-tuned on the same input. Roughly twice the latency of /generate."""
    try:
        base, tuned = service.compare(
            question=request.question, db_id=request.db_id, schema_text=request.schema_text
        )
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"unknown database {request.db_id!r}"
        ) from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None

    return CompareResponse(
        question=request.question,
        db_id=request.db_id,
        base_sql=base.sql,
        tuned_sql=tuned.sql,
        generation_seconds=base.generation_seconds + tuned.generation_seconds,
    )


@app.get("/metrics")
def metrics(service: Text2SqlService = Depends(get_service)) -> dict:
    """The persisted results from the training run.

    404 before the run has happened -- the API never invents numbers.
    """
    payload = service.metrics()
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail="no artifacts/metrics.json; the training run has not been ingested yet",
        )
    return payload
