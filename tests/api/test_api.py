"""API contract tests.

A stub generator is injected, so no model is downloaded and no torch is required. What is
under test is the contract: status codes, validation, and that the request actually reaches
the generator with the right prompt.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi", reason="serve extra not installed")

from fastapi.testclient import TestClient

from text2sql_lora.api.main import app, get_service
from text2sql_lora.api.service import Text2SqlService
from text2sql_lora.config import Paths
from text2sql_lora.data.schema import parse_db_entry
from text2sql_lora.models.stub import StubGenerator

from ..conftest import MINI_TABLES_ENTRY

pytestmark = pytest.mark.unit


class FakeTokenizer:
    """Just enough tokenizer to render a prompt. No vocabulary, no download."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "\n".join(f"<{m['role']}>{m['content']}" for m in messages)


@pytest.fixture
def service(tmp_path, mini_db, monkeypatch):
    generator = StubGenerator(
        responses=["SELECT COUNT(*) FROM student"],
        base_responses=["SELECT * FROM student"],
    )

    paths = Paths(data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts")
    monkeypatch.setattr(Paths, "db_path", lambda self, db_id: mini_db)

    built = Text2SqlService(paths=paths, generator=generator, tokenizer=FakeTokenizer())
    built._schemas = {"mini_school": parse_db_entry(dict(MINI_TABLES_ENTRY))}
    return built


@pytest.fixture
def client(service):
    app.dependency_overrides[get_service] = lambda: service
    yield TestClient(app)
    app.dependency_overrides.clear()


# --- health and discovery ----------------------------------------------------


def test_health_reports_state_without_loading_a_model(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["n_databases"] == 1
    assert body["metrics_available"] is False


def test_databases_are_listed(client):
    response = client.get("/databases")
    assert response.status_code == 200
    assert response.json() == ["mini_school"]


def test_schema_endpoint_returns_ddl(client):
    response = client.get("/schema/mini_school")

    assert response.status_code == 200
    body = response.json()
    assert "CREATE TABLE student" in body["schema_text"]
    assert body["n_tables"] == 2


def test_unknown_database_is_404(client):
    assert client.get("/schema/nope").status_code == 404


# --- generation --------------------------------------------------------------


def test_generate_by_db_id(client):
    response = client.post(
        "/generate", json={"question": "How many students?", "db_id": "mini_school"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["sql"] == "SELECT COUNT(*) FROM student"
    assert body["used_adapter"] is True
    assert body["generation_seconds"] >= 0


def test_generate_with_a_raw_schema_needs_no_database(client):
    """The model was fine-tuned on serialized DDL, so any DDL works -- not just Spider's."""
    response = client.post(
        "/generate",
        json={
            "question": "How many rows?",
            "schema_text": "CREATE TABLE widget (id INTEGER, name TEXT);",
        },
    )

    assert response.status_code == 200
    assert response.json()["db_id"] is None


def test_the_question_and_schema_actually_reach_the_model(client, service):
    """Guards against a serving layer that silently drops the schema and still returns
    plausible-looking SQL."""
    client.post(
        "/generate", json={"question": "How many students?", "db_id": "mini_school"}
    )

    prompt = service._generator.seen_prompts[-1]
    assert "How many students?" in prompt
    assert "CREATE TABLE student" in prompt


def test_use_adapter_false_disables_the_adapter(client):
    response = client.post(
        "/generate",
        json={"question": "q", "db_id": "mini_school", "use_adapter": False},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["used_adapter"] is False
    # The base path must actually produce base output, not silently reuse the adapter's.
    assert body["sql"] == "SELECT * FROM student"


def test_generate_with_execution_runs_the_sql(client):
    response = client.post(
        "/generate",
        json={"question": "How many students?", "db_id": "mini_school", "execute": True},
    )

    assert response.status_code == 200
    execution = response.json()["execution"]
    assert execution["ok"] is True
    assert execution["rows"] == [[3]]


def test_unknown_database_on_generate_is_404(client):
    response = client.post("/generate", json={"question": "q", "db_id": "nope"})
    assert response.status_code == 404


# --- validation --------------------------------------------------------------


def test_missing_both_schema_sources_is_rejected(client):
    response = client.post("/generate", json={"question": "q"})
    assert response.status_code == 422


def test_empty_question_is_rejected(client):
    response = client.post("/generate", json={"question": "", "db_id": "mini_school"})
    assert response.status_code == 422


def test_execute_without_a_db_id_is_rejected(client):
    """There would be nothing to execute against."""
    response = client.post(
        "/generate",
        json={"question": "q", "schema_text": "CREATE TABLE t (a TEXT);", "execute": True},
    )
    assert response.status_code == 422


# --- compare -----------------------------------------------------------------


def test_compare_returns_both_systems(client):
    response = client.post(
        "/compare", json={"question": "How many students?", "db_id": "mini_school"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "How many students?"
    assert body["base_sql"] == "SELECT * FROM student"
    assert body["tuned_sql"] == "SELECT COUNT(*) FROM student"
    # If these were equal the endpoint would be showing the same system twice, which is
    # exactly the failure the side-by-side demo exists to avoid.
    assert body["base_sql"] != body["tuned_sql"]


# --- metrics -----------------------------------------------------------------


def test_metrics_is_404_before_the_training_run(client):
    """The API must never invent numbers when the run has not happened."""
    response = client.get("/metrics")
    assert response.status_code == 404


def test_metrics_served_from_the_artifact(client, service):
    payload = {"results": {"lora_zero_shot": {"execution_accuracy": 0.5}}}
    service.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    service.paths.metrics_json().write_text(json.dumps(payload), encoding="utf-8")

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.json() == payload
    assert client.get("/health").json()["metrics_available"] is True
