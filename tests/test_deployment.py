"""Deployment behaviour tests: startup env handling and /health."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import app
from support import StubEmbedder, good_output, run_lifespan, start_with


@pytest.mark.parametrize(
    "value, expected",
    [("", app_module.DEFAULT_MODEL_PATH), ("   ", app_module.DEFAULT_MODEL_PATH), ("/mnt/private/model", "/mnt/private/model")],
    ids=["empty-uses-default", "blank-uses-default", "set-used-as-given"],
)
def test_lifespan_when_model_path_env_given_loads_that_path(monkeypatch: pytest.MonkeyPatch, value: str, expected: str) -> None:
    # MODEL_PATH="" would make SentenceTransformer("") guess a location; use the default path instead.
    seen: list[str] = []

    def fake(path: str, **kwargs: Any) -> StubEmbedder:
        seen.append(path)
        return StubEmbedder(good_output())

    monkeypatch.setenv("MODEL_PATH", value)
    monkeypatch.setattr(app_module, "InterpretableEmbedder", fake)
    run_lifespan()
    assert seen == [expected]


def test_lifespan_when_topics_path_set_passes_it_to_load_topics(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    monkeypatch.setenv("TOPICS_PATH", "/mnt/private/topics.csv")
    monkeypatch.setattr(app_module, "InterpretableEmbedder", lambda *a, **k: StubEmbedder(good_output()))
    monkeypatch.setattr(app_module, "load_topics", lambda path: seen.append(path) or [])
    run_lifespan()
    assert seen == ["/mnt/private/topics.csv"]


@pytest.fixture
def client_without_embedder(client: TestClient) -> Iterator[TestClient]:
    """Temporarily remove the loaded embedder, as if startup had not finished."""
    saved = app.state.embedder
    del app.state.embedder
    yield client
    app.state.embedder = saved


def test_health_endpoint_when_no_model_loaded_returns_503(client_without_embedder: TestClient) -> None:
    # A service that cannot serve requests must not look healthy to the load balancer.
    assert client_without_embedder.get("/health").status_code == 503


def test_health_endpoint_when_topics_not_loaded_returns_503(client: TestClient) -> None:
    # /similarity/explain cannot work without topics, so the service is not ready.
    saved = app.state.topics
    del app.state.topics
    try:
        assert client.get("/health").status_code == 503
    finally:
        app.state.topics = saved


@pytest.mark.parametrize("kind", ["directory", "blank"])
def test_lifespan_when_topics_path_unusable_raises_os_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str) -> None:
    # A directory or empty TOPICS_PATH must stop startup with an OSError, not start with no topics.
    monkeypatch.setenv("TOPICS_PATH", str(tmp_path) if kind == "directory" else "")
    with pytest.raises(OSError):
        start_with(monkeypatch, StubEmbedder(good_output()))


def test_lifespan_when_topics_path_unset_uses_default_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOPICS_PATH", raising=False)
    seen: list[str] = []
    monkeypatch.setattr(app_module, "load_topics", lambda path: seen.append(path) or [])
    start_with(monkeypatch, StubEmbedder(good_output()))
    assert seen == [app_module.DEFAULT_TOPICS_PATH]
