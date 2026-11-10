"""Shared pytest fixtures."""

import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import DEFAULT_MODEL_PATH, app

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session", autouse=True)
def isolated_environment() -> Iterator[None]:
    """Run from the project root (the app's default paths are relative) and ignore ambient settings."""
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(ROOT)
        mp.delenv("TOPICS_PATH", raising=False)
        mp.delenv("EXPOSE_SUMMARIES", raising=False)
        yield


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    """Client backed by the real model; session scope so the model loads once for the whole run.

    A missing model or topic file fails with a clear message (it is never skipped).
    """
    model_path = Path(os.environ.get("MODEL_PATH", "").strip() or ROOT / DEFAULT_MODEL_PATH)
    if not model_path.exists():
        pytest.fail(f"model not found at {model_path}")
    with TestClient(app) as c:
        yield c


@pytest.fixture
def use_embedder(client: TestClient) -> Iterator[Callable[[Any], None]]:
    """Swap the live embedder for the test; the real one is restored afterwards."""
    original = app.state.embedder

    def swap(embedder: Any) -> None:
        app.state.embedder = embedder

    yield swap
    app.state.embedder = original


@pytest.fixture
def set_state(client: TestClient) -> Iterator[Callable[[str, Any], None]]:
    """Set an attribute on app.state and restore it afterwards (monkeypatch would delete it instead)."""
    saved: dict[str, Any] = {}

    def setter(name: str, value: Any) -> None:
        saved.setdefault(name, getattr(app.state, name))
        setattr(app.state, name, value)

    yield setter
    for name, value in saved.items():
        setattr(app.state, name, value)


@pytest.fixture
def real_topics_path() -> Path:
    """Path of the shipped topic CSV; a missing file fails the test (it is never skipped)."""
    path = ROOT / "models" / "topic_mapping_summary.csv"
    if not path.exists():
        pytest.fail(f"topic file not found at {path}")
    return path


@pytest.fixture
def quiet_client(client: TestClient) -> TestClient:
    """Client that returns the 500 response instead of re-raising server errors."""
    return TestClient(app, raise_server_exceptions=False)
