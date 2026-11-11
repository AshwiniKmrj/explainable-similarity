"""Plain helpers shared by the test files (fixtures live in conftest.py)."""

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from fastapi import FastAPI

from explain import N_TOPICS

import app as app_module

TEST_API_KEY = "test-key-0123456789abcdef"
AUTH_HEADERS = {"X-API-Key": TEST_API_KEY}


class StubEmbedder:
    """Embedder whose output is fully controlled by the test."""

    def __init__(self, output: Any, dim: int = app_module.EMBEDDING_DIM, truncated: tuple[bool, bool] = (False, False)) -> None:
        self.output = output
        self.dim = dim
        self.flags = truncated

    def embed(self, texts: list[str]) -> Any:
        return self.output

    def truncated(self, texts: list[str]) -> list[bool]:
        return list(self.flags)


def good_output(dim: int = app_module.EMBEDDING_DIM) -> npt.NDArray[np.float32]:
    """A valid probe output: one finite float32 row."""
    return np.ones((1, dim), dtype=np.float32)


def run_lifespan() -> FastAPI:
    """Run the app's startup and shutdown on a throwaway FastAPI instance.

    The shared `app` (and the real model in `app.state.embedder`) is never touched.
    Returns the throwaway app so tests can inspect its state.
    """
    scratch_app = FastAPI()

    async def run() -> None:
        async with app_module.lifespan(scratch_app):
            pass

    asyncio.run(run())
    return scratch_app


def start_with(monkeypatch: pytest.MonkeyPatch, embedder: StubEmbedder) -> FastAPI:
    """Run the lifespan with `embedder` standing in for the real model."""
    monkeypatch.setattr(app_module, "InterpretableEmbedder", lambda *a, **k: embedder)
    return run_lifespan()


def write_topics(
    path: Path,
    *,
    dims: list[int] | None = None,
    header: str = ",dimension,Topic_label,summary",
    fmt: str = '{n},{d},Topic {d},"Summary of {d}."',
    override: dict[int, str] | None = None,
) -> Path:
    """Write a topic CSV; by default a valid one.

    `dims` is the dimension of each row in file order, `fmt` renders a row from its position `n`
    and dimension `d`, and `override` replaces whole raw lines by row number.
    """
    lines = [header]
    for n, d in enumerate(dims if dims is not None else range(N_TOPICS)):
        lines.append((override or {}).get(n, fmt.format(n=n, d=d)))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def fixed_embedder(a: dict[int, float], b: dict[int, float]) -> StubEmbedder:
    """Embedder returning the same two rows whatever the text; each row maps dimension -> value."""
    rows = np.zeros((2, app_module.EMBEDDING_DIM), dtype=np.float32)
    for i, x in a.items():
        rows[0, i] = x
    for i, x in b.items():
        rows[1, i] = x
    return StubEmbedder(rows)


def constant_embedder(value: float) -> StubEmbedder:
    """Embedder whose every output value is `value` (use nan or inf for broken output)."""
    return StubEmbedder(np.full((2, app_module.EMBEDDING_DIM), value, dtype=np.float32))


class RaisingEmbedder:
    """Embedder whose exception message contains the input text and a file path."""

    dim = app_module.EMBEDDING_DIM

    def embed(self, texts: list[str]) -> npt.NDArray[np.float32]:
        raise RuntimeError(f"cannot read C:\\secret\\weights.bin for {texts[0]}")
