import asyncio
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx2 import Response

import app as app_module
from app import SimilarityRequest, app, similarity

SENTENCE = "The quick brown fox jumps over the lazy dog."
OTHER_SENTENCE = "Machine learning models turn text into numerical vectors."


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """Client backed by the real model; module scope so the model loads once."""
    with TestClient(app) as c:
        yield c


def post(client: TestClient, a: str, b: str) -> Response:
    """POST two texts to /similarity."""
    return client.post("/similarity", json={"text_a": a, "text_b": b})


# --- health & happy path ----------------------------------------------------


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_identical_texts_score_one(client: TestClient) -> None:
    r = post(client, SENTENCE, SENTENCE)
    assert r.status_code == 200
    assert r.json()["similarity"] == pytest.approx(1.0, abs=1e-4)


def test_similar_scores_higher_than_unrelated(client: TestClient) -> None:
    similar = post(client, "A cat sits on the mat.", "A kitten rests on a rug.").json()["similarity"]
    unrelated = post(client, "A cat sits on the mat.", "Quarterly tax filing deadlines.").json()["similarity"]
    assert similar > unrelated


def test_similarity_is_symmetric(client: TestClient) -> None:
    ab = post(client, "Cats love to sleep in the sun.", "Dogs enjoy playing fetch in the park.").json()["similarity"]
    ba = post(client, "Dogs enjoy playing fetch in the park.", "Cats love to sleep in the sun.").json()["similarity"]
    assert ab == pytest.approx(ba, abs=1e-4)


def test_response_shape_and_range(client: TestClient) -> None:
    body = post(client, SENTENCE, OTHER_SENTENCE).json()
    assert set(body) == {"similarity"}
    assert -1.0001 <= body["similarity"] <= 1.0001


def test_similarity_handler_returns_python_float(client: TestClient) -> None:
    # Call the handler directly: it must return a built-in float, not np.float32/np.ndarray.
    result = similarity(SimilarityRequest(text_a=SENTENCE, text_b=OTHER_SENTENCE))
    assert type(result["similarity"]) is float


def test_embed_output_contract(client: TestClient) -> None:
    embedder = app.state.embedder
    out = embedder.embed([SENTENCE, OTHER_SENTENCE])

    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.shape == (2, embedder.dim)
    assert np.isfinite(out).all()  # no NaN, no inf


def test_embed_keeps_row_order(client: TestClient) -> None:
    embedder = app.state.embedder
    ab = embedder.embed([SENTENCE, OTHER_SENTENCE])
    ba = embedder.embed([OTHER_SENTENCE, SENTENCE])

    # Batching may pad differently, so allow a tiny numeric difference.
    assert np.allclose(ab[0], ba[1], atol=1e-4)
    assert np.allclose(ab[1], ba[0], atol=1e-4)


# --- input validation -------------------------------------------------------


@pytest.mark.parametrize("field", ["text_a", "text_b"])
@pytest.mark.parametrize("blank", ["", " ", "   ", "\n", "\t", " \n\t ", "\u00a0", "\u3000"])
def test_blank_text_rejected(client: TestClient, field: str, blank: str) -> None:
    payload = {"text_a": SENTENCE, "text_b": SENTENCE, field: blank}
    assert client.post("/similarity", json=payload).status_code == 422


def test_text_with_surrounding_whitespace_accepted(client: TestClient) -> None:
    assert post(client, f"  {SENTENCE}\n", SENTENCE).status_code == 200


@pytest.mark.parametrize("field", ["text_a", "text_b"])
def test_too_long_text_rejected(client: TestClient, field: str) -> None:
    payload = {"text_a": SENTENCE, "text_b": SENTENCE, field: "x" * 5001}
    assert client.post("/similarity", json=payload).status_code == 422


def test_max_length_text_accepted(client: TestClient) -> None:
    assert post(client, "x" * 5000, SENTENCE).status_code == 200


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"text_a": SENTENCE},
        {"text_b": SENTENCE},
        {"text_a": 1, "text_b": [SENTENCE]},
        {"text_a": None, "text_b": SENTENCE},
    ],
)
def test_invalid_payload_rejected(client: TestClient, payload: dict[str, Any]) -> None:
    assert client.post("/similarity", json=payload).status_code == 422


def test_non_json_body_rejected(client: TestClient) -> None:
    r = client.post("/similarity", content="this is not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


# --- startup probe ----------------------------------------------------------


class StubEmbedder:
    """Embedder whose output is fully controlled by the test."""

    def __init__(self, output: Any, dim: int = 4) -> None:
        self.output = output
        self.dim = dim

    def embed(self, texts: list[str]) -> Any:
        return self.output


def good_output(dim: int = 4) -> npt.NDArray[np.float32]:
    """A valid probe output: one finite float32 row."""
    return np.ones((1, dim), dtype=np.float32)


def nan_output() -> npt.NDArray[np.float32]:
    out = good_output()
    out[0, 0] = np.nan
    return out


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


def test_startup_succeeds_when_probe_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = StubEmbedder(good_output())
    scratch_app = start_with(monkeypatch, stub)
    assert scratch_app.state.embedder is stub


@pytest.mark.parametrize(
    "output, match",
    [
        (nan_output(), "NaN or inf"),
        (good_output().astype(np.float64), "float32"),
        (np.ones((1, 7), dtype=np.float32), "shape"),  # dim is 4
        (np.ones((2, 4), dtype=np.float32), "shape"),  # one text in, two rows out
        (good_output().tolist(), "ndarray"),
    ],
    ids=["nan", "float64", "wrong-dim", "wrong-rows", "not-ndarray"],
)
def test_startup_refused_when_probe_fails(monkeypatch: pytest.MonkeyPatch, output: Any, match: str) -> None:
    with pytest.raises(RuntimeError, match=match):
        start_with(monkeypatch, StubEmbedder(output))


def test_startup_fails_with_bad_model_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="not found"):
        run_lifespan()


# --- non-finite output at request time --------------------------------------


class BrokenEmbedder:
    """Embedder that returns the same non-finite value everywhere."""

    dim = 768

    def __init__(self, value: float) -> None:
        self.value = value

    def embed(self, texts: list[str]) -> npt.NDArray[np.float32]:
        return np.full((len(texts), self.dim), self.value, dtype=np.float32)


@pytest.fixture
def use_embedder(client: TestClient) -> Iterator[Callable[[Any], None]]:
    """Swap the live embedder for the test; the real one is restored afterwards."""
    original = app.state.embedder

    def swap(embedder: Any) -> None:
        app.state.embedder = embedder

    yield swap
    app.state.embedder = original


@pytest.mark.parametrize("bad", [np.nan, np.inf], ids=["nan", "inf"])
def test_non_finite_output_gives_clear_500(
    client: TestClient, use_embedder: Callable[[Any], None], caplog: pytest.LogCaptureFixture, bad: float
) -> None:
    use_embedder(BrokenEmbedder(bad))

    with caplog.at_level("ERROR", logger="embedding-api"):
        r = post(client, SENTENCE, OTHER_SENTENCE)

    assert r.status_code == 500
    assert r.json() == {"detail": app_module.SIMILARITY_FAILED_DETAIL}

    # The operator can see what happened, without the private texts.
    assert "non-finite embedding" in caplog.text
    assert SENTENCE not in caplog.text


def test_non_finite_score_gives_clear_500(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "cos_sim", lambda a, b: float("nan"))
    r = post(client, SENTENCE, OTHER_SENTENCE)
    assert r.status_code == 500
    assert r.json() == {"detail": app_module.SIMILARITY_FAILED_DETAIL}


def test_service_recovers_after_failure(client: TestClient, use_embedder: Callable[[Any], None]) -> None:
    real = app.state.embedder
    use_embedder(BrokenEmbedder(np.nan))
    assert post(client, SENTENCE, OTHER_SENTENCE).status_code == 500
    use_embedder(real)
    assert post(client, SENTENCE, OTHER_SENTENCE).status_code == 200


# --- misc -------------------------------------------------------------------


def test_wrong_method_not_allowed(client: TestClient) -> None:
    assert client.get("/similarity").status_code == 405


def test_published_schema_describes_similarity(client: TestClient) -> None:
    """The public contract: route, method and field names that API users depend on."""
    r = client.get("/openapi.json")
    assert r.status_code == 200
    schema = r.json()

    assert "post" in schema["paths"]["/similarity"]
    schemas = schema["components"]["schemas"]
    assert set(schemas["SimilarityRequest"]["properties"]) == {"text_a", "text_b"}
    assert set(schemas["SimilarityResponse"]["properties"]) == {"similarity"}
