import logging
import math
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, NoReturn

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field, StrictBool, field_validator, model_validator
from sentence_transformers.util import cos_sim
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from embedder import Embedder, InterpretableEmbedder
from explain import load_topics, pick_topics, topic_shares

DEFAULT_MODEL_PATH: str = "models/model_interpretable"
DEFAULT_TOPICS_PATH: str = "models/topic_mapping_summary.csv"
EMBEDDING_DIM: int = 768  
MAX_TEXT_CHARS: int = 4000  # about 1000 English words; the model reads at most 1024 tokens (see embedder.py)
MAX_TOPICS_PER_LIST: int = 10

logger = logging.getLogger("embedding-api")

PROBE_TEXT: str = "Startup probe sentence."
SIMILARITY_FAILED_DETAIL: str = "Similarity could not be computed for these texts."


def probe_embedder(embedder: Embedder) -> None:
    """Embed one sentence and verify the output contract.

    Raises:
        RuntimeError: if the output is not a finite float32 ndarray of shape (1, embedder.dim).
    """
    out = embedder.embed([PROBE_TEXT])
    if not isinstance(out, np.ndarray):
        raise RuntimeError(f"probe: expected np.ndarray, got {type(out).__name__}")
    if out.dtype != np.float32:
        raise RuntimeError(f"probe: expected float32, got {out.dtype}")
    if out.shape != (1, embedder.dim):
        raise RuntimeError(f"probe: expected shape (1, {embedder.dim}), got {out.shape}")
    if not np.isfinite(out).all():
        raise RuntimeError("probe: embedding contains NaN or inf")


def _parse_bool_env(name: str, default: bool) -> bool:
    """Read an env var that must be "true" or "false" (any case); unset or empty gives `default`.

    Raises:
        ValueError: for any other value, so a typo stops startup instead of silently meaning false.
    """
    raw = os.environ.get(name, "")
    value = raw.strip().lower()
    if not value:
        return default
    if value not in ("true", "false"):
        raise ValueError(f"{name} must be 'true' or 'false', got {raw!r}")
    return value == "true"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model and probe it; a broken model stops the service from starting."""
    model_path = os.environ.get("MODEL_PATH", "").strip() or DEFAULT_MODEL_PATH  # read at startup, not import
    embedder = InterpretableEmbedder(model_path, trust_remote_code=True)
    if embedder.dim != EMBEDDING_DIM:
        raise RuntimeError(f"model reports {embedder.dim} dimensions, expected {EMBEDDING_DIM}: wrong model?")
    probe_embedder(embedder)
    app.state.embedder = embedder
    app.state.topics = load_topics(os.environ.get("TOPICS_PATH", DEFAULT_TOPICS_PATH))
    # Topic summaries are available to callers that ask for them; set EXPOSE_SUMMARIES=false to withhold them.
    app.state.expose_summaries = _parse_bool_env("EXPOSE_SUMMARIES", default=True)
    yield


app = FastAPI(lifespan=lifespan)

MAX_BODY_BYTES: int = 128 * 1024  # two max-length texts need at most ~48 KB even when fully JSON-escaped


class BodySizeLimitMiddleware:
    """Answer 413 for request bodies larger than MAX_BODY_BYTES, without reading the rest into memory.

    A declared Content-Length is checked up front. For chunked uploads the bytes are counted as they
    arrive; once over the limit the body is cut short and whatever the app answers is replaced by the 413.
    """

    TOO_LARGE_BODY: bytes = b'{"detail":"Request body too large."}'

    def __init__(self, app: ASGIApp) -> None:
        self.asgi_app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.asgi_app(scope, receive, send)
            return

        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
            await JSONResponse(status_code=413, content={"detail": "Request body too large."})(scope, receive, send)
            return

        received = 0
        over_limit = False
        body_sent = False

        async def counting_receive() -> Message:
            nonlocal received, over_limit
            if over_limit:
                return {"type": "http.request", "body": b"", "more_body": False}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > MAX_BODY_BYTES:
                    over_limit = True
                    return {"type": "http.request", "body": b"", "more_body": False}
            return message

        async def replacing_send(message: Message) -> None:
            nonlocal body_sent
            if not over_limit:
                await send(message)
            elif message["type"] == "http.response.start":
                headers = [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(self.TOO_LARGE_BODY)).encode()),
                ]
                await send({"type": "http.response.start", "status": 413, "headers": headers})
            elif message["type"] == "http.response.body" and not body_sent:
                body_sent = True
                await send({"type": "http.response.body", "body": self.TOO_LARGE_BODY})

        await self.asgi_app(scope, counting_receive, replacing_send)


app.add_middleware(BodySizeLimitMiddleware)


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return the same generic JSON error as handled failures; log the type only (messages may hold texts)."""
    logger.error("unhandled %s on %s", type(exc).__name__, request.url.path)
    return JSONResponse(status_code=500, content={"detail": SIMILARITY_FAILED_DETAIL})


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return 422 with where and why a field failed, but never the submitted values (they may be private)."""
    errors = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


class SimilarityRequest(BaseModel):
    """Two non-blank texts of 1 to MAX_TEXT_CHARS characters each."""

    text_a: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    text_b: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)

    @field_validator("text_a", "text_b")
    @classmethod
    def not_blank(cls, v: str) -> str:
        """Reject whitespace-only text and text that is not valid Unicode (the text is not modified)."""
        if not v.strip():
            raise ValueError("text must not be blank")
        try:
            v.encode("utf-8")  # fails for lone surrogates such as the JSON escape \ud83d
        except UnicodeEncodeError:
            raise ValueError("text must be valid Unicode") from None
        return v


class SimilarityResponse(BaseModel):
    """Cosine similarity of the two texts, in [-1, 1], and whether the model cut off the end of either text."""

    similarity: float
    text_a_truncated: bool
    text_b_truncated: bool


class ExplainRequest(SimilarityRequest):
    """Two texts plus how many shared and differing topics to report."""

    top_shared: int = Field(default=2, ge=0, le=MAX_TOPICS_PER_LIST)
    top_differing: int = Field(default=2, ge=0, le=MAX_TOPICS_PER_LIST)
    include_summary: StrictBool = False  # ignored if the server sets EXPOSE_SUMMARIES=false

    @model_validator(mode="after")
    def asks_for_something(self) -> "ExplainRequest":
        """Reject a request that asks for no topics at all."""
        if self.top_shared == 0 and self.top_differing == 0:
            raise ValueError("top_shared and top_differing must not both be 0")
        return self


class TopicItem(BaseModel):
    """One topic and how much of the topic-level signal it accounts for."""

    topic: str
    contribution: float = Field(
        description="Fraction of the topic-level signal from this topic: positive if the texts share it, negative if "
        "they differ on it. The absolute values over all 300 topics sum to 1. Not a similarity score."
    )
    summary: str | None = None


class ExplainResponse(BaseModel):
    """Cosine similarity plus the topics that drive it. Lists can be shorter than requested or empty."""

    similarity: float
    text_a_truncated: bool
    text_b_truncated: bool
    shared_topics: list[TopicItem]
    differing_topics: list[TopicItem]


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Send visitors of the bare address to the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict[str, str]:
    """Return ok when the model and topics are loaded, otherwise 503."""
    if not (hasattr(app.state, "embedder") and hasattr(app.state, "topics")):
        raise HTTPException(status_code=503, detail="Service is not ready.")
    return {"status": "ok"}


@app.post("/similarity", response_model=SimilarityResponse)
def similarity(req: SimilarityRequest) -> dict[str, Any]:
    """Return the cosine similarity of the two texts and whether either was cut off by the model's token limit.

    \f
    Raises:
        HTTPException: 500 with a generic message if the model produces NaN or inf.
    """
    _, score, (a_cut, b_cut) = _embed_pair(req)
    return {"similarity": score, "text_a_truncated": a_cut, "text_b_truncated": b_cut}


@app.post("/similarity/explain", response_model=ExplainResponse, response_model_exclude_none=True)
def explain(req: ExplainRequest) -> dict[str, Any]:
    """Return the cosine similarity and the topics that are shared or differ between the texts.

    \f
    Raises:
        HTTPException: 500 with a generic message if the model produces NaN or inf.
    """
    embeddings, score, (a_cut, b_cut) = _embed_pair(req)
    try:
        shares = topic_shares(embeddings[0], embeddings[1])
    except ValueError:
        _fail_similarity(req, "embedding smaller than the topic range")
    shared, differing = pick_topics(shares, req.top_shared, req.top_differing)

    with_summary = req.include_summary and app.state.expose_summaries

    def items(indices: list[int]) -> list[dict[str, Any]]:
        out = []
        for i in indices:
            topic = app.state.topics[i]
            item: dict[str, Any] = {"topic": topic.label, "contribution": float(shares[i])}
            if with_summary:
                item["summary"] = topic.summary
            out.append(item)
        return out

    return {
        "similarity": score,
        "text_a_truncated": a_cut,
        "text_b_truncated": b_cut,
        "shared_topics": items(shared),
        "differing_topics": items(differing),
    }


def _embed_pair(req: SimilarityRequest) -> tuple[np.ndarray, float, list[bool]]:
    """Embed both texts and return the embeddings, their cosine similarity and which texts were truncated.

    Raises:
        HTTPException: 500 with a generic message if anything is non-finite.
    """
    embeddings = app.state.embedder.embed([req.text_a, req.text_b])
    expected_shape = (2, EMBEDDING_DIM)
    if not isinstance(embeddings, np.ndarray) or embeddings.shape != expected_shape:
        _fail_similarity(req, f"unexpected embedding shape, expected {expected_shape}")
    if not np.isfinite(embeddings).all():
        _fail_similarity(req, "non-finite embedding")

    score = float(cos_sim(embeddings[0], embeddings[1]))
    if not math.isfinite(score):
        _fail_similarity(req, "non-finite score")
    truncated = app.state.embedder.truncated([req.text_a, req.text_b])
    return embeddings, max(-1.0, min(1.0, score)), truncated  # float rounding can overshoot by a few ulps


def _fail_similarity(req: SimilarityRequest, reason: str) -> NoReturn:
    """Log the details (lengths only, the texts may be private), then raise a generic 500."""
    logger.error("similarity failed: %s (len_a=%d, len_b=%d)", reason, len(req.text_a), len(req.text_b))
    raise HTTPException(status_code=500, detail=SIMILARITY_FAILED_DETAIL)
