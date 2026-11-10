import logging
import math
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import NoReturn

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from sentence_transformers.util import cos_sim

from embedder import Embedder, InterpretableEmbedder

DEFAULT_MODEL_PATH: str = "models/model_interpretable"

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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model and probe it; a broken model stops the service from starting."""
    model_path = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)  # read at startup, not import
    embedder = InterpretableEmbedder(model_path, trust_remote_code=True)
    probe_embedder(embedder)
    app.state.embedder = embedder
    yield


app = FastAPI(lifespan=lifespan)


class SimilarityRequest(BaseModel):
    """Two non-blank texts of 1 to 5000 characters each."""

    text_a: str = Field(min_length=1, max_length=5000)
    text_b: str = Field(min_length=1, max_length=5000)

    @field_validator("text_a", "text_b")
    @classmethod
    def not_blank(cls, v: str) -> str:
        """Reject whitespace-only text (the text itself is not modified)."""
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


class SimilarityResponse(BaseModel):
    """Cosine similarity of the two texts, in [-1, 1]."""

    similarity: float


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@app.post("/similarity", response_model=SimilarityResponse)
def similarity(req: SimilarityRequest) -> dict[str, float]:
    """Return the cosine similarity of the two texts.

    Raises:
        HTTPException: 500 with a generic message if the model produces NaN or inf.
    """
    embeddings = app.state.embedder.embed([req.text_a, req.text_b])
    if not np.isfinite(embeddings).all():
        _fail_similarity(req, "non-finite embedding")

    score = float(cos_sim(embeddings[0], embeddings[1]))
    if not math.isfinite(score):
        _fail_similarity(req, "non-finite score")

    return {"similarity": score}


def _fail_similarity(req: SimilarityRequest, reason: str) -> NoReturn:
    """Log the details (lengths only, the texts may be private), then raise a generic 500."""
    logger.error("similarity failed: %s (len_a=%d, len_b=%d)", reason, len(req.text_a), len(req.text_b))
    raise HTTPException(status_code=500, detail=SIMILARITY_FAILED_DETAIL)
