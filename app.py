from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from embedder import DummyEmbedder


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.embedder = DummyEmbedder()   # later: the real model, loaded once here
    yield


app = FastAPI(lifespan=lifespan)


class SimilarityRequest(BaseModel):
    text_a: str = Field(min_length=1, max_length=5000)
    text_b: str = Field(min_length=1, max_length=5000)


class SimilarityResponse(BaseModel):
    similarity: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/similarity", response_model=SimilarityResponse)
def similarity(req: SimilarityRequest):
    a, b = app.state.embedder.embed([req.text_a, req.text_b])
    score = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return {"similarity": score}