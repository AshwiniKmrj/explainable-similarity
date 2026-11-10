import hashlib
from typing import Protocol

import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer


class Embedder(Protocol):
    """Anything that can turn texts into a float32 matrix of shape (len(texts), dim)."""

    dim: int

    def embed(self, texts: list[str]) -> npt.NDArray[np.float32]: ...


class DummyEmbedder:
    """Stand-in for the real model: same text always gives the same vector."""

    dim: int = 8

    def embed(self, texts: list[str]) -> npt.NDArray[np.float32]:
        rows = []
        for t in texts:
            seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:4], "little")
            rows.append(np.random.default_rng(seed).standard_normal(self.dim))
        return np.asarray(rows, dtype=np.float32)


class InterpretableEmbedder:
    """Wraps the real SentenceTransformer model, always running on CPU."""

    def __init__(self, file_path: str, trust_remote_code: bool = False) -> None:
        self.model = SentenceTransformer(file_path, trust_remote_code=trust_remote_code, device="cpu")
        dim = self.model.get_sentence_embedding_dimension()
        if dim is None:
            raise RuntimeError("model does not report an embedding dimension")
        self.dim: int = dim
        self.model.max_seq_length = 512

    def embed(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Encode texts into an array of shape (len(texts), dim)."""
        return self.model.encode(texts)
