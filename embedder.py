from typing import Protocol

import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer


# Commit of Alibaba-NLP/new-impl whose modeling.py is downloaded and executed when the model loads. Pinned so a
# change to that repository cannot alter the code this service runs; review the diff before bumping it.
REMOTE_CODE_REVISION: str = "40ced75c3017eb27626c9d4ea981bde21a2662f4"


class Embedder(Protocol):
    """Anything that can turn texts into a float32 matrix of shape (len(texts), dim)."""

    dim: int

    def embed(self, texts: list[str]) -> npt.NDArray[np.float32]: ...

    def truncated(self, texts: list[str]) -> list[bool]: ...


class InterpretableEmbedder:
    """Wraps the real SentenceTransformer model, always running on CPU."""

    def __init__(self, file_path: str, trust_remote_code: bool = False) -> None:
        self.model = SentenceTransformer(
            file_path,
            trust_remote_code=trust_remote_code,
            device="cpu",
            model_kwargs={"code_revision": REMOTE_CODE_REVISION},
        )
        dim = self.model.get_sentence_embedding_dimension()
        if dim is None:
            raise RuntimeError("model does not report an embedding dimension")
        self.dim: int = dim
        # Memory and latency grow with the token count (measured at roughly 0.9 MB per token for a pair of texts
        # on CPU), so inputs are capped well below the 8192 tokens the model supports. 1024 tokens is about
        # 700 English words and keeps a request within a 2 GB container.
        self.model.max_seq_length = 1024

    def embed(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Encode texts into an array of shape (len(texts), dim)."""
        return self.model.encode(texts)

    def truncated(self, texts: list[str]) -> list[bool]:
        """For each text, whether the model's token limit cuts part of it off (special tokens count)."""
        ids = self.model.tokenizer(texts, add_special_tokens=True, truncation=False)["input_ids"]
        return [len(i) > self.model.max_seq_length for i in ids]
