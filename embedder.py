import hashlib

import numpy as np


class DummyEmbedder:
    """Stand-in for the real model: same text always gives the same vector."""
    dim = 8

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = []
        for t in texts:
            seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:4], "little")
            rows.append(np.random.default_rng(seed).standard_normal(self.dim))
        return np.asarray(rows, dtype=np.float32)