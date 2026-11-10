"""Topic-level explanations of an embedding comparison.

The first N_TOPICS dimensions of the embedding each stand for one topic. For two embeddings we
multiply them dimension by dimension and express each product as a signed share of the total
absolute signal over the topic dimensions. Large positive shares are shared topics, large
negative shares are differing topics.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

N_TOPICS: int = 300


@dataclass(frozen=True)
class Topic:
    """A topic label and its longer description."""

    label: str
    summary: str


def load_topics(path: str | Path) -> list[Topic]:
    """Load the topic table; the result is indexed by embedding dimension.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file does not cover exactly dimensions 0..N_TOPICS-1, or a label is blank.
    """
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except csv.Error as exc:  # for example a field larger than the csv module's limit
        raise ValueError(f"{path}: not a readable topic CSV ({exc})") from exc

    try:
        rows.sort(key=lambda r: int(r["dimension"]))
        dims = [int(r["dimension"]) for r in rows]
    except (KeyError, ValueError, TypeError) as exc:  # TypeError: a short row leaves the value None
        raise ValueError(f"{path}: needs an integer 'dimension' column") from exc

    if dims != list(range(N_TOPICS)):
        raise ValueError(f"{path}: expected dimensions 0..{N_TOPICS - 1} exactly once each, got {len(dims)} rows")

    topics = []
    for row in rows:
        label = (row.get("Topic_label") or "").strip()
        if not label:
            raise ValueError(f"{path}: blank Topic_label for dimension {row['dimension']}")
        topics.append(Topic(label=label, summary=(row.get("summary") or "").strip()))
    return topics


def topic_shares(a: npt.NDArray[np.float32], b: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
    """Signed share of each topic dimension in the topic-level signal.

    Returns an array of length N_TOPICS whose absolute values sum to 1 (all zeros if the
    topic dimensions carry no signal).

    Raises:
        ValueError: if an embedding has fewer than N_TOPICS dimensions.
    """
    if a.shape[-1] < N_TOPICS or b.shape[-1] < N_TOPICS:
        raise ValueError(f"embeddings need at least {N_TOPICS} dimensions")
    products = a[:N_TOPICS].astype(np.float64) * b[:N_TOPICS].astype(np.float64)
    total = np.abs(products).sum()
    if total == 0.0:
        return np.zeros(N_TOPICS)
    return products / total


def pick_topics(shares: npt.NDArray[np.float64], k_shared: int, k_differing: int) -> tuple[list[int], list[int]]:
    """Indices of up to k_shared shared and k_differing differing topics, strongest first.

    There is no minimum size: weak topics are returned when stronger ones run out. A list is
    shorter than requested (or empty) when too few topics have the right sign, for example no
    differing topics for two identical texts.
    """
    order = np.argsort(shares)
    shared = [int(i) for i in order[::-1][:k_shared] if shares[i] > 0]
    differing = [int(i) for i in order[:k_differing] if shares[i] < 0]
    return shared, differing
