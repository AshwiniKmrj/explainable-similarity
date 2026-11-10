import csv
from pathlib import Path

import numpy as np
import pytest

from app import EMBEDDING_DIM
from explain import N_TOPICS, load_topics, pick_topics, topic_shares
from support import write_topics

DIM = EMBEDDING_DIM


def vec(values: dict[int, float], dim: int = DIM) -> np.ndarray:
    """A float32 vector of zeros with the given dimensions set."""
    v = np.zeros(dim, dtype=np.float32)
    for i, x in values.items():
        v[i] = x
    return v


# --- topic_shares -----------------------------------------------------------


def test_topic_shares_when_products_differ_in_sign_returns_signed_fractions() -> None:
    a = vec({3: 2.0, 5: 1.0})
    b = vec({3: 2.0, 5: -1.0})  # products: +4 and -1
    shares = topic_shares(a, b)

    assert shares[3] == pytest.approx(0.8)
    assert shares[5] == pytest.approx(-0.2)


def test_topic_shares_when_signal_beyond_topics_ignores_it() -> None:
    a = vec({3: 1.0, N_TOPICS: 100.0})
    b = vec({3: 1.0, N_TOPICS: 100.0})
    assert topic_shares(a, b)[3] == pytest.approx(1.0)


def test_topic_shares_when_signal_on_last_topic_dimension_includes_it() -> None:
    # Dimension 299 is the last topic; 300 is the first one that must be ignored.
    a = vec({N_TOPICS - 1: 1.0, N_TOPICS: 1.0})
    shares = topic_shares(a, a)
    assert shares[N_TOPICS - 1] == pytest.approx(1.0)


def test_topic_shares_when_arguments_swapped_returns_same_shares() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal(DIM).astype(np.float32), rng.standard_normal(DIM).astype(np.float32)
    assert np.allclose(topic_shares(a, b), topic_shares(b, a))


def test_topic_shares_when_no_overlap_returns_all_zeros() -> None:
    shares = topic_shares(vec({}), vec({3: 1.0}))
    assert shares.shape == (N_TOPICS,)
    assert not shares.any()


def test_topic_shares_when_embedding_too_short_raises_value_error() -> None:
    with pytest.raises(ValueError, match="at least"):
        topic_shares(np.ones(4, dtype=np.float32), np.ones(4, dtype=np.float32))


# --- pick_topics ------------------------------------------------------------


def test_pick_topics_when_several_topics_returns_strongest_first() -> None:
    shares = np.zeros(N_TOPICS)
    shares[[1, 2, 3]] = [0.2, 0.5, 0.3]
    shares[[10, 11]] = [-0.4, -0.1]
    shared, differing = pick_topics(shares, 2, 2)
    assert shared == [2, 3]
    assert differing == [10, 11]


def test_pick_topics_when_few_topics_have_signal_fills_the_list_with_the_weak_ones() -> None:
    # No minimum size: tiny contributions are returned, after the stronger ones.
    shares = np.zeros(N_TOPICS)
    shares[[0, 1]] = [0.5, 1e-9]
    shares[[2, 3]] = [-0.4, -1e-9]
    assert pick_topics(shares, 5, 5) == ([0, 1], [2, 3])


def test_pick_topics_when_no_topic_is_negative_returns_no_differing_topics() -> None:
    # Like two identical texts: every topic agrees, so nothing is called "differing".
    shares = np.full(N_TOPICS, 1 / N_TOPICS)
    shared, differing = pick_topics(shares, 2, 2)
    assert len(shared) == 2
    assert differing == []


@pytest.mark.parametrize("k_shared, k_differing, expected", [(0, 1, ([], [1])), (1, 0, ([0], []))], ids=["no-shared", "no-differing"])
def test_pick_topics_when_k_is_zero_returns_empty_list_for_that_side(k_shared: int, k_differing: int, expected: tuple[list[int], list[int]]) -> None:
    shares = np.zeros(N_TOPICS)
    shares[0], shares[1] = 0.6, -0.4
    assert pick_topics(shares, k_shared, k_differing) == expected


# --- load_topics ------------------------------------------------------------


def test_load_topics_when_rows_unsorted_returns_list_indexed_by_dimension(tmp_path: Path) -> None:
    dims = list(range(N_TOPICS))[::-1]  # rows deliberately out of order
    topics = load_topics(write_topics(tmp_path / "t.csv", dims=dims))
    assert len(topics) == N_TOPICS
    assert topics[0].label == "Topic 0"
    assert topics[299].summary == "Summary of 299."


def test_load_topics_when_label_and_summary_padded_strips_whitespace(tmp_path: Path) -> None:
    path = write_topics(tmp_path / "t.csv", fmt='{n},{d},"  Label {d}  ","  Summary {d}  "')
    topic = load_topics(path)[7]
    assert topic.label == "Label 7"
    assert topic.summary == "Summary 7"


def test_load_topics_when_summary_column_missing_returns_empty_summaries(tmp_path: Path) -> None:
    path = write_topics(tmp_path / "t.csv", header=",dimension,Topic_label", fmt="{n},{d},Label {d}")
    assert load_topics(path)[7].summary == ""


def test_load_topics_when_file_missing_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_topics(tmp_path / "missing.csv")


@pytest.mark.parametrize(
    "dims",
    [list(range(299)), list(range(300)) + [300], [0] + list(range(299)), list(range(1, 301))],
    ids=["too-few", "too-many", "duplicate", "wrong-range"],
)
def test_load_topics_when_dimensions_incomplete_raises_value_error(tmp_path: Path, dims: list[int]) -> None:
    with pytest.raises(ValueError, match="expected dimensions"):
        load_topics(write_topics(tmp_path / "t.csv", dims=dims))


def test_load_topics_when_dimension_column_missing_raises_value_error(tmp_path: Path) -> None:
    path = tmp_path / "t.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dimension"):
        load_topics(path)


def test_load_topics_when_label_blank_raises_value_error(tmp_path: Path) -> None:
    path = write_topics(tmp_path / "t.csv", override={7: "7,7, ,s"})
    with pytest.raises(ValueError, match="blank Topic_label"):
        load_topics(path)


def test_load_topics_when_real_file_returns_one_labelled_topic_per_dimension(real_topics_path: Path) -> None:
    topics = load_topics(real_topics_path)
    assert len(topics) == N_TOPICS
    assert all(t.label and t.summary for t in topics)


# --- load_topics with malformed files ---------------------------------------


def test_load_topics_when_row_short_raises_value_error(tmp_path: Path) -> None:
    # A truncated row must raise a clear ValueError about the file, not a bare TypeError.
    path = write_topics(tmp_path / "t.csv", override={7: "7"})
    with pytest.raises(ValueError, match="dimension"):
        load_topics(path)


def test_load_topics_when_field_oversized_raises_value_error(tmp_path: Path) -> None:
    # A cell beyond the csv module's limit must stop startup with a clear config error, not csv.Error.
    path = write_topics(tmp_path / "t.csv", override={2: f"2,2,Label,{'x' * (csv.field_size_limit() + 10)}"})
    with pytest.raises(ValueError, match="not a readable topic CSV"):
        load_topics(path)


def test_load_topics_when_file_not_utf8_raises_unicode_decode_error(tmp_path: Path) -> None:
    # Encoding oddities must fail loudly rather than load mojibake labels.
    path = tmp_path / "t.csv"
    path.write_bytes(b",dimension,Topic_label,summary\n0,0,\xff\xfe\x00bad,s\n")
    with pytest.raises(UnicodeDecodeError):
        load_topics(path)
