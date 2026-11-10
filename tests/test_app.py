from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from fastapi.testclient import TestClient
from httpx2 import Response

import app as app_module
from app import EMBEDDING_DIM, app, probe_embedder
from explain import N_TOPICS
from support import StubEmbedder, constant_embedder, fixed_embedder, good_output, start_with

SENTENCE = "The quick brown fox jumps over the lazy dog."
OTHER_SENTENCE = "Machine learning models turn text into numerical vectors."


def post_similarity(client: TestClient, a: str, b: str) -> Response:
    """POST two texts to /similarity."""
    return client.post("/similarity", json={"text_a": a, "text_b": b})


# --- health & happy path ----------------------------------------------------


def test_health_endpoint_when_service_is_up_returns_ok(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_root_endpoint_when_visited_redirects_to_docs(client: TestClient) -> None:
    r = client.get("/", follow_redirects=False)
    assert (r.status_code, r.headers["location"]) == (307, "/docs")


def test_similarity_endpoint_when_texts_related_scores_above_unrelated(client: TestClient) -> None:
    similar = post_similarity(client, "A cat sits on the mat.", "A kitten rests on a rug.").json()["similarity"]
    unrelated = post_similarity(client, "A cat sits on the mat.", "Quarterly tax filing deadlines.").json()["similarity"]
    assert similar > unrelated


def test_similarity_endpoint_when_valid_returns_only_public_keys(client: TestClient) -> None:
    assert set(post_similarity(client, SENTENCE, OTHER_SENTENCE).json()) == {"similarity", "text_a_truncated", "text_b_truncated"}


# --- input validation -------------------------------------------------------


@pytest.mark.parametrize("field", ["text_a", "text_b"])
@pytest.mark.parametrize("blank", ["", " ", "   ", "\n", "\t", " \n\t ", "\u00a0", "\u3000"])
def test_similarity_endpoint_when_text_is_blank_returns_422(client: TestClient, field: str, blank: str) -> None:
    payload = {"text_a": SENTENCE, "text_b": SENTENCE, field: blank}
    assert client.post("/similarity", json=payload).status_code == 422


def test_similarity_endpoint_when_text_has_surrounding_whitespace_returns_200(client: TestClient) -> None:
    assert post_similarity(client, f"  {SENTENCE}\n", SENTENCE).status_code == 200


@pytest.mark.parametrize("field", ["text_a", "text_b"])
def test_similarity_endpoint_when_text_over_max_chars_returns_422(client: TestClient, field: str) -> None:
    payload = {"text_a": SENTENCE, "text_b": SENTENCE, field: "x" * (app_module.MAX_TEXT_CHARS + 1)}
    assert client.post("/similarity", json=payload).status_code == 422


def test_similarity_endpoint_when_text_is_max_chars_returns_200(client: TestClient) -> None:
    assert post_similarity(client, "x" * app_module.MAX_TEXT_CHARS, SENTENCE).status_code == 200


@pytest.mark.parametrize("endpoint", ["/similarity", "/similarity/explain"])
@pytest.mark.parametrize(
    "payload",
    [{}, {"text_a": SENTENCE}, {"text_b": SENTENCE}],
    ids=["both-missing", "text_b-missing", "text_a-missing"],
)
def test_similarity_and_explain_endpoints_when_field_missing_return_422(client: TestClient, endpoint: str, payload: dict[str, Any]) -> None:
    assert client.post(endpoint, json=payload).status_code == 422


@pytest.mark.parametrize("endpoint", ["/similarity", "/similarity/explain"])
@pytest.mark.parametrize("field", ["text_a", "text_b"])
@pytest.mark.parametrize(
    "value",
    [1, 1.5, True, None, [], [SENTENCE], {}, {"a": 1}],
    ids=["int", "float", "bool", "null", "empty-list", "list", "empty-dict", "dict"],
)
def test_similarity_and_explain_endpoints_when_text_is_not_a_string_return_422(client: TestClient, endpoint: str, field: str, value: Any) -> None:
    payload = {"text_a": SENTENCE, "text_b": SENTENCE, field: value}
    assert client.post(endpoint, json=payload).status_code == 422


@pytest.mark.parametrize("endpoint", ["/similarity", "/similarity/explain"])
@pytest.mark.parametrize("body", [[], [{"text_a": SENTENCE, "text_b": SENTENCE}], "text", 1, None], ids=["empty-list", "list-of-object", "string", "int", "null"])
def test_similarity_and_explain_endpoints_when_body_is_not_an_object_return_422(client: TestClient, endpoint: str, body: Any) -> None:
    assert client.post(endpoint, json=body).status_code == 422


def test_similarity_endpoint_when_body_not_json_returns_422(client: TestClient) -> None:
    r = client.post("/similarity", content="this is not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


# --- startup probe ----------------------------------------------------------


def nan_output() -> npt.NDArray[np.float32]:
    out = good_output()
    out[0, 0] = np.nan
    return out


def test_lifespan_when_probe_passes_sets_embedder_on_app_state(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = StubEmbedder(good_output())
    scratch_app = start_with(monkeypatch, stub)
    assert scratch_app.state.embedder is stub


def test_probe_embedder_when_output_valid_does_not_raise() -> None:
    probe_embedder(StubEmbedder(good_output()))  # does not raise


@pytest.mark.parametrize(
    "output, match",
    [
        (nan_output(), "NaN or inf"),
        (good_output().astype(np.float64), "float32"),
        (np.ones((1, EMBEDDING_DIM + 1), dtype=np.float32), "shape"),
        (np.ones((2, EMBEDDING_DIM), dtype=np.float32), "shape"),  # one text in, two rows out
        (good_output().tolist(), "ndarray"),
    ],
    ids=["nan", "float64", "wrong-dim", "wrong-rows", "not-ndarray"],
)
def test_probe_embedder_when_output_invalid_raises_runtime_error(output: Any, match: str) -> None:
    with pytest.raises(RuntimeError, match=match):
        probe_embedder(StubEmbedder(output))


def test_lifespan_when_probe_fails_raises_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Right dimension but broken output: only the probe call in lifespan stops this model from serving.
    with pytest.raises(RuntimeError, match="NaN or inf"):
        start_with(monkeypatch, StubEmbedder(nan_output()))


def test_lifespan_when_model_dimension_wrong_raises_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # A different model would otherwise pass the probe, because the probe compares against the model's own dim.
    wrong = StubEmbedder(np.ones((1, 384), dtype=np.float32), dim=384)
    with pytest.raises(RuntimeError, match=str(EMBEDDING_DIM)):
        start_with(monkeypatch, wrong)


# --- truncation flags -------------------------------------------------------

CJK = "人"  # one token per character, plus two special tokens per text


@pytest.mark.parametrize("flags", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("endpoint", ["/similarity", "/similarity/explain"])
def test_similarity_and_explain_endpoints_when_embedder_reports_truncation_return_flags_per_text(
    client: TestClient, use_embedder: Callable[[Any], None], endpoint: str, flags: tuple[bool, bool]
) -> None:
    stub = fixed_embedder({3: 1.0}, {3: 1.0})
    stub.flags = flags
    use_embedder(stub)
    body = client.post(endpoint, json={"text_a": SENTENCE, "text_b": OTHER_SENTENCE}).json()
    assert (body["text_a_truncated"], body["text_b_truncated"]) == flags


@pytest.mark.parametrize("chars, expected", [(1022, False), (1023, True)], ids=["exactly-at-limit", "one-token-over"])
def test_truncated_when_real_model_counts_special_tokens_against_the_limit(client: TestClient, chars: int, expected: bool) -> None:
    # 1022 characters are 1022 tokens plus the two special tokens: exactly the 1024-token limit.
    assert app.state.embedder.truncated([CJK * chars]) == [expected]


def test_similarity_endpoint_when_only_one_text_exceeds_token_limit_flags_only_that_text(client: TestClient) -> None:
    body = post_similarity(client, SENTENCE, CJK * 2000).json()
    assert (body["text_a_truncated"], body["text_b_truncated"]) == (False, True)


# --- non-finite output at request time --------------------------------------


@pytest.mark.parametrize("endpoint", ["/similarity", "/similarity/explain"])
@pytest.mark.parametrize("bad", [np.nan, np.inf], ids=["nan", "inf"])
def test_similarity_and_explain_endpoints_when_embedding_non_finite_return_generic_500(
    client: TestClient, use_embedder: Callable[[Any], None], bad: float, endpoint: str
) -> None:
    use_embedder(constant_embedder(bad))
    r = client.post(endpoint, json={"text_a": SENTENCE, "text_b": OTHER_SENTENCE})
    assert r.status_code == 500
    assert r.json() == {"detail": app_module.SIMILARITY_FAILED_DETAIL}


@pytest.mark.parametrize("endpoint", ["/similarity", "/similarity/explain"])
def test_similarity_and_explain_endpoints_when_embedding_non_finite_log_without_texts(
    client: TestClient, use_embedder: Callable[[Any], None], caplog: pytest.LogCaptureFixture, endpoint: str
) -> None:
    # The operator can see what happened, without the private texts.
    use_embedder(constant_embedder(np.nan))
    with caplog.at_level("ERROR", logger="embedding-api"):
        client.post(endpoint, json={"text_a": SENTENCE, "text_b": OTHER_SENTENCE})
    assert "non-finite embedding" in caplog.text
    assert SENTENCE not in caplog.text


def test_similarity_endpoint_when_score_non_finite_returns_generic_500(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "cos_sim", lambda a, b: float("nan"))
    r = post_similarity(client, SENTENCE, OTHER_SENTENCE)
    assert r.status_code == 500
    assert r.json() == {"detail": app_module.SIMILARITY_FAILED_DETAIL}


# --- /similarity/explain ----------------------------------------------------


# Per-dimension products: dim 3 -> +4, dim 5 -> -1, dim 7 -> 0. The total absolute signal is 5,
# so the contributions are +0.8 (dim 3) and -0.2 (dim 5). Dim 300 is not a topic and is ignored.
FIXED = fixed_embedder({3: 2.0, 5: 1.0, 7: 1.0}, {3: 2.0, 5: -1.0, 300: 9.0})


def post_explain(client: TestClient, **extra: Any) -> Response:
    payload = {"text_a": SENTENCE, "text_b": OTHER_SENTENCE, **extra}
    return client.post("/similarity/explain", json=payload)


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ({0: 1.0}, {0: 3.0}, 1.0),  # same direction, different length
        ({0: 1.0}, {1: 1.0}, 0.0),  # perpendicular
        ({0: 1.0}, {0: -1.0}, -1.0),  # opposite
        ({0: 1.0, 1: 1.0}, {0: 1.0}, 2**-0.5),  # 45 degrees apart
    ],
    ids=["same-direction", "perpendicular", "opposite", "45-degrees"],
)
def test_similarity_endpoint_when_vectors_known_returns_exact_cosine(
    client: TestClient, use_embedder: Callable[[Any], None], a: dict[int, float], b: dict[int, float], expected: float
) -> None:
    use_embedder(fixed_embedder(a, b))
    assert post_similarity(client, SENTENCE, OTHER_SENTENCE).json()["similarity"] == pytest.approx(expected, abs=1e-6)


def test_explain_endpoint_when_no_counts_given_returns_two_of_each(
    client: TestClient, use_embedder: Callable[[Any], None]
) -> None:
    # Three topics agree and three disagree, so the defaults (2 and 2) are what limits the lists.
    use_embedder(
        fixed_embedder({1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0}, {1: 1.0, 2: 1.0, 3: 1.0, 4: -1.0, 5: -1.0, 6: -1.0})
    )
    body = post_explain(client).json()
    assert len(body["shared_topics"]) == 2
    assert len(body["differing_topics"]) == 2


def test_explain_endpoint_when_contribution_is_a_third_returns_it_unrounded(
    client: TestClient, use_embedder: Callable[[Any], None]
) -> None:
    use_embedder(fixed_embedder({0: 1.0, 1: 1.0, 2: 1.0}, {0: 1.0, 1: 1.0, 2: 1.0}))
    body = post_explain(client, top_shared=1).json()
    assert body["shared_topics"][0]["contribution"] == pytest.approx(1 / 3, abs=1e-12)


def test_explain_endpoint_when_dimension_product_positive_returns_labelled_shared_topic(client: TestClient, use_embedder: Callable[[Any], None]) -> None:
    use_embedder(FIXED)
    body = post_explain(client, top_shared=2, top_differing=2).json()
    assert [t["topic"] for t in body["shared_topics"]] == [app.state.topics[3].label]
    assert body["shared_topics"][0]["contribution"] == pytest.approx(0.8)


def test_explain_endpoint_when_dimension_product_negative_returns_labelled_differing_topic(client: TestClient, use_embedder: Callable[[Any], None]) -> None:
    use_embedder(FIXED)
    body = post_explain(client, top_shared=2, top_differing=2).json()
    assert [t["topic"] for t in body["differing_topics"]] == [app.state.topics[5].label]
    assert body["differing_topics"][0]["contribution"] == pytest.approx(-0.2)


def test_explain_endpoint_when_signal_only_beyond_topic_dimensions_returns_no_topics(
    client: TestClient, use_embedder: Callable[[Any], None]
) -> None:
    # Both texts agree only on dimension 300 (not a topic): the similarity sees it, the topics must not.
    use_embedder(fixed_embedder({N_TOPICS: 1.0}, {N_TOPICS: 1.0}))
    body = post_explain(client).json()
    assert (body["shared_topics"], body["differing_topics"]) == ([], [])


def test_explain_endpoint_when_valid_returns_same_similarity_as_similarity_endpoint(client: TestClient, use_embedder: Callable[[Any], None]) -> None:
    use_embedder(FIXED)
    plain = post_similarity(client, SENTENCE, OTHER_SENTENCE).json()["similarity"]
    assert post_explain(client).json()["similarity"] == pytest.approx(plain)


PARAGRAPH_A = (
    "Regular exercise strengthens the heart and improves blood circulation. Doctors recommend at least "
    "thirty minutes of moderate activity most days of the week. Combined with a balanced diet, it lowers "
    "the risk of heart disease and helps keep body weight under control."
)
PARAGRAPH_B = (
    "Staying physically active every week is one of the best ways to protect your cardiovascular system. "
    "Health experts advise walking, cycling or swimming for half an hour a day. Eating well alongside "
    "this routine reduces the chance of illness and supports a healthy weight."
)


def test_explain_endpoint_when_real_model_and_related_paragraphs_returns_well_formed_topics(client: TestClient) -> None:
    # Real model, no stub: which topics appear depends on the model, so only the structure is checked.
    text_a, text_b = PARAGRAPH_A, PARAGRAPH_B
    body = client.post("/similarity/explain", json={"text_a": text_a, "text_b": text_b, "top_shared": 3, "top_differing": 3}).json()
    labels = {t.label for t in app.state.topics}
    assert -1.0 <= body["similarity"] <= 1.0
    assert body["shared_topics"], "related texts should share at least one topic"
    assert len(body["shared_topics"]) <= 3 and len(body["differing_topics"]) <= 3
    assert all(t["topic"] in labels and 0 < t["contribution"] <= 1 for t in body["shared_topics"])
    assert all(t["topic"] in labels and -1 <= t["contribution"] < 0 for t in body["differing_topics"])


def test_explain_endpoint_when_valid_returns_only_public_keys(client: TestClient) -> None:
    body = post_explain(client).json()
    assert set(body) == {"similarity", "text_a_truncated", "text_b_truncated", "shared_topics", "differing_topics"}


def test_explain_endpoint_when_texts_identical_returns_no_differing_topics(client: TestClient, use_embedder: Callable[[Any], None]) -> None:
    use_embedder(fixed_embedder({3: 1.0, 5: 1.0}, {3: 1.0, 5: 1.0}))
    assert post_explain(client, top_differing=3).json()["differing_topics"] == []  # a text never opposes itself


@pytest.mark.parametrize(
    "extra",
    [
        {"top_shared": 0, "top_differing": 0},
        {"top_shared": -1},
        {"top_differing": -1},
        {"top_shared": 11},
        {"top_differing": 11},
        {"top_shared": 10**12},
        {"top_shared": 2.5},
        {"top_differing": None},
        {"top_differing": [1]},
        {"top_shared": "many"},
        {"include_summary": "yes"},
        {"include_summary": "true"},
        {"include_summary": 1},
        {"include_summary": 0},
        {"include_summary": None},
    ],
)
def test_explain_endpoint_when_request_invalid_returns_422(client: TestClient, extra: dict[str, Any]) -> None:
    assert post_explain(client, **extra).status_code == 422


def test_explain_endpoint_when_server_flag_off_hides_summaries(
    client: TestClient, use_embedder: Callable[[Any], None], set_state: Callable[[str, Any], None]
) -> None:
    use_embedder(FIXED)
    set_state("expose_summaries", False)
    body = post_explain(client, include_summary=True).json()
    assert body["shared_topics"] and body["differing_topics"]
    assert all("summary" not in t for t in body["shared_topics"] + body["differing_topics"])


def test_explain_endpoint_when_both_flags_on_returns_summaries(
    client: TestClient, use_embedder: Callable[[Any], None], set_state: Callable[[str, Any], None]
) -> None:
    use_embedder(FIXED)
    set_state("expose_summaries", True)
    body = post_explain(client, include_summary=True).json()
    assert body["shared_topics"][0]["summary"] == app.state.topics[3].summary


def test_explain_endpoint_when_request_flag_off_hides_summaries(
    client: TestClient, use_embedder: Callable[[Any], None], set_state: Callable[[str, Any], None]
) -> None:
    use_embedder(FIXED)
    set_state("expose_summaries", True)
    assert "summary" not in post_explain(client).json()["shared_topics"][0]


def test_explain_endpoint_when_embedding_shorter_than_topic_range_returns_generic_500(
    client: TestClient, use_embedder: Callable[[Any], None]
) -> None:
    use_embedder(StubEmbedder(np.ones((2, 4), dtype=np.float32), dim=4))
    r = post_explain(client)
    assert r.status_code == 500
    assert r.json() == {"detail": app_module.SIMILARITY_FAILED_DETAIL}


@pytest.mark.parametrize("endpoint", ["/similarity", "/similarity/explain"])
@pytest.mark.parametrize(
    "shape",
    [(1, EMBEDDING_DIM), (3, EMBEDDING_DIM), (2, EMBEDDING_DIM + 1), (EMBEDDING_DIM,)],
    ids=["one-row", "three-rows", "wrong-dim", "one-dimensional"],
)
def test_similarity_and_explain_endpoints_when_embedding_shape_wrong_return_generic_500(
    client: TestClient, use_embedder: Callable[[Any], None], endpoint: str, shape: tuple[int, ...]
) -> None:
    use_embedder(StubEmbedder(np.ones(shape, dtype=np.float32)))
    r = client.post(endpoint, json={"text_a": SENTENCE, "text_b": OTHER_SENTENCE})
    assert r.status_code == 500
    assert r.json() == {"detail": app_module.SIMILARITY_FAILED_DETAIL}


@pytest.mark.parametrize("endpoint", ["/similarity", "/similarity/explain"])
def test_similarity_and_explain_endpoints_when_embedder_returns_non_array_return_generic_500(
    client: TestClient, use_embedder: Callable[[Any], None], endpoint: str
) -> None:
    use_embedder(StubEmbedder(np.ones((2, EMBEDDING_DIM), dtype=np.float32).tolist()))
    r = client.post(endpoint, json={"text_a": SENTENCE, "text_b": OTHER_SENTENCE})
    assert r.status_code == 500
    assert r.json() == {"detail": app_module.SIMILARITY_FAILED_DETAIL}


@pytest.mark.parametrize("raw, expected", [(1.0000001, 1.0), (-1.0000001, -1.0), (0.5, 0.5)], ids=["above", "below", "inside"])
@pytest.mark.parametrize("endpoint", ["/similarity", "/similarity/explain"])
def test_similarity_and_explain_endpoints_when_score_outside_unit_range_clamp_to_it(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, endpoint: str, raw: float, expected: float
) -> None:
    monkeypatch.setattr(app_module, "cos_sim", lambda a, b: raw)
    r = client.post(endpoint, json={"text_a": SENTENCE, "text_b": OTHER_SENTENCE})
    assert r.json()["similarity"] == expected


@pytest.mark.parametrize("value", ["yes", "ture", "on", "1"])
def test_lifespan_when_expose_summaries_env_invalid_raises_value_error(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("EXPOSE_SUMMARIES", value)
    with pytest.raises(ValueError, match="EXPOSE_SUMMARIES"):
        start_with(monkeypatch, StubEmbedder(good_output()))


@pytest.mark.parametrize(
    "value, expected", [("true", True), ("TRUE", True), ("false", False), ("FALSE", False), ("", True), (None, True)]
)
def test_lifespan_when_expose_summaries_env_is_valid_or_unset_sets_flag(monkeypatch: pytest.MonkeyPatch, value: str | None, expected: bool) -> None:
    if value is None:  # variable unset: summaries are exposed by default
        monkeypatch.delenv("EXPOSE_SUMMARIES", raising=False)
    else:
        monkeypatch.setenv("EXPOSE_SUMMARIES", value)
    assert start_with(monkeypatch, StubEmbedder(good_output())).state.expose_summaries is expected
