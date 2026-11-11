"""Security tests: information leakage, error handling and resource abuse on the HTTP endpoints."""

import logging
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app as app_module
from support import AUTH_HEADERS, RaisingEmbedder, StubEmbedder, good_output, start_with

SECRET = "SECRET-PATIENT-RECORD-4711"


# --- information leakage ----------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [{"text_a": SECRET + "x" * app_module.MAX_TEXT_CHARS, "text_b": "ok"}, {"text_a": " ", "text_b": SECRET}],
    ids=["too-long", "other-field-blank"],
)
def test_similarity_endpoint_when_text_invalid_does_not_echo_any_text(client: TestClient, payload: dict[str, str]) -> None:
    # Private texts must not come back in the error body, nor reach any proxy log of responses.
    r = client.post("/similarity", json=payload)
    assert SECRET not in r.text


@pytest.mark.parametrize("endpoint", ["/similarity", "/similarity/explain"])
def test_similarity_and_explain_endpoints_when_successful_do_not_log_texts(client: TestClient, caplog: pytest.LogCaptureFixture, endpoint: str) -> None:
    # Guards against debug logging of request payloads.
    with caplog.at_level(logging.DEBUG):
        client.post(endpoint, json={"text_a": SECRET, "text_b": SECRET})
    assert SECRET not in caplog.text


@pytest.mark.parametrize("endpoint", ["/similarity", "/similarity/explain"])
def test_similarity_and_explain_endpoints_when_embedder_raises_log_type_only(
    quiet_client: TestClient, use_embedder: Callable[[Any], None], caplog: pytest.LogCaptureFixture, endpoint: str
) -> None:
    # Exception messages can hold the private text and file paths, so the log may name the type but not the message.
    use_embedder(RaisingEmbedder())
    with caplog.at_level("ERROR", logger="embedding-api"):
        quiet_client.post(endpoint, json={"text_a": SECRET, "text_b": "b"})
    assert "RuntimeError" in caplog.text
    assert SECRET not in caplog.text and "weights.bin" not in caplog.text


@pytest.mark.parametrize("endpoint", ["/similarity", "/similarity/explain"])
def test_similarity_and_explain_endpoints_when_embedder_raises_return_generic_500(
    quiet_client: TestClient, use_embedder: Callable[[Any], None], endpoint: str
) -> None:
    # Guards against stack traces, paths or texts in the error body; clients parse {"detail": ...}.
    use_embedder(RaisingEmbedder())
    r = quiet_client.post(endpoint, json={"text_a": SECRET, "text_b": "b"})
    assert r.status_code == 500
    assert SECRET not in r.text and "weights.bin" not in r.text and "Traceback" not in r.text
    assert r.json() == {"detail": app_module.SIMILARITY_FAILED_DETAIL}


# --- resource abuse ---------------------------------------------------------


def test_similarity_endpoint_when_body_too_large_returns_413(client: TestClient) -> None:
    # Fields are capped in length, but unknown extra keys let a client push arbitrary bytes into memory.
    body = '{"text_a":"a","text_b":"b","pad":"' + "x" * (12 * 1024 * 1024) + '"}'
    r = client.post("/similarity", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 413


def test_similarity_endpoint_when_streamed_body_too_large_returns_413(client: TestClient) -> None:
    # A chunked upload has no Content-Length, so the limit must also count the bytes as they arrive.
    def chunks() -> Iterator[bytes]:
        yield b'{"text_a":"a","text_b":"b","pad":"'
        for _ in range(4):
            yield b"x" * app_module.MAX_BODY_BYTES
        yield b'"}'

    r = client.post("/similarity", content=chunks(), headers={"Content-Type": "application/json"})
    assert r.status_code == 413


def test_similarity_endpoint_when_texts_max_length_fully_escaped_returns_200(client: TestClient) -> None:
    # Two maximum-length texts, with every character JSON-escaped to 6 bytes, must fit under the limit.
    escaped = "\\u00e9" * app_module.MAX_TEXT_CHARS
    body = ('{"text_a":"' + escaped + '","text_b":"' + escaped + '"}').encode("ascii")
    assert len(body) <= app_module.MAX_BODY_BYTES
    assert client.post("/similarity", content=body, headers={"Content-Type": "application/json"}).status_code == 200


def test_similarity_endpoint_when_json_deeply_nested_returns_400(quiet_client: TestClient) -> None:
    # Guards against recursion errors or crashes turning into 500s or killing the worker.
    body = "[" * 60_000 + "]" * 60_000
    r = quiet_client.post("/similarity", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 400


@pytest.mark.parametrize("text", ["\u200b", "\x00"])
def test_similarity_endpoint_when_text_is_only_zero_width_or_nul_returns_200(client: TestClient, text: str) -> None:
    # Zero-width and NUL characters are valid, non-blank text; the model handles them.
    assert client.post("/similarity", json={"text_a": text, "text_b": "ok"}).status_code == 200


@pytest.mark.parametrize("escape", ["\\ud83d", "\\udfff"], ids=["high", "low"])
def test_similarity_endpoint_when_text_has_lone_surrogate_returns_422(quiet_client: TestClient, escape: str) -> None:
    # Valid JSON can carry a lone surrogate, which is not valid Unicode text. The client library cannot
    # build such a body from a Python str, so the escape is sent as raw ASCII bytes.
    body = ('{"text_a":"a' + escape + 'b","text_b":"ok"}').encode("ascii")
    r = quiet_client.post("/similarity", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_similarity_endpoint_when_body_is_form_returns_422(client: TestClient) -> None:
    # Form bodies must not be parsed leniently into the model.
    assert client.post("/similarity", data={"text_a": "a", "text_b": "b"}).status_code == 422


def _padded_body(total: int) -> bytes:
    head, tail = b'{"text_a":"a","text_b":"b","pad":"', b'"}'
    return head + b"x" * (total - len(head) - len(tail)) + tail


def test_similarity_endpoint_when_body_exactly_at_limit_returns_200(client: TestClient) -> None:
    r = client.post("/similarity", content=_padded_body(app_module.MAX_BODY_BYTES), headers={"Content-Type": "application/json"})
    assert r.status_code == 200


def test_similarity_endpoint_when_body_one_byte_over_limit_returns_413(client: TestClient) -> None:
    r = client.post("/similarity", content=_padded_body(app_module.MAX_BODY_BYTES + 1), headers={"Content-Type": "application/json"})
    assert r.status_code == 413
    assert r.json() == {"detail": "Request body too large."}


# --- rate limiting ----------------------------------------------------------

PAIR = {"text_a": "a", "text_b": "b"}


@pytest.fixture
def limit_to(set_state: Callable[[str, Any], None]) -> Callable[[int], None]:
    """Switch the rate limit on at `n` POSTs per minute, starting from empty counters."""

    def apply(n: int) -> None:
        set_state("rate_limit_per_minute", n)
        set_state("rate_limit_hits", {})

    return apply


def test_similarity_endpoint_when_client_exceeds_rate_limit_returns_429(client: TestClient, limit_to: Callable[[int], None]) -> None:
    limit_to(2)
    statuses = [client.post("/similarity", json=PAIR).status_code for _ in range(3)]
    assert statuses == [200, 200, 429]


def test_similarity_endpoint_when_rate_limited_says_when_to_retry(client: TestClient, limit_to: Callable[[int], None]) -> None:
    limit_to(1)
    client.post("/similarity", json=PAIR)
    r = client.post("/similarity", json=PAIR)
    assert r.json() == {"detail": app_module.TOO_MANY_REQUESTS_DETAIL}
    assert 1 <= int(r.headers["Retry-After"]) <= app_module.RATE_WINDOW_SECONDS


def test_similarity_endpoint_when_rate_window_has_passed_accepts_again(
    client: TestClient, limit_to: Callable[[int], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    limit_to(1)
    now = 1000.0
    monkeypatch.setattr(app_module.time, "monotonic", lambda: now)
    client.post("/similarity", json=PAIR)
    now += app_module.RATE_WINDOW_SECONDS
    assert client.post("/similarity", json=PAIR).status_code == 200


def test_rate_limit_counts_invalid_requests_too(client: TestClient, limit_to: Callable[[int], None]) -> None:
    # Otherwise a client could hammer the service with requests that fail validation.
    limit_to(1)
    client.post("/similarity", json={})
    assert client.post("/similarity", json=PAIR).status_code == 429


def test_health_endpoint_when_client_is_rate_limited_still_answers(client: TestClient, limit_to: Callable[[int], None]) -> None:
    # Container health probes must never be throttled, or the platform restarts a busy but healthy service.
    limit_to(1)
    client.post("/similarity", json=PAIR)
    client.post("/similarity", json=PAIR)
    assert client.get("/health").status_code == 200


def test_rate_limit_when_forwarded_header_trusted_counts_each_forwarded_client_separately(
    client: TestClient, limit_to: Callable[[int], None], set_state: Callable[[str, Any], None]
) -> None:
    limit_to(1)
    set_state("trust_forwarded_for", True)
    first = client.post("/similarity", json=PAIR, headers={"X-Forwarded-For": "203.0.113.1"})
    other = client.post("/similarity", json=PAIR, headers={"X-Forwarded-For": "203.0.113.2"})
    assert (first.status_code, other.status_code) == (200, 200)


def test_rate_limit_when_forwarded_header_trusted_ignores_entries_the_caller_made_up(
    client: TestClient, limit_to: Callable[[int], None], set_state: Callable[[str, Any], None]
) -> None:
    # Only the last entry comes from the proxy; a caller rotating the earlier ones must stay in one bucket.
    limit_to(1)
    set_state("trust_forwarded_for", True)
    client.post("/similarity", json=PAIR, headers={"X-Forwarded-For": "1.1.1.1, 203.0.113.1:5050"})
    r = client.post("/similarity", json=PAIR, headers={"X-Forwarded-For": "2.2.2.2, 203.0.113.1:6060"})
    assert r.status_code == 429


def test_rate_limit_when_forwarded_header_not_trusted_ignores_it(client: TestClient, limit_to: Callable[[int], None]) -> None:
    # Without a proxy in front, the header is whatever the caller typed.
    limit_to(1)
    client.post("/similarity", json=PAIR, headers={"X-Forwarded-For": "203.0.113.1"})
    r = client.post("/similarity", json=PAIR, headers={"X-Forwarded-For": "203.0.113.2"})
    assert r.status_code == 429


def test_rate_limit_when_client_table_full_refuses_new_clients(
    client: TestClient, limit_to: Callable[[int], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The counters must not grow without bound.
    limit_to(5)
    monkeypatch.setattr(app_module, "MAX_TRACKED_CLIENTS", 1)
    app_module.app.state.rate_limit_hits["198.51.100.7"] = (app_module.time.monotonic(), 1)
    assert client.post("/similarity", json=PAIR).status_code == 429


@pytest.mark.parametrize("value", ["-1", "ten", "1.5"])
def test_startup_when_rate_limit_setting_invalid_fails(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", value)
    with pytest.raises(ValueError, match="RATE_LIMIT_PER_MINUTE"):
        start_with(monkeypatch, StubEmbedder(good_output()))


def test_startup_when_rate_limit_not_set_uses_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RATE_LIMIT_PER_MINUTE")
    started = start_with(monkeypatch, StubEmbedder(good_output()))
    assert started.state.rate_limit_per_minute == app_module.DEFAULT_RATE_LIMIT_PER_MINUTE


# --- authentication ---------------------------------------------------------

ENDPOINTS = ["/similarity", "/similarity/explain"]
KEY = AUTH_HEADERS["X-API-Key"]


@pytest.fixture
def anonymous(client: TestClient) -> TestClient:
    """Client for the running app that sends no API key unless the test adds one."""
    return TestClient(app_module.app)


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_model_endpoints_when_api_key_missing_return_401(anonymous: TestClient, endpoint: str) -> None:
    r = anonymous.post(endpoint, json=PAIR)
    assert r.status_code == 401
    assert r.json() == {"detail": app_module.INVALID_API_KEY_DETAIL}


@pytest.mark.parametrize(
    "key",
    ["", "wrong", KEY[:-1], KEY + "x", KEY.upper()],
    ids=["empty", "unrelated", "one-char-short", "one-char-long", "wrong-case"],
)
@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_model_endpoints_when_api_key_wrong_return_401(anonymous: TestClient, endpoint: str, key: str) -> None:
    assert anonymous.post(endpoint, json=PAIR, headers={"X-API-Key": key}).status_code == 401


def test_similarity_endpoint_when_api_key_has_non_ascii_bytes_returns_401(anonymous: TestClient) -> None:
    # Guards against the comparison crashing into a 500 on input it cannot compare.
    r = anonymous.post("/similarity", json=PAIR, headers={"X-API-Key": b"cl\xe9"})
    assert r.status_code == 401


def test_similarity_endpoint_when_api_key_missing_never_runs_the_model(
    anonymous: TestClient, use_embedder: Callable[[Any], None]
) -> None:
    # RaisingEmbedder fails loudly if it is called, so a 401 here proves the model was not reached.
    use_embedder(RaisingEmbedder())
    assert anonymous.post("/similarity", json=PAIR).status_code == 401


def test_similarity_endpoint_when_api_key_wrong_does_not_reveal_the_key(anonymous: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        r = anonymous.post("/similarity", json=PAIR, headers={"X-API-Key": "wrong"})
    assert KEY not in r.text + caplog.text


def test_health_endpoint_when_api_key_missing_still_answers(anonymous: TestClient) -> None:
    # Container health probes cannot send the key.
    assert anonymous.get("/health").status_code == 200


def test_rate_limit_applies_to_requests_with_wrong_api_key(anonymous: TestClient, limit_to: Callable[[int], None]) -> None:
    # Bounds how fast a caller can guess keys.
    limit_to(1)
    anonymous.post("/similarity", json=PAIR, headers={"X-API-Key": "guess-1"})
    assert anonymous.post("/similarity", json=PAIR, headers={"X-API-Key": "guess-2"}).status_code == 429


@pytest.mark.parametrize("value", [None, "", "   ", "short"], ids=["unset", "empty", "blank", "too-short"])
def test_startup_when_api_key_missing_or_weak_fails(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    # The service must never start unprotected because a setting was forgotten.
    if value is None:
        monkeypatch.delenv("API_KEY")
    else:
        monkeypatch.setenv("API_KEY", value)
    with pytest.raises(ValueError, match="API_KEY"):
        start_with(monkeypatch, StubEmbedder(good_output()))
