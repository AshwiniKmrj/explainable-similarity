"""Security tests: information leakage, error handling and resource abuse on the HTTP endpoints."""

import logging
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app as app_module
from support import RaisingEmbedder

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
