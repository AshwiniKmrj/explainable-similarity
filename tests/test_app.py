from fastapi.testclient import TestClient

from app import app


def test_health():
    with TestClient(app) as c:
        assert c.get("/health").json() == {"status": "ok"}


def test_identical_texts_score_one():
    with TestClient(app) as c:
        r = c.post("/similarity", json={"text_a": "hello", "text_b": "hello"})
        assert r.status_code == 200
        assert abs(r.json()["similarity"] - 1.0) < 1e-5


def test_empty_text_rejected():
    with TestClient(app) as c:
        r = c.post("/similarity", json={"text_a": "", "text_b": "hello"})
        assert r.status_code == 422