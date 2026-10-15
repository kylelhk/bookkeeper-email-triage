import pytest
from fastapi.testclient import TestClient

import app as triage_app

TOKEN = "test-token-123"


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("API_TOKEN", TOKEN)
    # Always force the mock: Tests must NEVER call a real AI (it costs quota and money), even if .env says PROVIDER=gemini.
    monkeypatch.setenv("PROVIDER", "mock")
    return TestClient(triage_app.app)


def auth(token=TOKEN):
    return {"Authorization": f"Bearer {token}"}


def test_health_is_open(client):
    assert client.get("/health").status_code == 200


def test_missing_token_rejected(client):
    assert client.post("/triage", json={"body": "hello"}).status_code == 401


def test_wrong_token_rejected(client):
    assert client.post(
        "/triage", json={"body": "hello"}, headers=auth("nope")).status_code == 401


def test_unconfigured_server_fails_closed(monkeypatch):
    monkeypatch.delenv("API_TOKEN", raising=False)
    r = TestClient(triage_app.app).post(
        "/triage", json={"body": "hello"}, headers=auth("anything"))
    assert r.status_code == 500


def test_empty_body_rejected(client):
    assert client.post(
        "/triage", json={"body": ""}, headers=auth()).status_code == 422


def test_valid_request_returns_draft_schema(client):
    r = client.post("/triage", json={"subject": "BAS",
                    "body": "Can you do my BAS please?"}, headers=auth())
    assert r.status_code == 200
    data = r.json()
    assert data["category"] == "BAS_GST"
    assert isinstance(data["missing_information"], list)
    assert "guardrail_notes" in data


def test_unusable_model_output_gives_502(client, monkeypatch):
    # The fake must accept one argument because the app calls run_model(email) (inside the try block of the /triage endpoint).
    # The leading underscore marks _email as intentionally unused: this fake ignores the input and always returns an invalid category.
    monkeypatch.setattr(triage_app, "run_model",
                        lambda _email: {"category": "NOT_REAL"})
    assert client.post(
        "/triage", json={"body": "hello"}, headers=auth()).status_code == 502


def test_provider_failure_gives_502_without_leaking_details(client, monkeypatch):
    # Same signature as run_model(email); _email is unused because this fake only simulates the provider crashing.
    def boom(_email):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(triage_app, "run_model", boom)
    r = client.post("/triage", json={"body": "hello"}, headers=auth())
    assert r.status_code == 502
    assert "secret" not in r.text
