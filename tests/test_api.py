import asyncio

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from triagem.api import main, metrics
from triagem.api.main import app
from triagem.config import CURRENT_MODEL_DIR
from triagem.inference.predictor import ModelNotFoundError

needs_model = pytest.mark.skipif(
    not (CURRENT_MODEL_DIR / "model.onnx").exists(),
    reason="requires a trained model in models/current (run scripts/treinar.py)",
)

LAUDO = (
    "The patient presented with acute chest pain radiating to the left arm, "
    "accompanied by dyspnea and diaphoresis. ECG showed ST elevation."
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@needs_model
def test_health_reports_the_loaded_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["modelo"]["runtime"] == "onnx"


@needs_model
def test_predict_returns_the_full_contract(client):
    response = client.post("/predict", json={"texto": LAUDO})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "categoria",
        "categoria_id",
        "prioridade",
        "confianca",
        "revisao_humana",
        "latencia_ms",
        "modelo",
    }
    assert body["prioridade"] in {"URGENTE", "ATENCAO", "NORMAL"}
    assert body["latencia_ms"] > 0


def test_predict_rejects_short_text(client):
    response = client.post("/predict", json={"texto": "dor"})
    assert response.status_code == 422


def test_predict_rejects_missing_field(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_validation_error_counts_as_error_metric(client):
    """A malformed request must still 422 with the usual body, but now also
    counts toward triagem_errors_total{tipo="validacao"} and
    triagem_requests_total{status="error"} so the error-rate panel has a
    real numerator. It must NOT add an observation to
    triagem_request_duration_seconds: the request never reached the model,
    so it has no service latency worth reporting, and feeding the latency
    histogram an invented zero would drag the p50/p95/p99 panel down."""
    errors_before = metrics.ERRORS.labels(tipo="validacao")._value.get()
    requests_before = metrics.REQUESTS.labels(
        endpoint="/predict", status="error", categoria="none"
    )._value.get()
    duration_count_before = (
        REGISTRY.get_sample_value(
            "triagem_request_duration_seconds_count", {"endpoint": "/predict"}
        )
        or 0.0
    )

    response = client.post("/predict", json={"texto": "dor"})

    assert response.status_code == 422
    body = response.json()
    assert body["detail"][0]["type"] == "string_too_short"
    assert body["detail"][0]["loc"] == ["body", "texto"]

    errors_after = metrics.ERRORS.labels(tipo="validacao")._value.get()
    requests_after = metrics.REQUESTS.labels(
        endpoint="/predict", status="error", categoria="none"
    )._value.get()
    duration_count_after = REGISTRY.get_sample_value(
        "triagem_request_duration_seconds_count", {"endpoint": "/predict"}
    )
    assert errors_after == errors_before + 1
    assert requests_after == requests_before + 1
    assert duration_count_after == duration_count_before


@needs_model
def test_metrics_endpoint_exposes_prometheus_format(client):
    client.post("/predict", json={"texto": LAUDO})
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "triagem_requests_total" in response.text
    assert "triagem_inference_duration_seconds" in response.text


# --- Degraded mode: no model loaded --------------------------------------
#
# main.lifespan catches ModelNotFoundError at startup and leaves
# `_predictor` as None instead of letting the app crash - "a container that
# refuses to boot tells an orchestrator nothing useful" (see the comment in
# main.py). The tests below reproduce exactly that post-boot-failure state
# by monkeypatching the module-level `_predictor` global directly, which is
# the same state main.lifespan would leave behind if Predictor() raised
# ModelNotFoundError (e.g. an empty TRIAGEM_MODELS_DIR). They run
# unconditionally - unlike the @needs_model tests above, they must pass
# whether or not a real model is present in models/current/, and they run
# independently of whichever model `client` already loaded: monkeypatch
# restores `_predictor` to its previous value after each test, so nothing
# leaks into other tests in this module.


def test_lifespan_survives_a_missing_model_at_startup(monkeypatch):
    """Directly exercises main.lifespan's `except ModelNotFoundError` branch:
    startup must not raise when no model artifact exists - it logs and
    leaves `_predictor` as None so /health and /predict can report the
    failure instead of the process refusing to boot (see the comment in
    main.py). This drives `lifespan` directly rather than opening a second
    TestClient on the shared `app` singleton, and manually restores
    `_predictor` afterwards, so it cannot clobber the model the
    module-scoped `client` fixture already loaded for the other tests in
    this module."""

    class _AlwaysMissing:
        def __init__(self, *args, **kwargs):
            raise ModelNotFoundError("simulated: no model artifact for this test")

    monkeypatch.setattr(main, "Predictor", _AlwaysMissing)
    original_predictor = main._predictor

    async def _boot_and_check() -> None:
        async with main.lifespan(main.app):
            assert main.get_predictor() is None

    try:
        asyncio.run(_boot_and_check())
    finally:
        main._predictor = original_predictor


def test_predict_returns_503_when_no_model_is_loaded(client, monkeypatch):
    monkeypatch.setattr(main, "_predictor", None)

    response = client.post("/predict", json={"texto": LAUDO})

    assert response.status_code == 503
    assert response.json()["detail"] == "modelo indisponivel"


def test_health_returns_503_when_no_model_is_loaded(client, monkeypatch):
    monkeypatch.setattr(main, "_predictor", None)

    response = client.get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "modelo indisponivel"
    assert body["modelo"] is None


# --- Degraded mode: inference failure -------------------------------------


class _FailingPredictor:
    """Stands in for a loaded Predictor whose ONNX session errors mid-request."""

    version = "test-version"
    runtime = "onnx"

    def predict(self, texto: str):
        raise RuntimeError("simulated inference failure")


def test_predict_failure_returns_500_and_counts_as_inference_error(client, monkeypatch):
    """Exercises /predict's except branch: an inference exception must become
    a generic 500 (never leaking exception internals to the client) and must
    increment triagem_errors_total{tipo="inferencia"} - the metric the
    Grafana 'Erros por tipo' panel exists to display. Unlike the validation-
    error counterpart above, this failure happens after work started, so it
    also must be excluded from the success-path metrics (observe_prediction
    is never called)."""
    monkeypatch.setattr(main, "_predictor", _FailingPredictor())

    errors_before = metrics.ERRORS.labels(tipo="inferencia")._value.get()
    confidence_count_before = (
        REGISTRY.get_sample_value("triagem_prediction_confidence_count") or 0.0
    )

    response = client.post("/predict", json={"texto": LAUDO})

    assert response.status_code == 500
    assert response.json()["detail"] == "falha na inferencia"

    errors_after = metrics.ERRORS.labels(tipo="inferencia")._value.get()
    confidence_count_after = REGISTRY.get_sample_value("triagem_prediction_confidence_count") or 0.0
    assert errors_after == errors_before + 1
    assert confidence_count_after == confidence_count_before
