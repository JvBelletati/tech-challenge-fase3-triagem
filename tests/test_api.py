import pytest
from fastapi.testclient import TestClient

from triagem.api import metrics
from triagem.api.main import app
from triagem.config import CURRENT_MODEL_DIR

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
    real numerator."""
    errors_before = metrics.ERRORS.labels(tipo="validacao")._value.get()
    requests_before = metrics.REQUESTS.labels(
        endpoint="/predict", status="error", categoria="none"
    )._value.get()

    response = client.post("/predict", json={"texto": "dor"})

    assert response.status_code == 422
    body = response.json()
    assert body["detail"][0]["type"] == "string_too_short"
    assert body["detail"][0]["loc"] == ["body", "texto"]

    errors_after = metrics.ERRORS.labels(tipo="validacao")._value.get()
    requests_after = metrics.REQUESTS.labels(
        endpoint="/predict", status="error", categoria="none"
    )._value.get()
    assert errors_after == errors_before + 1
    assert requests_after == requests_before + 1


@needs_model
def test_metrics_endpoint_exposes_prometheus_format(client):
    client.post("/predict", json={"texto": LAUDO})
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "triagem_requests_total" in response.text
    assert "triagem_inference_duration_seconds" in response.text
