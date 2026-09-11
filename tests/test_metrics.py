import pytest
from prometheus_client import generate_latest
from pydantic import ValidationError

from triagem.api import metrics as m
from triagem.api.schemas import PredictRequest
from triagem.inference.predictor import Prediction


def test_all_six_required_metrics_are_registered():
    exported = generate_latest().decode()
    for name in (
        "triagem_requests_total",
        "triagem_request_duration_seconds",
        "triagem_inference_duration_seconds",
        "triagem_errors_total",
        "triagem_prediction_confidence",
        "triagem_model",
    ):
        assert name in exported, f"{name} missing from /metrics output"


def test_observe_prediction_increments_the_counter():
    before = m.REQUESTS.labels(
        endpoint="/predict", status="200", categoria="neoplasms"
    )._value.get()
    prediction = Prediction(
        category_id=1,
        category="neoplasms",
        priority="URGENTE",
        confidence=0.9,
        needs_human_review=False,
        inference_seconds=0.001,
    )
    m.observe_prediction(prediction, http_seconds=0.004)
    after = m.REQUESTS.labels(endpoint="/predict", status="200", categoria="neoplasms")._value.get()
    assert after == before + 1


def test_request_rejects_text_below_minimum_length():
    with pytest.raises(ValidationError):
        PredictRequest(texto="curto")


def test_request_accepts_a_realistic_report():
    request = PredictRequest(texto="a" * 120)
    assert len(request.texto) == 120
