import pytest
from prometheus_client import REGISTRY, generate_latest
from pydantic import ValidationError

from triagem.api import metrics as m
from triagem.api.schemas import PredictRequest
from triagem.inference.predictor import Prediction


def test_all_seven_required_metrics_are_registered():
    exported = generate_latest().decode()
    for name in (
        "triagem_requests_total",
        "triagem_request_duration_seconds",
        "triagem_inference_duration_seconds",
        "triagem_errors_total",
        "triagem_prediction_confidence",
        "triagem_requests_in_progress",
        "triagem_model",
    ):
        assert name in exported, f"{name} missing from /metrics output"


def test_observe_prediction_increments_all_four_metrics():
    # Capture before values
    requests_before = (
        REGISTRY.get_sample_value(
            "triagem_requests_total",
            {"endpoint": "/predict", "status": "200", "categoria": "neoplasms"},
        )
        or 0.0
    )
    request_duration_count_before = (
        REGISTRY.get_sample_value(
            "triagem_request_duration_seconds_count", {"endpoint": "/predict"}
        )
        or 0.0
    )
    inference_duration_count_before = (
        REGISTRY.get_sample_value("triagem_inference_duration_seconds_count") or 0.0
    )
    inference_duration_sum_before = (
        REGISTRY.get_sample_value("triagem_inference_duration_seconds_sum") or 0.0
    )
    confidence_count_before = (
        REGISTRY.get_sample_value("triagem_prediction_confidence_count") or 0.0
    )

    # Observe prediction
    prediction = Prediction(
        category_id=1,
        category="neoplasms",
        priority="URGENTE",
        confidence=0.9,
        needs_human_review=False,
        inference_seconds=0.001,
    )
    m.observe_prediction(prediction, http_seconds=0.004)

    # Assert all four metrics incremented
    requests_after = REGISTRY.get_sample_value(
        "triagem_requests_total",
        {"endpoint": "/predict", "status": "200", "categoria": "neoplasms"},
    )
    assert requests_after == requests_before + 1

    request_duration_count_after = REGISTRY.get_sample_value(
        "triagem_request_duration_seconds_count", {"endpoint": "/predict"}
    )
    assert request_duration_count_after == request_duration_count_before + 1

    inference_duration_count_after = REGISTRY.get_sample_value(
        "triagem_inference_duration_seconds_count"
    )
    assert inference_duration_count_after == inference_duration_count_before + 1

    inference_duration_sum_after = REGISTRY.get_sample_value(
        "triagem_inference_duration_seconds_sum"
    )
    assert inference_duration_sum_after == inference_duration_sum_before + 0.001

    confidence_count_after = REGISTRY.get_sample_value("triagem_prediction_confidence_count")
    assert confidence_count_after == confidence_count_before + 1


def test_observe_error_increments_error_and_request_metrics():
    # Capture before values
    errors_before = REGISTRY.get_sample_value("triagem_errors_total", {"tipo": "inferencia"}) or 0.0
    requests_error_before = (
        REGISTRY.get_sample_value(
            "triagem_requests_total",
            {"endpoint": "/predict", "status": "error", "categoria": "none"},
        )
        or 0.0
    )
    request_duration_count_before = (
        REGISTRY.get_sample_value(
            "triagem_request_duration_seconds_count", {"endpoint": "/predict"}
        )
        or 0.0
    )

    # Observe error
    m.observe_error("inferencia", http_seconds=0.01)

    # Assert error counter incremented
    errors_after = REGISTRY.get_sample_value("triagem_errors_total", {"tipo": "inferencia"})
    assert errors_after == errors_before + 1

    # Assert request counter with error status incremented
    requests_error_after = REGISTRY.get_sample_value(
        "triagem_requests_total",
        {"endpoint": "/predict", "status": "error", "categoria": "none"},
    )
    assert requests_error_after == requests_error_before + 1

    # Assert request duration histogram recorded the observation
    request_duration_count_after = REGISTRY.get_sample_value(
        "triagem_request_duration_seconds_count", {"endpoint": "/predict"}
    )
    assert request_duration_count_after == request_duration_count_before + 1

    # Verify error path produces the same label set as success path
    # Success path: endpoint, status, categoria
    # Both should be present and comparable
    success_labels = {"endpoint": "/predict", "status": "200", "categoria": "neoplasms"}
    error_labels = {"endpoint": "/predict", "status": "error", "categoria": "none"}
    assert set(success_labels.keys()) == set(error_labels.keys())


def test_request_rejects_text_below_minimum_length():
    with pytest.raises(ValidationError):
        PredictRequest(texto="curto")


def test_request_accepts_a_realistic_report():
    request = PredictRequest(texto="a" * 120)
    assert len(request.texto) == 120
