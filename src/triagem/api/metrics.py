"""Prometheus instrumentation, isolated from routing and from inference.

The key decision here is separating inference time from HTTP time. Without
that split, the ONNX speedup is buried under framework overhead and the
Stage 4 latency comparison becomes unmeasurable from the dashboard.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Info

from triagem.inference.predictor import Prediction

# Buckets are tuned for a model that answers in well under a millisecond.
# The default prometheus_client buckets start at 5ms and would place every
# single request in the first bucket, making the histogram useless.
LATENCY_BUCKETS = (
    0.0005,
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
)

REQUESTS = Counter(
    "triagem_requests_total",
    "Total de requisicoes processadas",
    ["endpoint", "status", "categoria"],
)

REQUEST_DURATION = Histogram(
    "triagem_request_duration_seconds",
    "Latencia HTTP ponta a ponta",
    ["endpoint"],
    buckets=LATENCY_BUCKETS,
)

INFERENCE_DURATION = Histogram(
    "triagem_inference_duration_seconds",
    "Latencia apenas do modelo, sem overhead HTTP",
    buckets=LATENCY_BUCKETS,
)

ERRORS = Counter(
    "triagem_errors_total",
    "Total de erros por tipo",
    ["tipo"],
)

CONFIDENCE = Histogram(
    "triagem_prediction_confidence",
    "Distribuicao da confianca das predicoes",
    buckets=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

IN_PROGRESS = Gauge(
    "triagem_requests_in_progress",
    "Requisicoes sendo processadas neste momento",
)

MODEL_INFO = Info("triagem_model", "Modelo atualmente servido")


def set_model_info(version: str, runtime: str) -> None:
    MODEL_INFO.info({"versao": version, "runtime": runtime})


def observe_prediction(prediction: Prediction, http_seconds: float) -> None:
    """Record one successful prediction across every relevant metric."""
    REQUESTS.labels(endpoint="/predict", status="200", categoria=prediction.category).inc()
    REQUEST_DURATION.labels(endpoint="/predict").observe(http_seconds)
    INFERENCE_DURATION.observe(prediction.inference_seconds)
    CONFIDENCE.observe(prediction.confidence)


def observe_error(error_type: str, http_seconds: float | None, endpoint: str = "/predict") -> None:
    """Record one failed request.

    `http_seconds` is `None` when the request never reached a handler body -
    e.g. one rejected by Pydantic validation before `/predict` runs. That
    request has no service latency worth reporting, so we skip the duration
    histogram rather than feed it an invented zero that would drag the
    p50/p95/p99 latency panel toward zero. The request and error counters
    still need the observation regardless: that is what the error-rate panel
    counts. `endpoint` defaults to "/predict" so existing call sites (which
    only ever fail on that route today) keep working unchanged; pass the
    actual path explicitly from handlers, like the validation one, that can
    fire on any route.
    """
    ERRORS.labels(tipo=error_type).inc()
    REQUESTS.labels(endpoint=endpoint, status="error", categoria="none").inc()
    if http_seconds is not None:
        REQUEST_DURATION.labels(endpoint=endpoint).observe(http_seconds)
