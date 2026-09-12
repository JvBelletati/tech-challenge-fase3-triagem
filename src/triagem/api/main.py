"""FastAPI application. Knows nothing about ONNX - it only calls the predictor."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from triagem.api import metrics
from triagem.api.schemas import HealthResponse, ModelInfo, PredictRequest, PredictResponse
from triagem.inference.predictor import ModelNotFoundError, Predictor

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_predictor: Predictor | None = None


def get_predictor() -> Predictor | None:
    return _predictor


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load and warm the model once, at startup - never per request."""
    global _predictor
    variant = os.getenv("TRIAGEM_MODEL_VARIANT", "onnx")
    try:
        _predictor = Predictor(variant=variant)
        _predictor.warmup()
        metrics.set_model_info(_predictor.version, _predictor.runtime)
        logger.info("api ready, serving model %s", _predictor.version)
    except ModelNotFoundError:
        # Start anyway so /health can report the failure. A container that
        # refuses to boot tells an orchestrator nothing useful.
        logger.exception("no model available; /health will report unavailable")
        _predictor = None
    yield
    _predictor = None


app = FastAPI(
    title="Triagem de Laudos Medicos",
    description="Classifica laudos medicos e atribui prioridade de triagem clinica.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError) -> Response:
    """Count malformed requests as errors without changing FastAPI's 422 body.

    A too-short or missing `texto` fails Pydantic validation before the
    `/predict` handler body ever runs, so without this hook the error-rate
    panel's numerator (triagem_requests_total{status="error"}) would never
    move no matter how many malformed reports arrive. `tipo="validacao"`
    keeps this class of failure distinguishable in triagem_errors_total from
    genuine inference or model-availability errors.
    """
    metrics.observe_error("validacao", None, endpoint=request.url.path)
    return await request_validation_exception_handler(request, exc)


@app.get("/health", response_model=HealthResponse)
def health(response: Response) -> HealthResponse:
    predictor = get_predictor()
    if predictor is None:
        response.status_code = 503
        return HealthResponse(status="modelo indisponivel", modelo=None)
    return HealthResponse(
        status="ok",
        modelo=ModelInfo(versao=predictor.version, runtime=predictor.runtime),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    predictor = get_predictor()
    if predictor is None:
        metrics.observe_error("modelo_indisponivel", None)
        raise HTTPException(status_code=503, detail="modelo indisponivel")

    started = time.perf_counter()
    with metrics.IN_PROGRESS.track_inprogress():
        try:
            prediction = predictor.predict(request.texto)
        except Exception:
            metrics.observe_error("inferencia", time.perf_counter() - started)
            logger.exception("inference failed")
            raise HTTPException(status_code=500, detail="falha na inferencia") from None

    elapsed = time.perf_counter() - started
    metrics.observe_prediction(prediction, elapsed)

    return PredictResponse(
        categoria=prediction.category,
        categoria_id=prediction.category_id,
        prioridade=prediction.priority,
        confianca=round(prediction.confidence, 4),
        revisao_humana=prediction.needs_human_review,
        latencia_ms=round(elapsed * 1000, 3),
        modelo=ModelInfo(versao=predictor.version, runtime=predictor.runtime),
    )


@app.get("/metrics")
def prometheus_metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
