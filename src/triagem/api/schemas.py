"""Pydantic contracts. Field names are in Portuguese: the client is a Brazilian hospital."""

from __future__ import annotations

from pydantic import BaseModel, Field

from triagem.config import MAX_TEXT_LENGTH, MIN_TEXT_LENGTH


class PredictRequest(BaseModel):
    texto: str = Field(
        ...,
        min_length=MIN_TEXT_LENGTH,
        max_length=MAX_TEXT_LENGTH,
        description="Texto livre do laudo medico a ser triado",
    )


class ModelInfo(BaseModel):
    versao: str
    runtime: str


class PredictResponse(BaseModel):
    categoria: str
    categoria_id: int
    prioridade: str
    confianca: float
    revisao_humana: bool
    latencia_ms: float
    modelo: ModelInfo


class HealthResponse(BaseModel):
    status: str
    modelo: ModelInfo | None = None
