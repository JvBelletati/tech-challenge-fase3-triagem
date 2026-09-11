"""ONNX-backed inference. Knows nothing about HTTP."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from triagem.config import CURRENT_MODEL_DIR, LABEL_NAMES, MODEL_FILES
from triagem.inference.priority import assign_priority

logger = logging.getLogger(__name__)

INPUT_NAME = "texto"
WARMUP_TEXT = "routine follow up examination with no significant findings reported"


class ModelNotFoundError(FileNotFoundError):
    """Raised when no servable model artifact exists."""


@dataclass(frozen=True)
class Prediction:
    category_id: int
    category: str
    priority: str
    confidence: float
    needs_human_review: bool
    inference_seconds: float


class Predictor:
    """Loads an ONNX model once and reuses the session for every request."""

    def __init__(self, model_dir: Path = CURRENT_MODEL_DIR, variant: str = "onnx") -> None:
        if variant not in ("onnx", "onnx_quantized"):
            raise ValueError(
                f"unsupported variant '{variant}'. accepted values: 'onnx', 'onnx_quantized'"
            )
        self._model_dir = Path(model_dir)
        filename = (
            MODEL_FILES["onnx_quantized"] if variant == "onnx_quantized" else MODEL_FILES["onnx"]
        )
        model_path = self._model_dir / filename

        if not model_path.exists():
            raise ModelNotFoundError(f"no model at {model_path}. Run scripts/treinar.py first.")

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        self._runtime = variant
        self._version = self._read_version()
        logger.info("loaded model %s (%s) from %s", self._version, variant, model_path)

    def _read_version(self) -> str:
        metadata_path = self._model_dir / MODEL_FILES["metadata"]
        if not metadata_path.exists():
            return "unknown"
        return json.loads(metadata_path.read_text(encoding="utf-8")).get("version", "unknown")

    @property
    def version(self) -> str:
        return self._version

    @property
    def runtime(self) -> str:
        return self._runtime

    def predict(self, text: str) -> Prediction:
        """Classify one report and derive its triage priority."""
        payload = np.array([[str(text)]], dtype=object)

        started = time.perf_counter()
        label_output, probability_output = self._session.run(None, {INPUT_NAME: payload})
        elapsed = time.perf_counter() - started

        category_id = int(np.ravel(label_output)[0])
        probabilities = np.asarray(probability_output)[0]
        confidence = float(probabilities.max())
        category = LABEL_NAMES[category_id]
        priority, needs_review = assign_priority(category, confidence)

        return Prediction(
            category_id=category_id,
            category=category,
            priority=priority.value,
            confidence=confidence,
            needs_human_review=needs_review,
            inference_seconds=elapsed,
        )

    def warmup(self, rounds: int = 3) -> None:
        """Run throwaway inferences so the first real request is not an outlier.

        Without this the first request pays for lazy graph initialization and
        shows up as a p99 spike in Grafana that has nothing to do with load.
        """
        for _ in range(rounds):
            self.predict(WARMUP_TEXT)
        logger.info("warmup complete (%d rounds)", rounds)
