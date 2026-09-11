"""Convert the sklearn pipeline to ONNX, quantize it and verify parity."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType

from triagem.config import MIN_ONNX_PARITY

logger = logging.getLogger(__name__)

INPUT_NAME = "texto"


class ParityError(RuntimeError):
    """Raised when the ONNX model disagrees with sklearn too often."""


def _session(path: Path) -> ort.InferenceSession:
    """Build an inference session tuned for single-report latency.

    intra_op_num_threads=1 is deliberate: with a one-row payload the thread
    coordination costs more than the parallelism returns.
    """
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def export_to_onnx(pipeline, path: Path) -> Path:
    """Export the fitted pipeline (vectorizer included) to ONNX.

    zipmap=False makes the classifier emit a plain probability tensor instead
    of a sequence of dictionaries, which is both faster and far simpler to
    consume at request time.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model = to_onnx(
        pipeline,
        initial_types=[(INPUT_NAME, StringTensorType([None, 1]))],
        options={id(pipeline.named_steps["clf"]): {"zipmap": False}},
    )
    path.write_bytes(model.SerializeToString())
    logger.info("exported ONNX model to %s (%.2f MB)", path, path.stat().st_size / 1e6)
    return path


def quantize_model(src: Path, dst: Path) -> Path:
    """Apply dynamic int8 quantization.

    Measured outcome on this pipeline: no change. Kept because the brief asks
    for the technique to be applied and compared, and the negative result is
    itself a finding worth reporting - see count_graph_ops.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QUInt8)
    logger.info("quantized model written to %s (%.2f MB)", dst, dst.stat().st_size / 1e6)
    return dst


def count_graph_ops(path: Path) -> dict[str, int]:
    """Count operator types in an ONNX graph. Used to evidence quantization effects."""
    model = onnx.load(str(path))
    return dict(Counter(node.op_type for node in model.graph.node))


def measure_parity(pipeline, onnx_path: Path, texts: list[str]) -> float:
    """Fraction of inputs where ONNX and sklearn agree on the predicted label."""
    if not texts:
        raise ValueError("texts must not be empty")

    session = _session(onnx_path)
    batch = np.array([str(t) for t in texts], dtype=object).reshape(-1, 1)
    onnx_predictions = np.asarray(session.run(None, {INPUT_NAME: batch})[0]).ravel()
    sklearn_predictions = np.asarray(pipeline.predict(list(texts))).ravel()

    agreement = float((onnx_predictions == sklearn_predictions).mean())
    logger.info("onnx/sklearn agreement: %.4f", agreement)
    return agreement


def assert_parity(pipeline, onnx_path: Path, texts: list[str]) -> float:
    """Raise ParityError if agreement falls below the configured floor."""
    agreement = measure_parity(pipeline, onnx_path, texts)
    if agreement < MIN_ONNX_PARITY:
        raise ParityError(f"onnx agreement {agreement:.4f} below minimum {MIN_ONNX_PARITY}")
    return agreement
