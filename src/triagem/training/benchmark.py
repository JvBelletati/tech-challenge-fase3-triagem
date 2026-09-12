"""Latency and quality comparison across model variants.

Reports latency and F1 side by side on purpose: an optimization that degrades
accuracy is not an optimization, and a latency table without the quality
column cannot tell the difference.
"""

from __future__ import annotations

import logging
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LatencyResult:
    variant: str
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    f1_macro: float | None
    size_mb: float


def measure_latency(
    predict_fn: Callable[[str], object],
    texts: Sequence[str],
    rounds: int = 400,
    warmup: int = 50,
) -> dict[str, float]:
    """Time single-input predictions, which is what the API actually does.

    Batch throughput would flatter every variant equally and tell us nothing
    about the per-request latency the hospital experiences.
    """
    if not texts:
        raise ValueError("texts must not be empty")

    for i in range(warmup):
        predict_fn(texts[i % len(texts)])

    samples: list[float] = []
    for i in range(rounds):
        started = time.perf_counter()
        predict_fn(texts[i % len(texts)])
        samples.append((time.perf_counter() - started) * 1000)

    array = np.array(samples)
    return {
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "mean_ms": float(statistics.mean(samples)),
    }


def _size_mb(path: Path) -> float:
    return path.stat().st_size / 1e6 if path.exists() else 0.0


def benchmark_variants(
    model_dir: Path,
    texts: Sequence[str],
    labels: Sequence[int] | None = None,
    rounds: int = 400,
) -> list[LatencyResult]:
    """Benchmark sklearn, ONNX fp32 and ONNX int8 on the same inputs."""
    import joblib
    import onnxruntime as ort
    from sklearn.metrics import f1_score

    from triagem.config import MODEL_FILES

    model_dir = Path(model_dir)
    results: list[LatencyResult] = []

    def f1_for(predictions) -> float | None:
        if labels is None:
            return None
        return float(f1_score(labels, predictions, average="macro", zero_division=0))

    sklearn_path = model_dir / MODEL_FILES["sklearn"]
    pipeline = joblib.load(sklearn_path)
    stats = measure_latency(lambda text: pipeline.predict([text]), texts, rounds)
    results.append(
        LatencyResult(
            "sklearn",
            **stats,
            f1_macro=f1_for(pipeline.predict(list(texts))),
            size_mb=_size_mb(sklearn_path),
        )
    )

    for variant, key in (("onnx-fp32", "onnx"), ("onnx-int8", "onnx_quantized")):
        path = model_dir / MODEL_FILES[key]
        if not path.exists():
            logger.warning("skipping %s: %s not found", variant, path)
            continue

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

        stats = measure_latency(
            lambda text, s=session: s.run(None, {"texto": np.array([[text]], dtype=object)}),
            texts,
            rounds,
        )
        batch = np.array([str(t) for t in texts], dtype=object).reshape(-1, 1)
        predictions = np.asarray(session.run(None, {"texto": batch})[0]).ravel()
        results.append(
            LatencyResult(variant, **stats, f1_macro=f1_for(predictions), size_mb=_size_mb(path))
        )

    return results


def format_report(results: list[LatencyResult], graph_ops: dict[str, dict]) -> str:
    """Render the comparison as Markdown, including the quantization finding."""
    baseline = results[0]
    lines = [
        (
            "| Variante | p50 (ms) | p95 (ms) | p99 (ms) | F1-macro | "
            "Tamanho (MB) | Ganho vs baseline |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for result in results:
        speedup = baseline.p50_ms / result.p50_ms if result.p50_ms else float("nan")
        f1_text = f"{result.f1_macro:.4f}" if result.f1_macro is not None else "-"
        lines.append(
            f"| {result.variant} | {result.p50_ms:.3f} | {result.p95_ms:.3f} | "
            f"{result.p99_ms:.3f} | {f1_text} | {result.size_mb:.2f} | {speedup:.2f}x |"
        )

    lines.append("")
    lines.append("### Operadores do grafo ONNX")
    lines.append("")
    for name, ops in graph_ops.items():
        lines.append(f"- **{name}**: `{ops}`")

    return "\n".join(lines)
