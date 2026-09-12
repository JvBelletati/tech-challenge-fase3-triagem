from triagem.training.benchmark import LatencyResult, format_report, measure_latency


def test_measure_latency_returns_ordered_percentiles():
    stats = measure_latency(lambda text: len(text), ["abc"] * 50, rounds=50, warmup=5)
    assert stats["p50_ms"] <= stats["p95_ms"] <= stats["p99_ms"]
    assert stats["mean_ms"] > 0


def test_measure_latency_rejects_empty_texts():
    import pytest

    with pytest.raises(ValueError):
        measure_latency(lambda text: text, [], rounds=10, warmup=1)


def test_format_report_renders_every_variant_and_the_speedup():
    results = [
        LatencyResult("sklearn", 0.615, 0.783, 0.853, 0.650, 0.5589, 1.20),
        LatencyResult("onnx", 0.188, 0.257, 0.294, 0.200, 0.5591, 0.80),
    ]
    report = format_report(results, {"sklearn": {}, "onnx": {"LinearClassifier": 1}})
    assert "sklearn" in report and "onnx" in report
    assert "3.27" in report  # 0.615 / 0.188
