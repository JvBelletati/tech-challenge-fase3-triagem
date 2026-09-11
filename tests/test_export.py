import numpy as np
import onnxruntime as ort
import pytest

from triagem.config import MIN_ONNX_PARITY
from triagem.training.export import (
    ParityError,
    assert_parity,
    count_graph_ops,
    export_to_onnx,
    measure_parity,
    quantize_model,
)
from triagem.training.ingest import load_dataset
from triagem.training.train import train_model


def test_export_produces_a_loadable_onnx_model(sample_csv, tmp_path):
    df = load_dataset(sample_csv)
    result = train_model(df, test_size=0.25, random_state=42)
    path = export_to_onnx(result.pipeline, tmp_path / "model.onnx")

    assert path.exists()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].name == "texto"
    assert [o.name for o in session.get_outputs()] == ["label", "probabilities"]


def test_onnx_predictions_match_sklearn(sample_csv, tmp_path):
    """Exact equality is wrong here: the ONNX tokenizer is not bit-identical."""
    df = load_dataset(sample_csv)
    result = train_model(df, test_size=0.25, random_state=42)
    path = export_to_onnx(result.pipeline, tmp_path / "model.onnx")

    agreement = measure_parity(result.pipeline, path, list(result.X_test))
    assert agreement >= MIN_ONNX_PARITY


def test_quantization_produces_a_working_model(sample_csv, tmp_path):
    df = load_dataset(sample_csv)
    result = train_model(df, test_size=0.25, random_state=42)
    src = export_to_onnx(result.pipeline, tmp_path / "model.onnx")
    dst = quantize_model(src, tmp_path / "model_int8.onnx")

    assert dst.exists()
    session = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    out = session.run(None, {"texto": np.array([["chest pain and dyspnea"]], dtype=object)})
    assert int(np.ravel(out[0])[0]) in {1, 2, 3, 4, 5}


def test_quantization_leaves_the_classifier_untouched(sample_csv, tmp_path):
    """Documents the measured finding: dynamic int8 quantization never targets the classifier.

    The assertion compares operator-type histograms (a Counter of
    node.op_type), not byte-for-byte file contents - the two ONNX files do
    differ slightly in size because quantize_dynamic still rewrites some
    bookkeeping. What the histogram equality proves is narrower and is
    exactly what matters: no QuantizeLinear or MatMulInteger node appears in
    the quantized graph, because quantize_dynamic only rewrites MatMul/Gemm
    nodes and skl2onnx emits the classifier as an ai.onnx.ml
    LinearClassifier, which neither of those ops targets. This test pins
    that fact so the README claim stays accurate.
    """
    df = load_dataset(sample_csv)
    result = train_model(df, test_size=0.25, random_state=42)
    src = export_to_onnx(result.pipeline, tmp_path / "model.onnx")
    dst = quantize_model(src, tmp_path / "model_int8.onnx")

    assert count_graph_ops(src) == count_graph_ops(dst)
    assert "LinearClassifier" in count_graph_ops(src)


def test_assert_parity_raises_when_agreement_is_too_low(sample_csv, tmp_path):
    """Covers the failure path that gates production promotion (Tasks 6 and 12).

    Wraps the fitted pipeline with a stand-in whose predict() always returns
    an impossible label (-1), so it can never agree with the ONNX model's
    predictions (which are always in {1..5}). That forces 0.0 agreement,
    unambiguously below MIN_ONNX_PARITY, so this exercises the ParityError
    branch instead of only ever hitting the happy path.
    """
    df = load_dataset(sample_csv)
    result = train_model(df, test_size=0.25, random_state=42)
    path = export_to_onnx(result.pipeline, tmp_path / "model.onnx")

    class AlwaysDisagreeingPipeline:
        def predict(self, X):
            return np.full(len(list(X)), -1)

    with pytest.raises(ParityError):
        assert_parity(AlwaysDisagreeingPipeline(), path, list(result.X_test))


def test_assert_parity_returns_the_agreement_fraction_on_the_happy_path(sample_csv, tmp_path):
    df = load_dataset(sample_csv)
    result = train_model(df, test_size=0.25, random_state=42)
    path = export_to_onnx(result.pipeline, tmp_path / "model.onnx")

    agreement = assert_parity(result.pipeline, path, list(result.X_test))

    assert isinstance(agreement, float)
    assert agreement >= MIN_ONNX_PARITY
