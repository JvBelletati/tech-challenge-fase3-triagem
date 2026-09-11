import numpy as np
import onnxruntime as ort

from triagem.config import MIN_ONNX_PARITY
from triagem.training.export import (
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
    """Documents the measured finding: dynamic int8 does not target LinearClassifier.

    quantize_dynamic rewrites MatMul/Gemm nodes, but skl2onnx emits the
    classifier as ai.onnx.ml LinearClassifier, so the graph comes out
    unchanged. This test pins that fact so the README claim stays true.
    """
    df = load_dataset(sample_csv)
    result = train_model(df, test_size=0.25, random_state=42)
    src = export_to_onnx(result.pipeline, tmp_path / "model.onnx")
    dst = quantize_model(src, tmp_path / "model_int8.onnx")

    assert count_graph_ops(src) == count_graph_ops(dst)
    assert "LinearClassifier" in count_graph_ops(src)
