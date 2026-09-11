import pytest

from triagem.config import CURRENT_MODEL_DIR
from triagem.inference.predictor import ModelNotFoundError, Predictor

pytestmark = pytest.mark.skipif(
    not (CURRENT_MODEL_DIR / "model.onnx").exists(),
    reason="requires a trained model in models/current (run scripts/treinar.py)",
)

LAUDO = (
    "The patient presented with acute chest pain radiating to the left arm, "
    "accompanied by dyspnea and diaphoresis. ECG showed ST elevation."
)


@pytest.fixture(scope="module")
def predictor():
    return Predictor()


def test_predict_returns_a_complete_prediction(predictor):
    result = predictor.predict(LAUDO)
    assert result.category_id in {1, 2, 3, 4, 5}
    assert result.priority in {"URGENTE", "ATENCAO", "NORMAL"}
    assert 0.0 <= result.confidence <= 1.0
    assert result.inference_seconds > 0


def test_predict_is_deterministic(predictor):
    a, b = predictor.predict(LAUDO), predictor.predict(LAUDO)
    assert (a.category_id, a.confidence) == (b.category_id, b.confidence)


def test_warmup_does_not_raise(predictor):
    predictor.warmup(rounds=2)


def test_missing_model_directory_raises(tmp_path):
    with pytest.raises(ModelNotFoundError):
        Predictor(model_dir=tmp_path / "nope")


def test_unsupported_variant_raises():
    with pytest.raises(ValueError, match="unsupported variant"):
        Predictor(variant="onnx-int8")


def test_quantized_variant_loads_and_predicts():
    predictor_quantized = Predictor(variant="onnx_quantized")
    result = predictor_quantized.predict(LAUDO)
    assert result.category_id in {1, 2, 3, 4, 5}
    assert result.priority in {"URGENTE", "ATENCAO", "NORMAL"}
    assert 0.0 <= result.confidence <= 1.0
    assert result.inference_seconds > 0
