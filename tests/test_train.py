from sklearn.pipeline import Pipeline

from triagem.training.ingest import load_dataset
from triagem.training.train import build_pipeline, save_pipeline, train_model


def test_pipeline_has_named_steps():
    """The ONNX exporter looks up the 'clf' step by name - renaming breaks it."""
    pipe = build_pipeline()
    assert isinstance(pipe, Pipeline)
    assert list(pipe.named_steps) == ["tfidf", "clf"]


def test_pipeline_never_strips_accents():
    """Guards the skl2onnx constraint at the point where it is easiest to break."""
    assert build_pipeline().named_steps["tfidf"].strip_accents is None


def test_train_model_produces_metrics_and_holdout(sample_csv):
    df = load_dataset(sample_csv)
    result = train_model(df, test_size=0.25, random_state=42)
    assert 0.0 <= result.metrics["f1_macro"] <= 1.0
    assert len(result.X_test) == len(result.y_test)
    assert len(result.X_test) > 0


def test_training_is_reproducible(sample_csv):
    df = load_dataset(sample_csv)
    a = train_model(df, test_size=0.25, random_state=42)
    b = train_model(df, test_size=0.25, random_state=42)
    assert a.metrics["f1_macro"] == b.metrics["f1_macro"]


def test_save_pipeline_roundtrip(sample_csv, tmp_path):
    import joblib

    df = load_dataset(sample_csv)
    result = train_model(df, test_size=0.25, random_state=42)
    path = save_pipeline(result.pipeline, tmp_path / "model.joblib")
    assert path.exists()
    reloaded = joblib.load(path)
    assert reloaded.predict([df.iloc[0]["medical_abstract"]])[0] in {1, 2, 3, 4, 5}
