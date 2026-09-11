import pandas as pd
import pytest

from triagem.training.ingest import DataValidationError, load_dataset, validate_dataset


def test_load_dataset_returns_typed_columns(sample_csv):
    df = load_dataset(sample_csv)
    assert list(df.columns) == ["medical_abstract", "condition_label"]
    assert df["condition_label"].dtype.kind == "i"
    assert df["medical_abstract"].map(type).eq(str).all()


def test_load_dataset_drops_rows_with_missing_text(tmp_path):
    csv = tmp_path / "holes.csv"
    csv.write_text("condition_label,medical_abstract\n1,texto valido aqui\n2,\n", encoding="utf-8")
    df = load_dataset(csv)
    assert len(df) == 1


def test_validate_dataset_returns_stats(sample_csv):
    df = load_dataset(sample_csv)
    stats = validate_dataset(df, min_samples=100)
    assert stats["n_samples"] == 120
    assert stats["n_classes"] == 5
    assert set(stats["class_counts"]) == {1, 2, 3, 4, 5}


def test_validate_dataset_rejects_too_few_samples(sample_csv):
    """The brief requires at least 2000 samples - a truncated download must fail loudly."""
    df = load_dataset(sample_csv)
    with pytest.raises(DataValidationError, match="min_samples"):
        validate_dataset(df, min_samples=2_000)


def test_validate_dataset_rejects_missing_class():
    df = pd.DataFrame({"medical_abstract": ["a" * 30] * 10, "condition_label": [1] * 10})
    with pytest.raises(DataValidationError, match="classes"):
        validate_dataset(df, min_samples=5)
