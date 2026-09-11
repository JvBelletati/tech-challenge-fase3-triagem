from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from triagem.training.ingest import (
    DataValidationError,
    download_dataset,
    load_dataset,
    validate_dataset,
)


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


def test_download_dataset_creates_file_and_parents_on_first_call(tmp_path):
    """First call writes file and creates missing parent directories."""
    dest = tmp_path / "subdir" / "nested" / "data.csv"
    test_content = b"condition_label,medical_abstract\n1,test\n"

    with patch("triagem.training.ingest.urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = test_content
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None
        mock_urlopen.return_value = mock_response

        result = download_dataset(url="http://example.com/data.csv", dest=dest)

        assert result == dest
        assert dest.exists()
        assert dest.read_bytes() == test_content
        mock_urlopen.assert_called_once()


def test_download_dataset_uses_cache_when_file_exists(tmp_path):
    """Second call does NOT re-download when destination exists."""
    dest = tmp_path / "data.csv"
    original_content = b"original data"
    dest.write_bytes(original_content)

    with patch("triagem.training.ingest.urllib.request.urlopen") as mock_urlopen:
        result = download_dataset(url="http://example.com/data.csv", dest=dest)

        assert result == dest
        assert dest.read_bytes() == original_content
        mock_urlopen.assert_not_called()


def test_download_dataset_force_redownloads_existing_file(tmp_path):
    """force=True re-downloads even when file exists."""
    dest = tmp_path / "data.csv"
    old_content = b"old data"
    new_content = b"new data from server"
    dest.write_bytes(old_content)

    with patch("triagem.training.ingest.urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = new_content
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None
        mock_urlopen.return_value = mock_response

        result = download_dataset(url="http://example.com/data.csv", dest=dest, force=True)

        assert result == dest
        assert dest.read_bytes() == new_content
        mock_urlopen.assert_called_once()
