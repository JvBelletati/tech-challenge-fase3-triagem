"""Dataset download, loading and validation.

The validation step is what turns a retraining DAG from 'runs' into 'defends
production': a truncated download or a dropped class fails the run instead of
silently producing a worse model.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import pandas as pd

from triagem.config import (
    DATA_DIR,
    DATASET_URL,
    LABEL_COLUMN,
    LABEL_NAMES,
    MIN_SAMPLES,
    TEXT_COLUMN,
)

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Raised when the dataset is unfit for training."""


def download_dataset(
    url: str = DATASET_URL,
    dest: Path | None = None,
    force: bool = False,
) -> Path:
    """Download the corpus, caching it on disk.

    Cached by default so repeated DAG runs and local experiments do not
    re-fetch 14 MB every time.
    """
    dest = dest or (DATA_DIR / "raw" / "medical_tc_train.csv")
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        logger.info("dataset already cached at %s", dest)
        return dest

    logger.info("downloading dataset from %s", url)
    with urllib.request.urlopen(url, timeout=120) as response:
        dest.write_bytes(response.read())
    logger.info("saved %.1f MB to %s", dest.stat().st_size / 1e6, dest)
    return dest


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the CSV into a typed, clean two-column frame."""
    df = pd.read_csv(path)

    missing = {TEXT_COLUMN, LABEL_COLUMN} - set(df.columns)
    if missing:
        raise DataValidationError(f"missing expected columns: {sorted(missing)}")

    df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.strip()
    df = df[df[TEXT_COLUMN] != ""]
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    return df.reset_index(drop=True)


def validate_dataset(df: pd.DataFrame, min_samples: int = MIN_SAMPLES) -> dict:
    """Check the frame is fit for training. Raises DataValidationError if not."""
    n_samples = len(df)
    if n_samples < min_samples:
        raise DataValidationError(f"got {n_samples} rows, min_samples={min_samples} required")

    counts = df[LABEL_COLUMN].value_counts().to_dict()
    expected = set(LABEL_NAMES)
    if set(counts) != expected:
        raise DataValidationError(f"expected classes {sorted(expected)}, found {sorted(counts)}")

    stats = {
        "n_samples": n_samples,
        "n_classes": len(counts),
        "class_counts": {int(k): int(v) for k, v in sorted(counts.items())},
        "mean_text_length": float(df[TEXT_COLUMN].str.len().mean()),
    }
    logger.info("dataset validated: %s", stats)
    return stats
