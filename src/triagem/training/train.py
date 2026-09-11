"""Training pipeline. Called by the Airflow DAG and by scripts/treinar.py."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from triagem.config import (
    LABEL_COLUMN,
    LOGREG_PARAMS,
    RANDOM_STATE,
    TEST_SIZE,
    TEXT_COLUMN,
    TFIDF_PARAMS,
)
from triagem.training.evaluate import evaluate_model

logger = logging.getLogger(__name__)


@dataclass
class TrainedModel:
    pipeline: Pipeline
    metrics: dict
    X_test: np.ndarray
    y_test: np.ndarray


def build_pipeline() -> Pipeline:
    """Build the TF-IDF + logistic regression pipeline.

    Unigrams beat bigrams on this corpus on both axes at once - 22% faster and
    1.4 points more accurate (measured, see the spec's spike appendix). The
    step names matter: the ONNX exporter looks up 'clf' by name.
    """
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
            ("clf", LogisticRegression(random_state=RANDOM_STATE, **LOGREG_PARAMS)),
        ]
    )


def train_model(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> TrainedModel:
    """Train on a stratified split and evaluate on the holdout."""
    X = df[TEXT_COLUMN].to_numpy(dtype=object)
    y = df[LABEL_COLUMN].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = build_pipeline()
    logger.info("training on %d samples", len(X_train))
    pipeline.fit(X_train, y_train)

    metrics = evaluate_model(pipeline, X_test, y_test)
    logger.info("f1_macro=%.4f accuracy=%.4f", metrics["f1_macro"], metrics["accuracy"])

    return TrainedModel(pipeline=pipeline, metrics=metrics, X_test=X_test, y_test=y_test)


def save_pipeline(pipeline: Pipeline, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info("saved sklearn pipeline to %s (%.2f MB)", path, path.stat().st_size / 1e6)
    return path
