"""Model evaluation and the promotion gate."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

from triagem.config import LABEL_NAMES

logger = logging.getLogger(__name__)


def evaluate_model(pipeline, X: np.ndarray, y: np.ndarray) -> dict:
    """Compute holdout metrics for a fitted pipeline."""
    predictions = pipeline.predict(X)
    report = classification_report(
        y,
        predictions,
        labels=sorted(LABEL_NAMES),
        target_names=[LABEL_NAMES[k] for k in sorted(LABEL_NAMES)],
        output_dict=True,
        zero_division=0,
    )
    return {
        "f1_macro": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, predictions)),
        "n_test": int(len(y)),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in (LABEL_NAMES[k] for k in sorted(LABEL_NAMES))
        },
    }


def should_promote(
    candidate: dict,
    current: dict | None,
    tolerance: float = 0.0,
) -> tuple[bool, str]:
    """Decide whether a freshly trained model may replace the served one.

    A retraining run that degrades quality must not reach production. The
    tolerance exists so that pure noise between runs does not block an
    otherwise healthy retrain.
    """
    new_score = float(candidate["f1_macro"])

    if current is None:
        return True, f"no current model to compare against (candidate f1={new_score:.4f})"

    old_score = float(current["f1_macro"])
    if new_score >= old_score - tolerance:
        return True, f"candidate f1={new_score:.4f} >= current f1={old_score:.4f}"

    return False, (
        f"rejected: candidate f1={new_score:.4f} below current f1={old_score:.4f} "
        f"(tolerance={tolerance})"
    )
