"""Maps a predicted condition category to a clinical triage priority.

Pure domain logic: no I/O, no model, no framework. Kept separate from the
predictor so the clinical rule can be reviewed and tested on its own.
"""

from __future__ import annotations

from enum import StrEnum

from triagem.config import CONFIDENCE_THRESHOLD


class Priority(StrEnum):
    URGENTE = "URGENTE"
    ATENCAO = "ATENCAO"
    NORMAL = "NORMAL"


class UnknownCategoryError(ValueError):
    """Raised when a category has no priority mapping."""


CATEGORY_TO_PRIORITY: dict[str, Priority] = {
    "neoplasms": Priority.URGENTE,
    "cardiovascular diseases": Priority.URGENTE,
    "nervous system diseases": Priority.ATENCAO,
    "digestive system diseases": Priority.ATENCAO,
    "general pathological conditions": Priority.NORMAL,
}


def assign_priority(
    category: str,
    confidence: float,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> tuple[Priority, bool]:
    """Return the triage priority and whether a human must review the case.

    The error cost here is asymmetric: missing an urgent case can kill a
    patient, while over-escalating one only costs a clinician's time. So a
    low-confidence prediction is never allowed to resolve to NORMAL - it is
    escalated to ATENCAO and flagged for human review.
    """
    try:
        base = CATEGORY_TO_PRIORITY[category]
    except KeyError as exc:
        raise UnknownCategoryError(f"no priority mapping for category {category!r}") from exc

    needs_review = confidence < threshold
    if needs_review and base is Priority.NORMAL:
        return Priority.ATENCAO, True
    return base, needs_review
