import pytest

from triagem.config import LABEL_NAMES
from triagem.inference.priority import (
    CATEGORY_TO_PRIORITY,
    Priority,
    UnknownCategoryError,
    assign_priority,
)


def test_every_known_category_has_a_priority():
    """A category without a mapping would crash in production at request time."""
    assert set(CATEGORY_TO_PRIORITY) == set(LABEL_NAMES.values())


def test_cardiovascular_is_urgent():
    priority, review = assign_priority("cardiovascular diseases", confidence=0.9)
    assert priority is Priority.URGENTE
    assert review is False


def test_low_confidence_never_yields_normal():
    """The core clinical safety rule: an uncertain case is never downgraded."""
    priority, review = assign_priority("general pathological conditions", confidence=0.1)
    assert priority is Priority.ATENCAO
    assert review is True


def test_high_confidence_normal_stays_normal():
    priority, review = assign_priority("general pathological conditions", confidence=0.8)
    assert priority is Priority.NORMAL
    assert review is False


def test_low_confidence_urgent_stays_urgent_but_is_flagged():
    """Escalation only moves upward - it must never demote an urgent case."""
    priority, review = assign_priority("neoplasms", confidence=0.1)
    assert priority is Priority.URGENTE
    assert review is True


def test_confidence_exactly_at_threshold_is_not_review():
    priority, review = assign_priority("general pathological conditions", confidence=0.40)
    assert priority is Priority.NORMAL
    assert review is False


def test_unknown_category_raises():
    with pytest.raises(UnknownCategoryError):
        assign_priority("dermatology", confidence=0.9)
