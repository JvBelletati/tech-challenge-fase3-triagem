from triagem.training.evaluate import should_promote


def test_promotes_when_no_current_model():
    ok, reason = should_promote({"f1_macro": 0.1}, None)
    assert ok is True
    assert "no current model" in reason


def test_promotes_when_candidate_is_better():
    ok, _ = should_promote({"f1_macro": 0.60}, {"f1_macro": 0.55})
    assert ok is True


def test_rejects_when_candidate_is_worse():
    """Without this gate the DAG happily ships a regression to production."""
    ok, reason = should_promote({"f1_macro": 0.50}, {"f1_macro": 0.55})
    assert ok is False
    assert "0.50" in reason and "0.55" in reason


def test_equal_scores_are_promoted():
    ok, _ = should_promote({"f1_macro": 0.55}, {"f1_macro": 0.55})
    assert ok is True


def test_tolerance_allows_small_regression():
    ok, _ = should_promote({"f1_macro": 0.549}, {"f1_macro": 0.550}, tolerance=0.005)
    assert ok is True
