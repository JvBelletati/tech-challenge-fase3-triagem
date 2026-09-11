from triagem import config


def test_label_names_cover_all_five_classes():
    assert set(config.LABEL_NAMES) == {1, 2, 3, 4, 5}
    assert config.LABEL_NAMES[4] == "cardiovascular diseases"


def test_tfidf_params_keep_strip_accents_unset():
    """skl2onnx raises NotImplementedError unless strip_accents is None."""
    assert config.TFIDF_PARAMS.get("strip_accents") is None
    assert config.TFIDF_PARAMS["ngram_range"] == (1, 1)
    assert config.TFIDF_PARAMS["max_features"] == 20_000


def test_confidence_threshold_matches_measured_value():
    assert config.CONFIDENCE_THRESHOLD == 0.40
