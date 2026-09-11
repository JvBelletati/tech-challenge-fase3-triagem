"""Single source of truth for paths, hyperparameters and constants."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = Path(os.getenv("TRIAGEM_MODELS_DIR", PROJECT_ROOT / "models"))
CURRENT_MODEL_DIR = MODELS_DIR / "current"
CANDIDATES_DIR = MODELS_DIR / "candidates"
DATA_DIR = Path(os.getenv("TRIAGEM_DATA_DIR", PROJECT_ROOT / "data"))

DATASET_URL = (
    "https://raw.githubusercontent.com/sebischair/"
    "Medical-Abstracts-TC-Corpus/main/medical_tc_train.csv"
)

TEXT_COLUMN = "medical_abstract"
LABEL_COLUMN = "condition_label"

LABEL_NAMES: dict[int, str] = {
    1: "neoplasms",
    2: "digestive system diseases",
    3: "nervous system diseases",
    4: "cardiovascular diseases",
    5: "general pathological conditions",
}

# Measured on the 2026-09-10 spike: 0.40 routes 8.1% of predictions to human
# review. 0.50 would route 24.7% and 0.60 would route 42.9% - operationally
# unworkable queues.
CONFIDENCE_THRESHOLD = float(os.getenv("TRIAGEM_CONFIDENCE_THRESHOLD", "0.40"))

# WARNING: strip_accents must stay unset. Any other value makes skl2onnx raise
# NotImplementedError and breaks the ONNX export.
TFIDF_PARAMS: dict = {
    "lowercase": True,
    "ngram_range": (1, 1),
    "min_df": 3,
    "max_features": 20_000,
    "sublinear_tf": True,
}

LOGREG_PARAMS: dict = {
    "max_iter": 1000,
    "C": 4.0,
    "class_weight": "balanced",
}

RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_SAMPLES = 2_000
MIN_ONNX_PARITY = 0.98

MODEL_FILES = {
    "onnx": "model.onnx",
    "onnx_quantized": "model_int8.onnx",
    "sklearn": "model.joblib",
    "metadata": "metadata.json",
}

MIN_TEXT_LENGTH = 20
MAX_TEXT_LENGTH = 20_000
