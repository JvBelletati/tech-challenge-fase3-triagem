"""End-to-end training CLI: download, validate, train, export, promote.

Usage:
    python scripts/treinar.py                 # full dataset
    python scripts/treinar.py --sample 2000   # quick baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import UTC, datetime

import sklearn

from triagem.config import CANDIDATES_DIR, CURRENT_MODEL_DIR, MODEL_FILES
from triagem.training.evaluate import should_promote
from triagem.training.export import assert_parity, export_to_onnx, quantize_model
from triagem.training.ingest import download_dataset, load_dataset, validate_dataset
from triagem.training.train import save_pipeline, train_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("treinar")


def read_current_metrics() -> dict | None:
    path = CURRENT_MODEL_DIR / MODEL_FILES["metadata"]
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("metrics")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="train on N rows only")
    parser.add_argument("--min-samples", type=int, default=2_000)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    csv_path = download_dataset(force=args.force_download)
    df = load_dataset(csv_path)
    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=42).reset_index(drop=True)
    stats = validate_dataset(df, min_samples=min(args.min_samples, len(df)))

    result = train_model(df)

    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    save_pipeline(result.pipeline, CANDIDATES_DIR / MODEL_FILES["sklearn"])
    onnx_path = export_to_onnx(result.pipeline, CANDIDATES_DIR / MODEL_FILES["onnx"])
    quantize_model(onnx_path, CANDIDATES_DIR / MODEL_FILES["onnx_quantized"])
    parity = assert_parity(result.pipeline, onnx_path, list(result.X_test))

    metadata = {
        "version": datetime.now(UTC).strftime("%Y%m%d-%H%M%S"),
        "trained_at": datetime.now(UTC).isoformat(),
        "metrics": result.metrics,
        "onnx_parity": parity,
        "n_samples": stats["n_samples"],
        "sklearn_version": sklearn.__version__,
    }
    (CANDIDATES_DIR / MODEL_FILES["metadata"]).write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    promote, reason = should_promote(result.metrics, read_current_metrics())
    logger.info("promotion decision: %s (%s)", promote, reason)
    if not promote:
        logger.warning("candidate kept in %s, production untouched", CANDIDATES_DIR)
        return 1

    CURRENT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for filename in MODEL_FILES.values():
        # Use copy, not copy2: copystat rejects Docker bind mounts. metadata.json
        # already carries version/trained_at, which is sufficient provenance.
        shutil.copy(CANDIDATES_DIR / filename, CURRENT_MODEL_DIR / filename)
    logger.info("promoted model version %s to %s", metadata["version"], CURRENT_MODEL_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
