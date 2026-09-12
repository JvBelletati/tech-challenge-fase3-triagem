"""Weekly retraining pipeline for the triage model.

This DAG orchestrates the training pipeline implemented in
triagem.training.* (ingest, train, evaluate, export): ingerir_dados,
validar_dados, treinar_modelo, avaliar_modelo and exportar_onnx are each a
thin call into that package, which is exactly what makes the pipeline
itself unit-tested with pytest without ever starting Airflow (see
tests/test_ingest.py, tests/test_train.py, tests/test_evaluate.py and
tests/test_export.py).

promover_modelo and rejeitar_candidato are the exception: promoting or
rejecting a candidate is orchestration-specific (building metadata.json,
copying artifacts into models/current/, deciding whether to skip), so that
logic lives inline in these two tasks rather than in triagem.training.*.
The wiring between tasks, the AirflowSkipException short-circuit in
exportar_onnx, and rejeitar_candidato's trigger_rule="all_done" branch are
therefore not testable by pytest alone - tests/test_dag.py covers what it
can without a scheduler (import, task set, dependency order, catchup) via
DagBag, but the actual skip-propagation behaviour is only exercised by
running the DAG inside Airflow itself.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import UTC, datetime

import pendulum
from airflow.exceptions import AirflowSkipException

# Airflow 3 moved the decorators into the Task SDK. airflow.decorators still
# resolves for now, but airflow.sdk is the supported path on the pinned 3.3.1.
from airflow.sdk import dag, task

from triagem.config import CANDIDATES_DIR, CURRENT_MODEL_DIR, MODEL_FILES
from triagem.training.evaluate import should_promote
from triagem.training.export import assert_parity, export_to_onnx, quantize_model
from triagem.training.ingest import download_dataset, load_dataset, validate_dataset
from triagem.training.train import save_pipeline, train_model

logger = logging.getLogger(__name__)

CANDIDATE_PIPELINE = CANDIDATES_DIR / MODEL_FILES["sklearn"]
CANDIDATE_METRICS = CANDIDATES_DIR / "metrics.json"


@dag(
    dag_id="retreino_triagem",
    description="Ingestao, treino, avaliacao e promocao do classificador de laudos",
    schedule="@weekly",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["mlops", "triagem", "tech-challenge"],
    default_args={"retries": 1},
)
def retreino_triagem():
    @task
    def ingerir_dados() -> str:
        return str(download_dataset())

    @task
    def validar_dados(csv_path: str) -> dict:
        df = load_dataset(csv_path)
        return validate_dataset(df)

    @task
    def treinar_modelo(csv_path: str) -> dict:
        df = load_dataset(csv_path)
        result = train_model(df)
        CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
        save_pipeline(result.pipeline, CANDIDATE_PIPELINE)
        CANDIDATE_METRICS.write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
        # Hold out a slice of the test set so the export task can check parity
        # against the exact same texts the model was evaluated on.
        (CANDIDATES_DIR / "parity_texts.json").write_text(
            json.dumps([str(t) for t in result.X_test[:500]]), encoding="utf-8"
        )
        return result.metrics

    @task
    def avaliar_modelo(candidate_metrics: dict) -> bool:
        current_path = CURRENT_MODEL_DIR / MODEL_FILES["metadata"]
        current = None
        if current_path.exists():
            current = json.loads(current_path.read_text(encoding="utf-8")).get("metrics")

        approved, reason = should_promote(candidate_metrics, current)
        logger.info("promotion gate: %s", reason)
        return approved

    @task
    def exportar_onnx(approved: bool) -> dict:
        if not approved:
            raise AirflowSkipException("candidate rejected by the quality gate")

        import joblib

        pipeline = joblib.load(CANDIDATE_PIPELINE)
        onnx_path = export_to_onnx(pipeline, CANDIDATES_DIR / MODEL_FILES["onnx"])
        quantize_model(onnx_path, CANDIDATES_DIR / MODEL_FILES["onnx_quantized"])

        texts = json.loads((CANDIDATES_DIR / "parity_texts.json").read_text(encoding="utf-8"))
        # assert_parity, not measure_parity: a model whose ONNX export disagrees
        # with the pipeline we actually evaluated must never reach production.
        parity = assert_parity(pipeline, onnx_path, texts)
        return {"onnx_path": str(onnx_path), "parity": parity}

    @task
    def promover_modelo(export_info: dict, metrics: dict) -> str:
        version = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        metadata = {
            "version": version,
            "trained_at": datetime.now(UTC).isoformat(),
            "metrics": metrics,
            "onnx_parity": export_info["parity"],
            "promoted_by": "airflow:retreino_triagem",
        }
        (CANDIDATES_DIR / MODEL_FILES["metadata"]).write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        CURRENT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        for filename in MODEL_FILES.values():
            # Use copy, not copy2: copystat rejects Docker bind mounts. metadata.json
            # already carries version/trained_at, which is sufficient provenance.
            shutil.copy(CANDIDATES_DIR / filename, CURRENT_MODEL_DIR / filename)

        logger.info("promoted model %s", version)
        return version

    @task(trigger_rule="all_done")
    def rejeitar_candidato(approved: bool) -> None:
        """Terminal branch that records a rejected retrain without failing the run."""
        if approved:
            raise AirflowSkipException("candidate was promoted; nothing to reject")
        logger.warning(
            "retraining rejected: candidate underperformed the served model. "
            "Artifacts kept in %s for inspection.",
            CANDIDATES_DIR,
        )

    csv_path = ingerir_dados()
    stats = validar_dados(csv_path)
    metrics = treinar_modelo(csv_path)
    stats >> metrics

    approved = avaliar_modelo(metrics)
    export_info = exportar_onnx(approved)
    promover_modelo(export_info, metrics)
    rejeitar_candidato(approved)


retreino_triagem()
