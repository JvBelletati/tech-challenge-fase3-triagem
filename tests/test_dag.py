import pytest

try:
    from airflow.models import DagBag
except ImportError:
    pytest.skip(
        "Airflow is optional outside the DAG-validation job",
        allow_module_level=True,
    )
    DagBag = None

EXPECTED_TASKS = {
    "ingerir_dados",
    "validar_dados",
    "treinar_modelo",
    "avaliar_modelo",
    "exportar_onnx",
    "promover_modelo",
    "rejeitar_candidato",
}


@pytest.fixture(scope="module")
def dagbag():
    return DagBag("airflow/dags", include_examples=False)


def test_dag_imports_without_errors(dagbag):
    assert dagbag.import_errors == {}


def test_dag_exposes_every_expected_task(dagbag):
    dag = dagbag.get_dag("retreino_triagem")
    assert dag is not None
    assert {task.task_id for task in dag.tasks} == EXPECTED_TASKS


def test_pipeline_order_is_ingest_then_validate_then_train(dagbag):
    dag = dagbag.get_dag("retreino_triagem")
    assert dag.get_task("validar_dados").upstream_task_ids == {"ingerir_dados"}
    # treinar_modelo consumes the CSV path from ingerir_dados AND is gated on
    # validar_dados, so it has two upstreams - not one.
    assert dag.get_task("treinar_modelo").upstream_task_ids == {
        "ingerir_dados",
        "validar_dados",
    }


def test_catchup_is_disabled(dagbag):
    """Catchup on a weekly retrain would fire a backlog of pointless runs."""
    assert dagbag.get_dag("retreino_triagem").catchup is False
