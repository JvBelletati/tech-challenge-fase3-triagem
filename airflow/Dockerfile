FROM apache/airflow:3.3.1

# Same trap as the API image: apache/airflow:3.3.1 (Debian bookworm) ships
# with no en_US.UTF-8 locale (only C, C.utf8, POSIX are available). The
# exportar_onnx task calls assert_parity, which opens an onnxruntime
# inference session - the StringNormalizer op used by the exported TF-IDF
# vectorizer crashes at session init without a real en_US.UTF-8 locale. The
# base image's default user is airflow (uid 50000), not root, so apt-get
# needs an explicit switch to root and back.
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends locales \
    && sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen \
    && locale-gen \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# The DAG imports triagem.training.*, so the package must exist inside the
# Airflow image. Installing it beats _PIP_ADDITIONAL_REQUIREMENTS, which
# reinstalls on every container start.
COPY --chown=airflow:root pyproject.toml README.md /opt/triagem/
COPY --chown=airflow:root src/ /opt/triagem/src/

USER airflow
RUN pip install --no-cache-dir "/opt/triagem[training]" \
    # triagem's base dependencies pin fastapi==0.141.1 for the API service,
    # which pip happily installs here even though no training task imports
    # it. That version is newer than the one apache-airflow-core 3.3.1
    # requires (<0.137.0) and breaks Airflow's own execution API: cadwyn
    # (Airflow's API-versioning layer) fails its route-schema conversion at
    # startup, every task-instance state update returns HTTP 500, and the
    # scheduler crash-loops trying to unpickle the resulting error. Nothing
    # under triagem.training imports fastapi, so pinning it back to
    # Airflow's required range is safe here.
    && pip install --no-cache-dir "fastapi<0.137.0,>=0.129.0"
