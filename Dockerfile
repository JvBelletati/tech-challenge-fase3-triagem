FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# onnxruntime's StringNormalizer op (used by the TF-IDF vectorizer exported
# from skl2onnx) fails to initialize the inference session without a real
# en_US.UTF-8 locale - python:3.12-slim ships with none by default.
RUN apt-get update \
    && apt-get install -y --no-install-recommends locales \
    && sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen \
    && locale-gen \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

COPY pyproject.toml README.md ./
COPY src/ ./src/

# Base dependencies only - no [training] extra. scikit-learn, pandas and
# skl2onnx are training-time dependencies and have no business inside an
# inference image; leaving them out is most of the image size saving.
RUN pip install --no-cache-dir .

COPY models/current/ /app/models/current/

ENV TRIAGEM_MODELS_DIR=/app/models

RUN useradd --create-home --uid 1000 triagem && chown -R triagem:triagem /app
USER triagem

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "triagem.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
