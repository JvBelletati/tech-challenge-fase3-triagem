"""Produce the Stage 4 latency comparison report.

Usage:
    python scripts/benchmark.py --rounds 400
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

from sklearn.model_selection import train_test_split

from triagem.config import (
    CURRENT_MODEL_DIR,
    LABEL_COLUMN,
    MODEL_FILES,
    PROJECT_ROOT,
    RANDOM_STATE,
    TEST_SIZE,
    TEXT_COLUMN,
)
from triagem.training.benchmark import benchmark_variants, format_report
from triagem.training.export import count_graph_ops
from triagem.training.ingest import download_dataset, load_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=400)
    parser.add_argument("--samples", type=int, default=400)
    args = parser.parse_args()

    df = load_dataset(download_dataset())
    # Reproduce the exact train/test split from training to ensure F1 is measured on
    # the held-out set, not on training rows. This prevents data leakage.
    X = df[TEXT_COLUMN].to_numpy(dtype=object)
    y = df[LABEL_COLUMN].to_numpy(dtype=int)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    # Take the first args.samples rows from the held-out test set
    n_holdout = len(X_test)
    n_used = min(args.samples, n_holdout)
    texts = X_test[:n_used].tolist()
    labels = y_test[:n_used].tolist()

    results = benchmark_variants(CURRENT_MODEL_DIR, texts, labels, rounds=args.rounds)
    graph_ops = {
        "onnx-fp32": count_graph_ops(CURRENT_MODEL_DIR / MODEL_FILES["onnx"]),
        "onnx-int8": count_graph_ops(CURRENT_MODEL_DIR / MODEL_FILES["onnx_quantized"]),
    }

    table = format_report(results, graph_ops)
    if n_used >= n_holdout:
        # Covers every row of the held-out split, exactly what training/evaluation
        # scored to produce metadata.json - so the two numbers must reconcile.
        f1_note = (
            f"**F1-macro medido nas {n_used} linhas completas do conjunto de testes "
            f"retido (held-out test set)** para evitar vazamento de dados. Por cobrir "
            f"o held-out inteiro, os valores de sklearn desta tabela reconciliam "
            f"exatamente com `metadata.json` (mesmo split, mesmas linhas)."
        )
    else:
        # A partial slice of the held-out set: still leakage-free, but not the same
        # population metadata.json was scored on, so the numbers will not match it.
        f1_note = (
            f"**F1-macro medido num recorte parcial de {n_used} das {n_holdout} linhas "
            f"do conjunto de testes retido (held-out test set)** — evita vazamento de "
            f"dados, mas por não cobrir o held-out inteiro (rode com `--samples "
            f"{n_holdout}` para isso) este valor **não** deve ser comparado diretamente "
            f"com `metadata.json`."
        )
    output = Path(PROJECT_ROOT) / "docs" / "benchmarks" / "comparativo-latencia.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"# Comparativo de latencia — Etapa 4\n\n"
        f"Medido em {date.today().isoformat()} | {args.rounds} inferencias por variante | "
        f"uma requisicao por vez, CPU, `intra_op_num_threads=1`.\n\n"
        f"{f1_note}\n\n{table}\n",
        encoding="utf-8",
    )
    print(table)
    print(f"\nrelatorio salvo em {output}")


if __name__ == "__main__":
    main()
