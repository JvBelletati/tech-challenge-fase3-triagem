"""Produce the Stage 4 latency comparison report.

Usage:
    python scripts/benchmark.py --rounds 400
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

from triagem.config import CURRENT_MODEL_DIR, MODEL_FILES, PROJECT_ROOT, RANDOM_STATE
from triagem.training.benchmark import benchmark_variants, format_report
from triagem.training.export import count_graph_ops
from triagem.training.ingest import download_dataset, load_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=400)
    parser.add_argument("--samples", type=int, default=400)
    args = parser.parse_args()

    df = load_dataset(download_dataset()).sample(n=args.samples, random_state=RANDOM_STATE)
    texts = df["medical_abstract"].astype(str).tolist()
    labels = df["condition_label"].astype(int).tolist()

    results = benchmark_variants(CURRENT_MODEL_DIR, texts, labels, rounds=args.rounds)
    graph_ops = {
        "onnx-fp32": count_graph_ops(CURRENT_MODEL_DIR / MODEL_FILES["onnx"]),
        "onnx-int8": count_graph_ops(CURRENT_MODEL_DIR / MODEL_FILES["onnx_quantized"]),
    }

    table = format_report(results, graph_ops)
    output = Path(PROJECT_ROOT) / "docs" / "benchmarks" / "comparativo-latencia.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"# Comparativo de latencia — Etapa 4\n\n"
        f"Medido em {date.today().isoformat()} | {args.rounds} inferencias por variante | "
        f"uma requisicao por vez, CPU, `intra_op_num_threads=1`.\n\n{table}\n",
        encoding="utf-8",
    )
    print(table)
    print(f"\nrelatorio salvo em {output}")


if __name__ == "__main__":
    main()
