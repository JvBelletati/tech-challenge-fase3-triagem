"""Measure end-to-end HTTP latency against a running API.

Usage:
    python scripts/medir_latencia.py --requests 500
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request

LAUDOS = [
    "The patient presented with acute chest pain radiating to the left arm, "
    "accompanied by dyspnea and diaphoresis. ECG showed ST elevation.",
    "Routine endoscopy revealed mild gastritis without evidence of ulceration "
    "or malignancy. Biopsy samples were obtained for histological analysis.",
    "MRI of the brain demonstrated multiple periventricular lesions consistent "
    "with demyelinating disease. Clinical correlation is recommended.",
    "Histopathology confirmed a well differentiated adenocarcinoma with clear "
    "surgical margins and no lymphovascular invasion identified.",
    "Laboratory findings were within normal limits. No acute pathological "
    "process was identified on the current examination.",
]


def post(url: str, text: str) -> float:
    payload = json.dumps({"texto": text}).encode()
    request = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=30) as response:
        response.read()
    return (time.perf_counter() - started) * 1000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    args = parser.parse_args()

    for i in range(args.warmup):
        post(args.url, LAUDOS[i % len(LAUDOS)])

    samples = [post(args.url, LAUDOS[i % len(LAUDOS)]) for i in range(args.requests)]
    samples.sort()

    def percentile(p: float) -> float:
        return samples[min(int(len(samples) * p), len(samples) - 1)]

    print(f"requisicoes : {len(samples)}")
    print(f"media       : {statistics.mean(samples):.3f} ms")
    print(f"p50         : {percentile(0.50):.3f} ms")
    print(f"p95         : {percentile(0.95):.3f} ms")
    print(f"p99         : {percentile(0.99):.3f} ms")


if __name__ == "__main__":
    main()
