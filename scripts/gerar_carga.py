"""Generate realistic traffic so the Grafana dashboard has something to show.

Without this the dashboard is empty during the demo recording.

Usage:
    python scripts/gerar_carga.py --duration 120 --rps 20
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.request

LAUDOS = [
    "The patient presented with acute chest pain radiating to the left arm, "
    "accompanied by dyspnea and diaphoresis. Electrocardiogram showed ST segment elevation "
    "in the anterior leads consistent with acute myocardial infarction.",
    "Upper endoscopy revealed mild antral gastritis without evidence of ulceration or "
    "malignancy. Multiple biopsy samples were obtained for histological analysis and "
    "Helicobacter pylori testing.",
    "Magnetic resonance imaging of the brain demonstrated multiple periventricular white "
    "matter lesions consistent with a demyelinating process. Clinical correlation with "
    "cerebrospinal fluid analysis is recommended.",
    "Histopathological examination confirmed a well differentiated adenocarcinoma with "
    "clear surgical margins. No lymphovascular invasion was identified in the specimen "
    "submitted for analysis.",
    "Laboratory findings were within normal reference limits. No acute pathological process "
    "was identified on the current examination. Routine follow up in twelve months.",
    "Abdominal ultrasound showed hepatic steatosis without focal lesions. The gallbladder "
    "was unremarkable and no biliary dilatation was observed during the study.",
]

MALFORMED = "dor"  # deliberately too short: produces the 422s that give the error panel a signal


def send(url: str, text: str) -> int:
    payload = json.dumps({"texto": text}).encode()
    request = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("--duration", type=int, default=120, help="segundos")
    parser.add_argument("--rps", type=float, default=20.0)
    parser.add_argument(
        "--error-rate",
        type=float,
        default=0.05,
        help="fracao de requisicoes invalidas, para o painel de erro ter sinal",
    )
    args = parser.parse_args()

    interval = 1.0 / args.rps
    deadline = time.time() + args.duration
    sent = 0
    statuses: dict[int, int] = {}

    print(f"gerando ~{args.rps} req/s por {args.duration}s contra {args.url}")
    while time.time() < deadline:
        text = MALFORMED if random.random() < args.error_rate else random.choice(LAUDOS)
        status = send(args.url, text)
        statuses[status] = statuses.get(status, 0) + 1
        sent += 1
        if sent % 100 == 0:
            print(f"  {sent} requisicoes | {statuses}")
        time.sleep(interval)

    print(f"total: {sent} requisicoes | distribuicao de status: {statuses}")


if __name__ == "__main__":
    main()
