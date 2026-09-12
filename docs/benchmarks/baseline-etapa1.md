# Baseline de latência — Etapa 1

**Data da medição:** 2026-09-11

## O que foi medido

Latência HTTP ponta a ponta de `POST /predict`, do host (Windows, fora do
container) até a API FastAPI rodando dentro do container Docker, publicada
na porta 8000. Cada amostra é o tempo de uma requisição completa — conexão
TCP, envio do corpo JSON, processamento pelo FastAPI, inferência ONNX e
resposta — medido no cliente com `time.perf_counter()` ao redor de
`urllib.request.urlopen`. **Não** é o tempo de inferência isolado do modelo:
inclui overhead de rede (localhost/Docker Desktop no Windows), do runtime
ASGI (uvicorn/Starlette) e da serialização/desserialização JSON, além do
tempo de inferência em si.

Para referência, uma chamada isolada de `/predict` durante a verificação
manual reportou `latencia_ms: 0.565` no campo interno do `PredictResponse`
(tempo apenas de `predictor.predict()`, medido dentro do processo) — a
diferença enorme para os números abaixo confirma que o que este benchmark
captura é dominado por overhead de rede/HTTP, não pelo modelo.

## Condições da medição

- **Requisições medidas:** 500 (`--requests 500`, valor padrão do script)
- **Aquecimento (warm-up):** 50 requisições descartadas antes da medição
  (`--warmup 50`, padrão do script), além do warm-up interno de 3 rounds que
  o próprio `Predictor` já executa no startup do processo
- **Hardware:** CPU (`CPUExecutionProvider` do ONNX Runtime); nenhuma GPU
  envolvida
- **Modelo servido:** versão `20260911-114522` (variante `onnx`, arquivo
  `model.onnx`), lida de `models/current/metadata.json` e confirmada no
  próprio log de startup do container (`loaded model 20260911-114522 (onnx)
  from /app/models/current/model.onnx`) e na resposta de `/health`
- **Imagem:** `triagem-api:latest`, tamanho total ~486 MB (`docker images`;
  `docker image inspect` reporta 118.540.697 bytes só na camada final da
  imagem) — bem abaixo de 1 GB, já que scikit-learn, pandas e skl2onnx (o
  extra `[training]`) não entram na imagem de inferência
- **Cliente de medição:** `.venv/Scripts/python.exe scripts/medir_latencia.py
  --requests 500`, rodando no host, usando só a biblioteca padrão
  (`urllib.request`), sem reuso de conexão HTTP entre requisições

## Verificação manual do container

```
$ curl -s http://localhost:8000/health
{"status":"ok","modelo":{"versao":"20260911-114522","runtime":"onnx"}}

$ curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
    -d '{"texto":"The patient presented with acute chest pain radiating to the left arm, accompanied by dyspnea and diaphoresis. ECG showed ST elevation."}'
{"categoria":"general pathological conditions","categoria_id":5,"prioridade":"ATENCAO",
 "confianca":0.3993,"revisao_humana":true,"latencia_ms":0.565,
 "modelo":{"versao":"20260911-114522","runtime":"onnx"}}
```

Reverificado nesta tarefa (com o mesmo `models/current/`, versão
`20260911-114522`, chamando `Predictor.predict()` diretamente em vez do
container, já que o Docker estava indisponível na máquina): o texto acima
mede `confianca=0.3993`, `categoria="general pathological conditions"`,
`prioridade="ATENCAO"`, `revisao_humana=True` — condiz exatamente com a
resposta documentada. Esta é a **mesma variante curta** do laudo usada em
`tests/test_api.py`; a variante mais longa de `scripts/gerar_carga.py`
(com a frase completa sobre o eletrocardiograma) mede `confianca=0.4548`,
`categoria="cardiovascular diseases"`, `prioridade="URGENTE"`,
`revisao_humana=False` — **não** aciona a regra de segurança. As duas não
são intercambiáveis nos exemplos deste projeto.

## Resultado da medição

Saída real de `scripts/medir_latencia.py --requests 500`:

```
requisicoes : 500
media       : 7.083 ms
p50         : 3.459 ms
p95         : 24.638 ms
p99         : 27.967 ms
```

## Observação sobre os números

O p50 (3.46 ms) é coerente com o custo de inferência quase nulo (0.565 ms)
somado ao overhead de uma requisição HTTP local. Mas o salto entre p50 e
p95/p99 (de ~3.5 ms para ~25-28 ms, quase 7-8x) chama atenção e não deveria
ser suavizado: o script não reutiliza conexão HTTP entre requisições
(`urllib.request.urlopen` abre uma conexão TCP nova a cada chamada), e o
tráfego passa pelo NAT/proxy de rede que o Docker Desktop no Windows usa
para publicar a porta do container no host. É plausível que uma fração das
conexões (por volta de 5%) pague um custo extra de handshake/roteamento
nessa camada de virtualização de rede do Windows, e não algo relacionado ao
modelo ou ao FastAPI. Isso não foi investigado a fundo nesta tarefa — fica
registrado como característica observada do ambiente de medição (host
Windows + Docker Desktop), não como comportamento do modelo em si.

## Contexto

Este é o baseline de latência da **Etapa 1**: a API contêinerizada, servindo
o modelo ONNX real committado em `models/current/`, medida ponta a ponta do
host para o container, sem nenhuma otimização de performance aplicada além
do que já existia (sessão ONNX reutilizada, warm-up no startup). Os
resultados da Etapa 4 (otimização) deverão ser comparados contra estes
números — em especial a média e o p50, que refletem o caminho comum, e o
p95/p99, que capturam a causa de variância descrita acima.
