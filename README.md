# Triagem de Laudos Médicos

Serviço de Machine Learning que classifica laudos médicos em texto livre e devolve uma prioridade clínica de triagem (URGENTE / ATENCAO / NORMAL), servido por uma API ONNX com observabilidade Prometheus/Grafana e retreino automatizado via Airflow.

## O problema

Um hospital de referência recebe um volume alto de laudos em texto livre e processa a fila por ordem de chegada. Isso significa que um laudo com achado cardiovascular crítico pode esperar atrás de dezenas de laudos de rotina, simplesmente porque chegou depois. O objetivo deste sistema é classificar automaticamente cada laudo e atribuir uma prioridade de triagem, permitindo reordenar a fila por urgência clínica em vez de por ordem de chegada. O foco avaliativo do projeto não é a acurácia do modelo — é o ciclo de vida dele em produção: CI/CD, orquestração de retreino, observabilidade e latência.

## Arquitetura da solução

```
                              FLUXO EM TEMPO REAL (a cada requisicao)

  cliente HTTP         +------------------+      +-----------+      +------------+      +-----------+
  (hospital,     ----> |  API (FastAPI +  | ---> | /metrics  | ---> | Prometheus | ---> |  Grafana  |
   POST /predict)      |  ONNX Runtime)   |      | (client)  |      | (scrape 5s)|      | 7 paineis |
                        +------------------+      +-----------+      +------------+      +-----------+
                                 ^
                                 | le o modelo servido (variante onnx)
                                 |
                        +------------------+
                        | models/current/  |  <- volume compartilhado, sem rebuild de imagem
                        | model.onnx +     |
                        | metadata.json    |
                        +------------------+
                                 ^
                                 | copia ao promover
                                 |
                              FLUXO ASSINCRONO (Airflow, semanal, catchup=False)

  +----------------------------------------------------------------------------------+
  |  DAG retreino_triagem (7 tasks)                                                   |
  |                                                                                    |
  |  ingerir_dados -> validar_dados -> treinar_modelo -> avaliar_modelo                |
  |                                                            |                       |
  |                                     F1 >= producao ------- +------- F1 < producao  |
  |                                          |                                |        |
  |                                          v                                v        |
  |                                   exportar_onnx                  rejeitar_candidato|
  |                                          |                                         |
  |                                          v                                         |
  |                                   promover_modelo                                  |
  +----------------------------------------------------------------------------------+
```

## Decisão de arquitetura de deploy em nuvem (Azure)

Esta seção documenta a análise textual de deploy exigida pela Etapa 1. Nenhum deploy real foi feito — a análise é derivada da evidência medida neste repositório (`docs/benchmarks/`), não de material de marketing dos serviços.

### Batch versus real-time

A triagem é, por natureza, **real-time**. O valor da informação decai em minutos: um laudo triado em um lote noturno chega depois da decisão clínica que deveria orientar. Processamento em lote só se justificaria para reprocessar histórico (por exemplo, reclassificar laudos antigos após uma mudança de modelo), nunca para o caminho principal de triagem.

### Serviço de serving: Azure Container Apps

A escolha é sustentada pelos números medidos neste projeto, não por preferência genérica:

- O modelo responde em **~0,17 ms** (`docs/benchmarks/comparativo-latencia.md`, variante `onnx-fp32`), rodando em CPU, sem GPU e sem dependência pesada. Isso significa que o gargalo de um serviço real não é computação — é **concorrência de requisições** durante os picos de plantão de um hospital, não uma carga constante ao longo do dia.
- **Azure Container Apps** escala horizontalmente por número de requisições e **escala a zero** fora de horário de pico, o que casa exatamente com esse padrão de carga: pagar por capacidade ociosa às 3h da manhã não faz sentido quando o modelo tem meio milissegundo de custo por chamada.
- **AKS foi descartado**: traria um plano de controle Kubernetes inteiro (nós, etcd, upgrades, RBAC de cluster) para servir um único contêiner de ~486 MB sem estado. É complexidade operacional sem contrapartida — nenhum dos requisitos deste serviço (multi-tenant, orquestração de múltiplos serviços internos, autoscaling por métricas customizadas complexas) justifica um cluster dedicado.
- **Azure ML Endpoints foi descartado**: é otimizado para inferência que se beneficia de GPU e de features de MLOps (data drift automático, A/B testing de modelo) que este pipeline não usa — o modelo é TF-IDF + Regressão Logística em CPU, e o ganho de latência já foi obtido via ONNX, não via hardware especializado.

### Registro de imagens: Azure Container Registry

A imagem `triagem-api` (486 MB, ver `docs/benchmarks/baseline-etapa1.md`) seria publicada no **Azure Container Registry**, alimentado pelo mesmo workflow do GitHub Actions (`.github/workflows/ci.yml`, job `build`) que já constrói e testa a imagem a cada push — bastaria adicionar um passo de `docker push` autenticado no ACR ao job existente, sem reescrever o pipeline de CI.

### Artefatos de modelo: Azure Blob Storage

O volume `models/` (montado hoje via bind mount local em `docker-compose.yml`) seria substituído por **Azure Blob Storage**, com versionamento por container/blob e `metadata.json` (o mesmo arquivo já gerado pelo pipeline de treino) como manifesto de promoção — a API leria a versão ativa de lá em vez de um diretório local, preservando exatamente a mesma lógica que hoje lê `models/current/metadata.json`.

### Orquestração: Azure Data Factory ou Airflow gerenciado em AKS

Duas opções, com um trade-off que vale registrar sem meio-termo:

- **Azure Data Factory** é mais barato para operar (sem cluster dedicado, cobrança por execução de pipeline), mas exigiria **reescrever a DAG** `retreino_triagem` em atividades nativas do Data Factory — o código de `airflow/dags/retreino_triagem_dag.py` e os testes em `tests/test_dag.py` seriam descartados.
- **Airflow gerenciado em AKS** custa mais (cluster sempre ligado), mas **preserva o código como está**: a DAG continua orquestrando as mesmas funções de `src/triagem/training/*` sem nenhuma reescrita.

Dado que o pipeline de treino já está implementado e testado, a opção que preserva esse investimento (Airflow em AKS) é defensável mesmo custando mais — mas o Data Factory é a escolha correta se o objetivo for reduzir custo operacional e a equipe aceitar reimplementar a orquestração.

### Observabilidade: Azure Monitor (managed Prometheus) + Azure Managed Grafana

O ponto central aqui é a portabilidade: `monitoring/prometheus/prometheus.yml` e o JSON do dashboard (`monitoring/grafana/dashboards/triagem.json`) migram **praticamente sem alteração** para o Azure Monitor com *managed Prometheus* e para o Azure Managed Grafana — só o alvo de scrape muda (de `api:8000` para o endpoint do Container App). Essa é a vantagem concreta de ter instrumentado o serviço com padrões abertos (`prometheus-client`, formato de exposição Prometheus) em vez de uma solução proprietária: a análise que já existe (as 7 métricas, os 7 painéis) não precisa ser refeita, só reapontada.

### Rede e conformidade

Laudo médico é dado de saúde. A topologia recomendada:

- Azure Container Apps em um ambiente com **VNet interna**, sem IP público exposto.
- Acesso de fora da VNet apenas via **Private Endpoint**.
- **Logs sem conteúdo de laudo**: a API, como implementada hoje, nunca registra o texto do laudo em log nem em métrica. `src/triagem/api/main.py` só loga exceções (`logger.exception("inference failed")`, sem o corpo da requisição) e o módulo de métricas (`src/triagem/api/metrics.py`) só observa `categoria`, `confianca` e latência — nunca `request.texto`. Essa propriedade já existe no código atual e se mantém válida em qualquer ambiente de deploy.

## Stack tecnológica

| Tecnologia | Papel |
|---|---|
| Python 3.12 | linguagem do projeto (local e nas imagens Docker) |
| FastAPI + Pydantic | API HTTP, validação de contrato e documentação automática (`/docs`) |
| Uvicorn | servidor ASGI |
| scikit-learn (TF-IDF + Regressão Logística) | treino do modelo |
| skl2onnx / ONNX Runtime | exportação e inferência do modelo em produção |
| prometheus-client | instrumentação de métricas na API |
| Prometheus | coleta e armazenamento de séries temporais |
| Grafana | dashboard de observabilidade (7 painéis, provisionado por arquivo) |
| Apache Airflow 3.3.1 | orquestração do pipeline de retreino (`retreino_triagem`) |
| PostgreSQL 16 | metadata store do Airflow |
| Docker / Docker Compose | containerização e orquestração local (com `profiles`) |
| GitHub Actions | CI/CD (lint, testes, build de imagem, validação da DAG) |
| Ruff | lint e formatação |
| pytest | testes automatizados |

## Como executar

Pré-requisitos: Docker e Docker Compose. O modelo servido já vem versionado no repositório (`models/current/`), então subir a stack não exige treinar nada nem baixar o dataset; o download só acontece se você optar por retreinar (via DAG ou `scripts/treinar.py`), e mesmo nesse caso não exige nenhuma credencial externa — o dataset é baixado por HTTPS público.

```bash
git clone <url-do-repositorio> triagem-laudos
cd triagem-laudos
docker compose up -d --build
```

Isso sobe **API + Prometheus + Grafana** (o profile padrão). URLs disponíveis:

| Serviço | URL |
|---|---|
| API (Swagger) | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana (sem login, acesso anônimo) | http://localhost:3000 |

Para popular o dashboard com tráfego realista:

```bash
python scripts/gerar_carga.py
```

Esse script usa só a biblioteca padrão do Python — não exige instalar as dependências do projeto.

Para subir também o Airflow e ver a DAG de retreino, primeiro crie o `.env`
e os diretórios que o Compose monta (`data/` e `airflow/logs/` são
ignorados pelo git, e se o Docker os criar sozinho eles ficam com dono
`root`, enquanto os serviços do Airflow rodam como UID 50000 — a promoção e
a ingestão falham com `PermissionError` num clone limpo no Linux):

```bash
echo "AIRFLOW_UID=$(id -u)" > .env   # Linux/macOS; no Windows Docker Desktop
mkdir -p data airflow/logs
docker compose --profile airflow up -d
```

Airflow fica disponível em http://localhost:8080, sem tela de login: `AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_ALL_ADMINS=true` (já configurado no `docker-compose.yml`) dá acesso direto de administrador, no mesmo espírito do acesso anônimo do Grafana.

**Nota:** `docker compose down` (sem `--profile airflow`) não para os serviços do profile `airflow` caso estejam no ar — se for necessário liberar a porta 8080, use `docker compose --profile airflow down`. (O Postgres do Airflow não expõe porta nenhuma ao host — roda só na rede interna do Compose, então não há conflito de porta 5432 a liberar.)

Exemplo real de chamada ao endpoint principal (resposta colada de uma execução real desta stack):

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texto":"The patient presented with acute chest pain radiating to the left arm, ..."}'
```

```json
{"categoria":"nervous system diseases","categoria_id":3,"prioridade":"ATENCAO","confianca":0.5082,"revisao_humana":false,"latencia_ms":0.326,"modelo":{"versao":"20260911-114522","runtime":"onnx"}}
```

## O modelo

**Dataset:** [Medical Abstracts TC Corpus](https://github.com/sebischair/Medical-Abstracts-TC-Corpus), 11.550 linhas de treino, 5 classes, baixado por HTTPS direto — sem credencial do Kaggle, o que mantém a ingestão da DAG reprodutível por quem for avaliar.

**Pipeline:** `TfidfVectorizer` (unigramas, `min_df=3`, `max_features=20_000`, `sublinear_tf=True`) → `LogisticRegression` (`C=4.0`, `max_iter=1000`, `class_weight="balanced"`).

**Resultado (conjunto de teste retido, split estratificado de 2.310 linhas):** F1-macro **0,5589**, acurácia 0,5593. Concordância entre predição ONNX e sklearn: **98,83%**.

**Honestidade sobre o número:** um F1-macro de ~0,56 em 5 classes de abstracts médicos não é um resultado forte em termos absolutos (o acaso ficaria perto de 0,20). É adequado ao propósito desta fase: a rubrica avalia o ciclo de vida do modelo em produção — CI/CD, orquestração, observabilidade, latência — não desempenho de ponta em classificação. O número é reportado como é, sem maquiagem.

**Regressão Logística em vez de Random Forest (desvio documentado):** o enunciado do PDF sugere "TF-IDF + Random Forest", mas autoriza explicitamente "ou modelo leve similar". A substituição foi deliberada, por quatro razões:

1. Random Forest sobre uma matriz esparsa de dezenas de milhares de features tem inferência lenta — exatamente a métrica que a Etapa 4 pede para otimizar.
2. O artefato serializado de um Random Forest fica na casa de centenas de MB, inviabilizando uma imagem Docker enxuta.
3. Probabilidades de Random Forest não são bem calibradas; a regra de segurança clínica (seção seguinte) depende de uma confiança utilizável.
4. Regressão Logística exporta de forma limpa e estável para ONNX.

## Regra de triagem

| Categoria prevista | Prioridade |
|---|---|
| cardiovascular diseases | URGENTE |
| neoplasms | URGENTE |
| nervous system diseases | ATENCAO |
| digestive system diseases | ATENCAO |
| general pathological conditions | NORMAL |

**Regra de segurança clínica:** se a confiança da predição for menor que **0,40**, o caso nunca é rebaixado para NORMAL — é elevado a ATENCAO e marcado com `revisao_humana: true`. O custo do erro é assimétrico: classificar um caso urgente como normal tem consequência clínica grave, enquanto o inverso custa apenas tempo de revisão.

**O limiar foi escolhido por medição, não por palpite.** Na distribuição real de confiança do modelo treinado, 0,40 encaminha **8,1%** das predições para revisão humana. As alternativas foram descartadas por inviabilidade operacional: 0,50 encaminharia 24,7% e 0,60 encaminharia 42,9% — uma fila de revisão maior do que o problema que o sistema veio resolver.

## Otimização de latência

Medido com 400 rodadas de inferência por variante, uma requisição por vez, CPU, `intra_op_num_threads=1`, F1-macro nas 2.310 linhas completas do conjunto de testes retido (`docs/benchmarks/comparativo-latencia.md`):

| Variante | p50 (ms) | p95 (ms) | p99 (ms) | F1-macro | Tamanho (MB) | Ganho vs. baseline |
|---|---|---|---|---|---|---|
| sklearn (joblib) | 0,580 | 0,826 | 0,933 | 0,5589 | 1,22 | 1,00x |
| ONNX Runtime fp32 | 0,174 | 0,225 | 0,243 | 0,5591 | 0,80 | 3,34x |
| ONNX Runtime int8 | 0,177 | 0,223 | 0,251 | 0,5591 | 0,80 | 3,28x |

Por medir F1 no held-out completo (não numa amostra parcial), os valores de sklearn desta tabela reconciliam exatamente com `models/current/metadata.json` (0,5589).

Três descobertas, nessa ordem de importância:

1. **Unigramas venceram bigramas** — a única das três encontrada por *profiling* em vez de por receita. Medição mostrou que 73% do tempo de inferência é tokenização/TF-IDF e só 27% é o classificador, o que redirecionou a otimização para o espaço de features: trocar `ngram=(1,2)` + 50k features por `ngram=(1,1)` + 20k features deixou o modelo mais rápido **e** mais preciso ao mesmo tempo (0,786 ms → 0,614 ms de p50 e F1 0,5445 → 0,5589, ambos medidos no held-out completo de 2.310 linhas, no spike de validação que comparou as duas configurações de features — uma comparação distinta da tabela de latência acima, que mede apenas a configuração atual de unigramas contra os três runtimes).
2. **ONNX Runtime entrega ~3,3x de ganho** no p50 (0,580 ms → 0,174 ms na variante fp32) sem custo de acurácia (F1 praticamente idêntico, 0,5589 → 0,5591).
3. **A quantização dinâmica int8 não teve efeito neste pipeline** — resultado negativo, reportado com a causa raiz: `quantize_dynamic` reescreve operadores `MatMul`/`Gemm`, mas o skl2onnx exporta o classificador como um `LinearClassifier`, operador do domínio `ai.onnx.ml` que a quantização dinâmica nunca toca. O grafo quantizado tem exatamente a mesma contagem de operadores do original (ver tabela em `docs/benchmarks/comparativo-latencia.md`) e nenhuma diferença de latência mensurável — mas não é byte a byte idêntico: `model.onnx` tem 799.552 bytes e `model_int8.onnx` tem 799.780 bytes, porque `quantize_dynamic` ainda reescreve algum bookkeeping do grafo (ver `tests/test_export.py::test_quantization_leaves_the_classifier_untouched`), só nunca o classificador. Por isso a API serve, por padrão, a variante `onnx` (fp32), não a `onnx_quantized`.

**Contexto adicional — por que a API mede inferência separada de HTTP:** o baseline da Etapa 1 (`docs/benchmarks/baseline-etapa1.md`), medido do host Windows para o container via HTTP, mostrou média de 7,083 ms, p50 de 3,459 ms, p95 de 24,638 ms e p99 de 27,967 ms em 500 requisições. Só olhando esse número, o ganho do ONNX pareceria irrelevante. Mas a métrica `triagem_inference_duration_seconds`, medida **dentro** do container, isolando só o modelo, mostrou p95 de 0,478 ms contra p95 de 0,491 ms da métrica HTTP total — uma diferença de ~13 microssegundos, que é o overhead real do framework (FastAPI/Starlette/Uvicorn). A distância entre os 25 ms observados do host e os 0,5 ms medidos dentro do container é causada pela camada de rede do Docker Desktop no Windows (NAT/proxy de porta), não pela aplicação. Essa é a justificativa concreta para instrumentar latência de modelo separada de latência HTTP: sem a separação, o ganho do ONNX ficaria mascarado por overhead de rede que não tem relação com o modelo.

## CI/CD

`.github/workflows/ci.yml`, disparado em push e pull request para `main`. Quatro jobs:

| Job | O que protege |
|---|---|
| `lint` | Ruff check + format check — impede código fora do padrão de entrar em `main` |
| `test` | pytest com cobertura — impede regressão na lógica de domínio, contratos da API e pipeline de treino |
| `build` | constrói a imagem Docker e faz smoke test em `/health` e `/predict` — impede que uma imagem que não sobe ou não responde chegue a ser publicada |
| `validate-dag` | importa a DAG do Airflow (falha em erro de sintaxe/import) e roda `pytest tests/test_dag.py` — impede que um erro na definição da DAG, ou uma regressão na sua estrutura (tasks, dependências, `catchup`), só seja descoberto quando o Airflow tentar carregá-la em produção |

## Pipeline de retreino

A DAG `retreino_triagem` (`airflow/dags/retreino_triagem_dag.py`, agendamento semanal, `catchup=False`) tem 7 tasks: `ingerir_dados → validar_dados → treinar_modelo → avaliar_modelo → exportar_onnx → promover_modelo`, com `rejeitar_candidato` como ramo terminal quando o gate de qualidade reprova o candidato.

**Decisão estrutural:** a maior parte da lógica de negócio vive fora da DAG — `ingerir_dados`, `validar_dados`, `treinar_modelo`, `avaliar_modelo` e `exportar_onnx` são chamadas finas para `src/triagem/training/*`, o que torna **o pipeline de treino** (`ingest.py`, `train.py`, `evaluate.py`, `export.py`) testável com pytest sem precisar subir o Airflow. `promover_modelo` e `rejeitar_candidato` são a exceção deliberada: montar `metadata.json` e copiar artefatos para `models/current/` é lógica específica de orquestração, então fica inline nessas duas tasks. A fiação entre tasks, o comportamento de `AirflowSkipException` e da `trigger_rule="all_done"` de `rejeitar_candidato` não são totalmente testáveis fora do Airflow: `tests/test_dag.py` verifica import, presença das 7 tasks e ordem de dependências usando `DagBag`, mas isso exige o pacote `apache-airflow` instalado (é pulado no job `test` do CI, que não o instala, e roda no job `validate-dag`, que instala) e não substitui uma execução real. A DAG foi verificada rodando ponta a ponta no Airflow local (`docker compose --profile airflow up -d`), promovendo um modelo com sucesso.

**Gate de qualidade:** um retreino só é promovido se o F1-macro do candidato for maior ou igual ao do modelo em produção (`should_promote`, em `src/triagem/training/evaluate.py`). Um candidato pior é descartado (`rejeitar_candidato`) e os artefatos ficam em `models/candidates/` para inspeção, sem substituir o modelo servido. Sem esse gate, a DAG seria só três tasks em fila; com ele, ela defende a produção.

**O ciclo de retreino não é fechado automaticamente:** `promover_modelo` copia os novos artefatos para `models/current/`, mas a API carrega o modelo em memória uma única vez, no startup (`lifespan`, em `src/triagem/api/main.py`), e continua servindo a versão antiga até o processo ser reiniciado — depois de uma promoção, rode `docker compose restart api` para a API passar a servir o modelo recém-promovido.

## Estrutura de pastas

```
.github/workflows/ci.yml          # lint, testes, build da imagem, validacao da DAG
airflow/
  Dockerfile                      # imagem do Airflow com o pacote triagem instalado
  dags/retreino_triagem_dag.py    # DAG de retreino (7 tasks, sem logica propria)
src/triagem/
  config.py                       # paths, hiperparametros e constantes - fonte unica da verdade
  inference/
    predictor.py                  # carrega a sessao ONNX, roda inferencia, mede latencia
    priority.py                   # regra de prioridade clinica, pura, sem I/O
  api/
    main.py                       # rotas FastAPI (/health, /predict, /metrics)
    schemas.py                    # contratos Pydantic (campos em portugues)
    metrics.py                    # instrumentacao Prometheus
  training/
    ingest.py                     # download e validacao do dataset
    train.py                      # pipeline TF-IDF + LogisticRegression
    evaluate.py                   # metricas e gate de promocao (should_promote)
    export.py                     # exportacao ONNX, quantizacao, verificacao de paridade
    benchmark.py                  # comparativo de latencia entre variantes
monitoring/
  prometheus/prometheus.yml       # config de scrape (5s)
  grafana/
    dashboards/triagem.json       # 7 paineis
    provisioning/                 # datasource + dashboard, sem clique manual
models/
  current/                        # modelo servido: .onnx, .joblib, metadata.json - versionado em git
  candidates/                     # saida de retreinos ainda nao promovidos - fora do git
scripts/
  gerar_carga.py                  # gerador de trafego para o dashboard nao ficar vazio
  medir_latencia.py               # medicao de latencia HTTP ponta a ponta (baseline Etapa 1)
  benchmark.py                    # CLI do comparativo sklearn vs ONNX vs ONNX int8
  treinar.py                      # CLI de treino ponta a ponta (usada fora do Airflow tambem)
tests/
  fixtures/                       # recorte pequeno do dataset, sem dependencia de rede
docs/
  benchmarks/                     # relatorios medidos (baseline Etapa 1, comparativo Etapa 4)
docker-compose.yml                # api + prometheus + grafana (default) e airflow via profile
Dockerfile                        # imagem de inferencia, sem dependencias de treino
pyproject.toml
```

## Decisões de projeto e trade-offs

- **Regressão Logística no lugar de Random Forest.** Desvio consciente do enunciado (que autoriza "ou modelo leve similar"): inferência mais rápida, artefato menor, probabilidades calibradas para a regra de triagem e exportação ONNX limpa — ver seção "O modelo".
- **Modelo versionado em git (`models/current/`).** Sem isso, quem avalia precisaria treinar o modelo antes de conseguir subir a stack a partir de um clone limpo. O modelo é pequeno o bastante (ONNX de 0,80 MB) para isso ser razoável; `models/candidates/` fica fora do controle de versão.
- **Quantização mantida no pipeline apesar de não render ganho de latência.** O enunciado pede uma técnica de otimização aplicada e comparada — o ganho já veio do ONNX Runtime. Em vez de esconder o resultado negativo da quantização, ele é medido, reportado e explicado pela causa raiz (o classificador vira um operador `ai.onnx.ml` que a quantização dinâmica não alcança). Um resultado negativo bem diagnosticado vale mais do que um número maquiado.
