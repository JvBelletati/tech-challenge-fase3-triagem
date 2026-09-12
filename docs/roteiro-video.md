# Roteiro do vídeo — método STAR (cronometrado, máximo 5:00)

Roteiro de trabalho: cada bloco traz a janela de tempo, os pontos a falar
(não é texto para decorar palavra por palavra) e o que mostrar na tela.
Fale naturalmente a partir dos pontos — o cronômetro é o limite duro, não a
meta.

---

## Checklist de pré-gravação

Fazer **antes** de apertar "gravar", nesta ordem:

1. **Subir a stack completa** (API + Prometheus + Grafana):
   ```bash
   docker compose up -d --build
   ```
   Confirmar saudável: `curl -s http://localhost:8000/health`.

2. **Gerar carga por ~2 minutos antes de gravar**, para o Grafana já ter
   histórico e não abrir com o painel vazio na borda esquerda:
   ```bash
   .venv/Scripts/python.exe scripts/gerar_carga.py --duration 120 --rps 20
   ```
   Deixar terminar antes de começar a gravação de fato (a gravação em si
   não precisa esperar mais carga — o histórico dos últimos 2 minutos já
   estará no Grafana).

3. **Deixar as abas já abertas** antes de gravar, na ordem em que serão
   usadas:
   - `http://localhost:8000/docs` (Swagger UI)
   - Terminal com o comando `curl -X POST /predict` pronto (só falta apertar
     Enter)
   - GitHub → aba Actions do repositório (workflow verde)
   - `http://localhost:8080` (Airflow), já na tela da DAG `retreino_triagem`
   - `http://localhost:3000` (Grafana), já no dashboard "Triagem de Laudos
     Medicos" (UID `triagem-main`), no painel "Latencia do modelo vs.
     overhead HTTP"
   - `docs/benchmarks/comparativo-latencia.md` aberto num editor, para
     mostrar a tabela de latência sem precisar digitar nada

4. **Captura do dashboard**: o enunciado aceita "print OU JSON" como
   evidência do dashboard. O JSON (`monitoring/grafana/dashboards/triagem.json`)
   já está versionado no repositório — não precisa gerar nada para isso. O
   print/captura de tela do dashboard com tráfego é um extra que você tira
   você mesmo durante a gravação (ex.: pausar e printar), não é bloqueante
   para o vídeo.

5. Ter à mão, sem precisar procurar: o texto do laudo cardiovascular de
   exemplo (chest pain + ST elevation, ver bloco Action) e a tabela de
   latência do Result.

---

## Situation — 0:00–0:45

**Falar:**
- Hoje, laudos médicos são processados por ordem de chegada (FIFO): não há
  triagem por gravidade.
- Consequência concreta: um achado cardiovascular crítico pode ficar na fila
  atrás de exames de rotina, só porque chegou depois.
- O custo do erro aqui é **assimétrico**: deixar passar um caso urgente pode
  matar um paciente; escalar um caso à toa custa só o tempo de um clínico
  revisando. Essa assimetria é o fio condutor de todas as decisões técnicas
  do projeto — vai aparecer de novo no Result.

**Mostrar na tela:** um slide único com o problema (fila FIFO → achado
crítico atrás de rotina → custo assimétrico do erro). Sem terminal, sem
código ainda.

---

## Task — 0:45–1:20

**Falar:**
- Os requisitos desta fase do desafio: API em contêiner, CI/CD automatizado,
  retreino orquestrado, observabilidade e latência otimizada.
- Enquadrar rapidamente que cada um desses vira uma seção do vídeo daqui a
  pouco (arquitetura → API/CI/Airflow/Grafana → números de latência).

**Mostrar na tela:** tabela de requisitos da fase (os cinco itens acima),
pode ser o mesmo slide ou uma tabela simples.

---

## Action — 1:20–3:30 (o bloco mais longo — ritmo importa)

**Falar (arquitetura, ~20s):**
- Visão geral rápida: FastAPI serve o modelo em ONNX Runtime; Airflow
  orquestra o retreino semanal com gate de qualidade; Prometheus/Grafana
  observam tudo em produção.

**Mostrar: `POST /predict` ao vivo (~40s)**
- Abrir `http://localhost:8000/docs` por 2 segundos só para situar (Swagger
  existe), depois ir direto ao terminal.
- Rodar ao vivo:
  ```bash
  curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"texto":"The patient presented with acute chest pain radiating to the left arm, accompanied by dyspnea and diaphoresis. Electrocardiogram showed ST segment elevation in the anterior leads consistent with acute myocardial infarction."}'
  ```
- Apontar no JSON de resposta os campos `prioridade` e `revisao_humana` —
  esses dois campos são o produto final do sistema, o resto é meio de
  campo. (Guardar este mesmo exemplo: ele volta com mais peso no Result.)

**Mostrar: GitHub Actions verde (~30s)**
- Aba Actions já aberta. Passar pelos 4 jobs do workflow: `lint`, `test`,
  `build` (build da imagem Docker + smoke test do `/predict`), e
  `validate-dag` (garante que a DAG do Airflow importa sem erro antes mesmo
  de rodar).

**Mostrar: Airflow — DAG e gate de promoção (~40s)**
- `http://localhost:8080`, DAG `retreino_triagem`.
- Falar as 7 tasks em sequência: `ingerir_dados` → `validar_dados` →
  `treinar_modelo` → `avaliar_modelo` → e daí o gate: `exportar_onnx` +
  `promover_modelo` só rodam se o candidato for aprovado; `rejeitar_candidato`
  é o caminho terminal quando o modelo novo é pior que o atual.
- Enfatizar: o gate compara F1 do candidato contra o modelo em produção —
  um retreino que piora o modelo nunca chega a produção sozinho.

**Mostrar: Grafana com carga rodando (~30s)**
- Dashboard "Triagem de Laudos Medicos", painel **"Latencia do modelo vs.
  overhead HTTP"** — apontar que ele separa o tempo de inferência (dentro do
  processo) do tempo de HTTP ponta a ponta (rede + ASGI), porque são coisas
  diferentes e a otimização de latência (Result) atacou a primeira.

---

## Result — 3:30–4:45

**Falar — o número principal (~20s):**
- ONNX Runtime entrega **3,29x** de ganho no p50 (0,570 ms → 0,173 ms) na
  troca de sklearn puro para ONNX, **sem custo de acurácia**: F1-macro
  praticamente igual, 0,5445 → 0,5427.

**Mostrar:** tabela comparativa de `docs/benchmarks/comparativo-latencia.md`
(sklearn / onnx-fp32 / onnx-int8, colunas p50/p95/p99/F1/tamanho).

**Falar — as lições aprendidas (isto é o que o método STAR realmente
cobra, e é o material mais forte do projeto — não apressar):**

1. **A quantização dinâmica int8 não deu ganho nenhum**, e o grafo ONNX
   explica por quê: `quantize_dynamic` reescreve nós `MatMul`/`Gemm`, mas o
   `skl2onnx` exporta o classificador como um `LinearClassifier` do
   `ai.onnx.ml` — um operador que a quantização dinâmica nunca toca.
   Profiling mostrou também que 73% do tempo de inferência é tokenização e
   só 27% é o classificador, então o teto de ganho possível já era baixo
   antes mesmo de tentar.

2. **Unigramas venceram bigramas nos dois eixos ao mesmo tempo** — mais
   rápido (0,786 ms → 0,614 ms) **e** mais preciso (F1 0,5445 → 0,5589).
   Contraintuitivo (mais contexto geralmente ajuda), e foi descoberto
   medindo, não advinhando.

3. **A regra de segurança pegou um erro real do modelo.** Voltar ao exemplo
   do Action: aquele mesmo laudo cardiovascular clássico (dor no peito, ST
   elevation) foi classificado pelo modelo como `general pathological
   conditions` — categoria de baixa prioridade — com confiança de apenas
   0,3993, abaixo do limiar de 0,40. Por isso o sistema não deixou passar
   como NORMAL: escalou para `ATENCAO` e marcou `revisao_humana: true`. É o
   design de custo assimétrico do Situation funcionando na prática, sobre um
   caso real — a coisa mais forte para mostrar no vídeo inteiro.

**Mostrar na tela:** tabela comparativa de latência; se der tempo, voltar
1s ao JSON do `/predict` do Action para reforçar visualmente o ponto 3.

---

## Fecho — 4:45–5:00

**Falar:**
- Próximo passo natural do projeto: monitorar *drift* de dados/modelo ao
  longo do tempo, usando o histograma de confiança das predições que já
  está instrumentado no Grafana — não precisa de nada novo para começar,
  só olhar a métrica que já existe.

**Mostrar na tela:** painel "Confianca das predicoes (p10 / p50) e
requisicoes em voo" do dashboard.

---

## Referência rápida de tempo

| Bloco | Janela | Duração |
|---|---|---|
| Situation | 0:00–0:45 | 45s |
| Task | 0:45–1:20 | 35s |
| Action | 1:20–3:30 | 2:10 |
| Result | 3:30–4:45 | 1:15 |
| Fecho | 4:45–5:00 | 15s |
| **Total** | | **5:00** |
