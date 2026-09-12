# Roteiro do vídeo — método STAR (cronometrado, máximo 5:00)

Roteiro de trabalho: cada bloco traz a janela de tempo, os pontos a falar
(não é texto para decorar palavra por palavra) e o que mostrar na tela. Os
sub-tempos de cada bloco **somam exatamente** a duração do bloco — isso é
verificado abaixo, bloco a bloco, para não estourar em silêncio. A
contagem de palavras usa 140 palavras/minuto como referência de fala
natural (é uma referência, não uma meta a maximizar — ficar abaixo do
número de palavras é seguro; ultrapassá-lo é o risco).

---

## Checklist de pré-gravação

**Pré-requisito para o sub-segmento do GitHub Actions:** o repositório
precisa estar **publicado no GitHub** e o CI precisa ter terminado **verde**
antes desta checklist se aplicar a esse item. Publicar é decisão do
usuário, feita fora desta tarefa. **Se o repositório ainda não estiver
publicado**, pule o sub-segmento "GitHub Actions verde" do bloco Action
inteiro (não tente simular) — o bloco Action fica com 25s a menos (105s em
vez de 130s), o que é seguro (terminar um bloco mais cedo não é problema;
só estourar é). O restante da checklist não depende de publicação.

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
     Enter) — **ver aviso sobre o texto exato abaixo**
   - GitHub → aba Actions do repositório (workflow verde) — **só abrir esta
     aba se o pré-requisito acima estiver satisfeito**
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
   exemplo (ver aviso abaixo) e a tabela de latência do Result.

> **⚠️ AVISO — use o texto exato, não parafraseie:** o exemplo do laudo
> cardiovascular só funciona se for **colado literalmente**, não digitado de
> memória nem resumido, nem trocado pela variante mais longa que aparece em
> `scripts/gerar_carga.py`. O texto abaixo foi medido diretamente contra o
> `Predictor` nesta verificação e mede confiança **0,3993** (abaixo do
> limiar de 0,40), classificado como `general pathological conditions`, e
> por isso escala para `ATENCAO` com `revisao_humana: true` — é o momento
> mais importante do vídeo. Como contraste medido (não suposto): a variante
> longa de `scripts/gerar_carga.py`, que descreve o mesmo quadro clínico com
> mais detalhe ("...Electrocardiogram showed ST segment elevation in the
> anterior leads consistent with acute myocardial infarction"), mede
> confiança **0,4548**, é classificada como `cardiovascular diseases` e sai
> como `URGENTE` com `revisao_humana: false` — **não** escala. As duas
> variantes descrevem o mesmo caso, mas só a de baixo produz o efeito de
> segurança que este bloco demonstra. Copie e cole exatamente:
> ```
> The patient presented with acute chest pain radiating to the left arm, accompanied by dyspnea and diaphoresis. ECG showed ST elevation.
> ```

---

## Situation — 0:00–0:45 (~85 palavras)

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

## Task — 0:45–1:20 (~55 palavras)

**Falar:**
- Os requisitos desta fase do desafio: API em contêiner, CI/CD automatizado,
  retreino orquestrado, observabilidade e latência otimizada.
- Cada um desses vira uma seção do Action daqui a pouco.

**Mostrar na tela:** tabela de requisitos da fase (os cinco itens acima),
pode ser o mesmo slide ou uma tabela simples.

---

## Action — 1:20–3:30 (130s — 5 sub-segmentos, somam exatamente 130s)

Este é o bloco mais apertado do roteiro: a maior parte do tempo é
demonstração ao vivo, não narração. A narração é propositalmente leve —
legenda sobre o que já está na tela, não explicação linha a linha. Se a
gravação estourar no dia, corte pela ordem de prioridade indicada (1 = corta
primeiro, 5 = não corta).

| # | Sub-segmento | Janela | Duração | Prioridade de corte |
|---|---|---|---|---|
| 1 | Arquitetura (visão geral) | 1:20–1:35 | 15s | **1 (corta primeiro)** |
| 2 | `POST /predict` ao vivo | 1:35–2:10 | 35s | **5 (nunca cortar)** |
| 3 | GitHub Actions verde *(só se publicado)* | 2:10–2:35 | 25s | **2** |
| 4 | Airflow — DAG e gate | 2:35–3:10 | 35s | **3** |
| 5 | Grafana com carga | 3:10–3:30 | 20s | **4** |

Soma: 15 + 35 + 25 + 35 + 20 = **130s = 2:10** ✓

### 1. Arquitetura (visão geral) — 1:20–1:35 (15s, corte prioridade 1, ~22 palavras)

**Falar:**
- "Visão geral: FastAPI serve o modelo em ONNX; Airflow orquestra o retreino
  semanal com gate de qualidade; Prometheus e Grafana observam tudo em
  produção."

**Mostrar:** slide ou diagrama da arquitetura (mesmo do README), 15s e corta.

### 2. `POST /predict` ao vivo — 1:35–2:10 (35s, corte prioridade 5 — nunca cortar, ~20 palavras)

**Falar (mínimo — deixar a demonstração falar por si):**
- "Rodando ao vivo. Guardem este resultado — ele volta com mais peso lá no
  Result." (ao apontar `prioridade` e `revisao_humana` na resposta)

**Mostrar:**
- 2s em `http://localhost:8000/docs` só para situar que o Swagger existe.
- Terminal, rodar (texto exato do aviso da checklist):
  ```bash
  curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"texto":"The patient presented with acute chest pain radiating to the left arm, accompanied by dyspnea and diaphoresis. ECG showed ST elevation."}'
  ```
- Apontar (sem ler em voz alta o JSON inteiro) os campos `prioridade` e
  `revisao_humana` na resposta.

### 3. GitHub Actions verde — 2:10–2:35 (25s, corte prioridade 2, ~13 palavras — só se repositório publicado)

**Falar:**
- "Quatro jobs: lint, test, build com smoke test do predict, e validate-dag."

**Mostrar:** aba Actions, os 4 jobs verdes (`lint`, `test`, `build`,
`validate-dag`).

### 4. Airflow — DAG e gate de promoção — 2:35–3:10 (35s, corte prioridade 3, ~30 palavras)

**Falar:**
- "Sete tasks: ingerir, validar, treinar, avaliar — e o gate: só promove se
  o candidato for igual ou melhor que o atual; senão, rejeitar_candidato
  encerra o pipeline sem quebrar nada."

**Mostrar:** `http://localhost:8080`, DAG `retreino_triagem`, grafo das 7
tasks.

### 5. Grafana com carga rodando — 3:10–3:30 (20s, corte prioridade 4, ~18 palavras)

**Falar:**
- "Este painel separa inferência de HTTP — a otimização do Result atacou a
  inferência, não a rede."

**Mostrar:** dashboard "Triagem de Laudos Medicos", painel **"Latencia do
modelo vs. overhead HTTP"**.

**Total de palavras faladas no bloco Action: ~103** (bem abaixo do teto de
~300 para 130s a 140 ppm — de propósito, porque a demonstração ao vivo já
ocupa o tempo visualmente).

---

## Result — 3:30–4:45 (75s — 4 sub-segmentos, somam exatamente 75s, ~175 palavras)

As três lições aprendidas abaixo são o material mais forte do projeto — é
literalmente o que o método STAR pede em "Result", e não devem ser
cortadas. Se algo tiver que ceder tempo no dia da gravação, ceda no Action
(ver prioridades de corte acima), nunca aqui.

| # | Sub-segmento | Janela | Duração |
|---|---|---|---|
| 1 | Número principal (ONNX ~3,3x) | 3:30–3:45 | 15s |
| 2 | Lição 1 — quantização sem ganho | 3:45–4:10 | 25s |
| 3 | Lição 2 — unigramas venceram bigramas | 4:10–4:25 | 15s |
| 4 | Lição 3 — a regra de segurança pegou um erro real | 4:25–4:45 | 20s |

Soma: 15 + 25 + 15 + 20 = **75s = 1:15** ✓

### 1. Número principal — 3:30–3:45 (15s, ~30 palavras)

**Falar:**
- "ONNX Runtime entrega 3,34 vezes de ganho no p50, de 0,580 para 0,174
  milissegundos, na troca de sklearn puro para ONNX — sem custo de
  acurácia: F1 praticamente igual, 0,5589 para 0,5591."

**Mostrar:** tabela comparativa de `docs/benchmarks/comparativo-latencia.md`.

### 2. Lição 1 — quantização sem ganho — 3:45–4:10 (25s, ~55 palavras)

**Falar:**
- "Primeira lição: quantização int8 não deu ganho nenhum. O grafo explica
  por quê — o classificador vira um LinearClassifier do ai.onnx.ml, que a
  quantização dinâmica nunca toca, ela só reescreve MatMul e Gemm. E
  profiling mostrou que 73% do tempo é tokenização, só 27% é o
  classificador — o teto de ganho já era baixo."

**Mostrar:** contagem de operadores do grafo ONNX (fp32 vs. int8,
idênticos).

### 3. Lição 2 — unigramas venceram bigramas — 4:10–4:25 (15s, ~35 palavras)

**Falar:**
- "Segunda lição, contraintuitiva: unigramas venceram bigramas nos dois
  eixos ao mesmo tempo — mais rápido, 0,786 para 0,614 milissegundos, e
  mais preciso, F1 de 0,5445 para 0,5589. Isso veio de medir, não de
  advinhar."

**Mostrar:** tabela comparativa de latência (mesma da etapa 1).

### 4. Lição 3 — a regra de segurança pegou um erro real — 4:25–4:45 (20s, ~55 palavras)

**Falar:**
- "Terceira lição, a mais forte: aquele mesmo laudo cardiovascular do Action
  foi classificado como condição de baixa prioridade, com confiança de
  apenas 0,3993 — abaixo do limiar de 0,40. Por isso o sistema não deixou
  passar como normal: escalou para atenção e marcou revisão humana. É o
  custo assimétrico do Situation funcionando, ao vivo, num caso real."

**Mostrar:** voltar 1s ao JSON do `/predict` do Action, apontando
`prioridade: ATENCAO` e `revisao_humana: true`.

**Total de palavras faladas no bloco Result: ~175** (dentro do teto de
~175 para 75s a 140 ppm).

---

## Fecho — 4:45–5:00 (15s, ~30 palavras)

**Falar:**
- "Próximo passo natural: monitorar drift de dados e do modelo ao longo do
  tempo, usando o histograma de confiança das predições que já está
  instrumentado no Grafana — não precisa de nada novo para começar."

**Mostrar na tela:** painel "Confianca das predicoes (p10 / p50) e
requisicoes em voo" do dashboard.

---

## Referência rápida de tempo e palavras

| Bloco | Janela | Duração | Palavras (~140 ppm) | Teto a 140 ppm |
|---|---|---|---|---|
| Situation | 0:00–0:45 | 45s | ~85 | ~105 |
| Task | 0:45–1:20 | 35s | ~55 | ~82 |
| Action | 1:20–3:30 | 2:10 (130s) | ~103 | ~303 (deliberadamente sub-usado — demo ao vivo) |
| Result | 3:30–4:45 | 1:15 (75s) | ~175 | ~175 |
| Fecho | 4:45–5:00 | 15s | ~30 | ~35 |
| **Total** | | **5:00** | **~448** | |

Todos os blocos somam exatamente 5:00 no nível superior, e agora cada bloco
também soma exatamente sua própria janela no nível dos sub-segmentos (ver
tabelas de Action e Result acima) — a aritmética não depende mais de
compensação entre blocos.
