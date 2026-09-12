# Guia prático do Airflow neste projeto

Guia de operação para quem nunca usou Airflow. Todos os comandos aqui foram
executados e verificados nesta stack, na versão **Apache Airflow 3.3.1**.

O Airflow deste projeto **já está configurado**. Este documento é sobre operá-lo,
não sobre montá-lo.

---

## 1. Subir e abrir

```bash
docker compose --profile airflow up -d
```

Depois abra **http://localhost:8080**.

Não há tela de login: a stack usa `AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_ALL_ADMINS=true`,
que entrega acesso de administrador direto. Isso é deliberado — uma tela de login
entre quem avalia e a entrega é atrito puro. **Não use essa configuração fora de
um ambiente local.**

A primeira subida leva de 1 a 2 minutos: o serviço `airflow-init` precisa rodar
`airflow db migrate` antes de os outros componentes iniciarem.

---

## 2. Os três processos, e por que isso importa

O Airflow 3 separa responsabilidades em processos distintos. Saber qual faz o quê
resolve quase todo problema de "não está funcionando".

| Processo | Função | Sintoma se ele cair |
|---|---|---|
| `airflow-dag-processor` | lê os `.py` de `airflow/dags/` e registra as DAGs | as DAGs **somem da interface, sem nenhuma mensagem de erro** |
| `airflow-scheduler` | decide o que executar e quando | tudo aparece na tela, nada executa |
| `airflow-apiserver` | serve a interface web e a API REST | você não enxerga nada, mas as tarefas continuam rodando |

No Airflow 2 o processador de DAGs vivia dentro do scheduler. Na versão 3 ele é um
processo separado e obrigatório — omiti-lo do `docker-compose.yml` é a causa mais
comum de "minha DAG não aparece".

Conferir se os três estão vivos:

```bash
curl -s http://localhost:8080/api/v2/monitor/health
```

Espera-se `healthy` em `metadatabase`, `scheduler` e `dag_processor`. O `triggerer`
vem nulo de propósito: ele só é necessário para operadores adiáveis, que este
projeto não usa.

---

## 3. Vocabulário mínimo

- **DAG** — o pipeline inteiro. Aqui: `retreino_triagem`.
- **Task** — um passo do pipeline. Aqui são 7.
- **DAG Run** — uma execução do pipeline, identificada por um `run_id`
  como `scheduled__2026-09-06T00:00:00+00:00`.
- **Task Instance** — um passo dentro de uma execução específica. É o que fica
  colorido na tela.

Estados que você vai ver:

| Estado | Significado |
|---|---|
| `success` | terminou bem |
| `failed` | quebrou e esgotou as tentativas |
| `up_for_retry` | quebrou e vai tentar de novo |
| `skipped` | pulada de propósito, por decisão do fluxo |
| `queued` | esperando vaga para executar |
| `upstream_failed` | não rodou porque algo antes dela falhou |

`skipped` **não é erro**. Nesta DAG, `rejeitar_candidato` aparece como pulada
sempre que o modelo é aprovado — é o comportamento correto.

---

## 4. As operações que você vai realmente usar

### Disparar uma execução na mão

Na interface: botão **Acionar**, canto superior direito da página da DAG.

Pela linha de comando:

```bash
docker compose exec airflow-scheduler airflow dags trigger retreino_triagem
```

Uma execução completa leva **cerca de 45 segundos** com o dataset já em cache
(medido: ingestão 14s, validação 2s, treino 15s, avaliação 2s, export ONNX 4s,
promoção 2s). Só a primeira vez é mais lenta, por causa do download de 14 MB.

### Ler o log de uma tarefa

Na interface: clique na tarefa, depois na aba de logs.

**É onde o erro real aparece.** Nenhuma outra tela conta o que de fato quebrou.
Quando algo falha, o log é o primeiro lugar a olhar, não o último.

### Limpar uma tarefa para reexecutá-la

Esta é a operação mais útil do Airflow e a menos óbvia para quem está começando.

Limpar uma tarefa faz o Airflow tratá-la como se nunca tivesse rodado. O scheduler
então reexecuta **apenas ela**, aproveitando os resultados das tarefas anteriores.
Você conserta um passo que falhou sem refazer o pipeline inteiro.

Na interface: clique na tarefa e escolha limpar.

Pela linha de comando:

```bash
docker compose exec airflow-scheduler \
  airflow tasks clear retreino_triagem -t promover_modelo -y -s 2026-09-05 -e 2026-09-08
```

As flags: `-t` filtra a tarefa, `-y` confirma sem perguntar, `-s` e `-e` delimitam
a janela de datas lógicas. Acrescente `-d` para limpar também as tarefas
posteriores, e `-f` para limpar somente as que falharam.

### Consultar estado sem abrir o navegador

```bash
# todas as DAGs registradas
docker compose exec airflow-scheduler airflow dags list

# execuções de uma DAG
docker compose exec airflow-scheduler airflow dags list-runs retreino_triagem

# estado de cada tarefa de uma execução
docker compose exec airflow-scheduler \
  airflow tasks states-for-dag-run retreino_triagem scheduled__2026-09-06T00:00:00+00:00

# DAGs que falharam ao importar
docker compose exec airflow-scheduler airflow dags list-import-errors
```

> **No Git Bash do Windows:** comandos que passam caminhos absolutos do container
> (`/opt/airflow/...`) são mastigados pela conversão automática de caminhos. Prefixe
> com `MSYS_NO_PATHCONV=1` nesses casos. Os comandos acima não precisam.

---

## 5. A DAG deste projeto

```
ingerir_dados → validar_dados → treinar_modelo → avaliar_modelo ─┬─ aprovado  → exportar_onnx → promover_modelo
                                                                 └─ reprovado → rejeitar_candidato
```

| Tarefa | O que faz |
|---|---|
| `ingerir_dados` | baixa o corpus (cacheado em `data/`) |
| `validar_dados` | falha alto se vierem poucas linhas ou faltar alguma classe |
| `treinar_modelo` | treina o pipeline TF-IDF + regressão logística |
| `avaliar_modelo` | compara o F1 do candidato com o do modelo em produção |
| `exportar_onnx` | converte para ONNX, quantiza e confere a paridade das predições |
| `promover_modelo` | copia os artefatos para `models/current/` |
| `rejeitar_candidato` | registra a rejeição sem derrubar a execução |

O `avaliar_modelo` é o coração do pipeline: ele **só deixa promover se o modelo novo
não for pior** que o que está em produção. Quando aprova, `rejeitar_candidato` fica
pulada. Se um dia o retreino piorar o modelo, os papéis se invertem — `promover_modelo`
é que fica pulada, e produção segue intocada.

O `exportar_onnx` usa `assert_parity`, que levanta exceção se o modelo ONNX
discordar do pipeline scikit-learn em mais de 2% das predições. Uma conversão
defeituosa nunca chega a produção.

**A API não recarrega o modelo sozinha.** Ela carrega uma vez na inicialização.
Depois de uma promoção, rode `docker compose restart api` para servir o modelo novo.

---

## 6. Quando algo dá errado

**A DAG não aparece na lista.**
Veja o log do processador: `docker compose logs airflow-dag-processor | tail -30`.
Erro de sintaxe ou de import no arquivo da DAG impede o registro. Confirme também
com `airflow dags list-import-errors`.

**A tarefa fica presa em `queued` e nunca sai.**
O scheduler não está processando. Verifique `docker compose ps` e o endpoint de
saúde. Se você trocou o `AIRFLOW__API_AUTH__JWT_SECRET` e reiniciou só um dos
componentes, eles param de se autenticar entre si — recrie os três juntos.

**Uma tarefa falha e você quer saber por quê.**
Sempre o log da tarefa. Ele traz o traceback completo, com o arquivo e a linha.

**Alterei o arquivo da DAG e nada mudou.**
A pasta é montada por bind mount, então a alteração chega em segundos. Se não
chegar, o processador pode ter travado: `docker compose restart airflow-dag-processor`.

---

## 7. Armadilhas que este projeto já encontrou

**A data lógica não é a data de execução.** Uma execução chamada
`scheduled__2026-09-06` pode ter rodado no dia 12. O nome identifica o *período*
coberto, não o instante em que aconteceu. Praticamente todo iniciante tropeça nisso.

**`docker compose down` não derruba o Airflow.** Sem `--profile airflow`, os
containers do profile continuam de pé segurando a porta 8080. Use
`docker compose --profile airflow down`.

**`shutil.copy` e `copy2` falham ao escrever em `models/` pelo bind mount.** As duas
aplicam `chmod` no destino, e `chmod` exige ser dono do arquivo. Sob o bind mount os
artefatos pertencem ao uid 0 enquanto o Airflow roda como uid 50000: a escrita do
conteúdo passa, o `chmod` levanta `EPERM`. A promoção usa `shutil.copyfile`, que
copia só os bytes. Foi um bug real deste projeto, diagnosticado isolando cada
operação de arquivo dentro do container.

**No Linux, `data/` e `airflow/logs/` precisam existir antes do primeiro `up`.**
São ignorados pelo git; o Docker os cria pertencendo ao root, e o Airflow (uid 50000)
não consegue escrever. Veja as duas linhas de preparação no README.

---

## 8. Antes de gravar o vídeo

1. Suba a stack completa e espere os três componentes ficarem saudáveis.
2. Confirme que a última execução está verde:
   `docker compose exec airflow-scheduler airflow dags list-runs retreino_triagem`
3. Deixe aberta a página da DAG, na aba de execuções.
4. Uma execução ao vivo leva cerca de 45 segundos com o dataset em cache, o que
   **cabe confortavelmente** no bloco de Airflow do roteiro. Se preferir não
   depender da rede e do relógio na hora, dispare antes e mostre o resultado
   pronto — as duas opções funcionam.

O que vale mostrar na tela: o grafo com as 7 tarefas, o `rejeitar_candidato`
pulado (que evidencia o gate de qualidade tendo aprovado o modelo) e o log de
`avaliar_modelo`, onde aparece a comparação de F1 que autorizou a promoção.
