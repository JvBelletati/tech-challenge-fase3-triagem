# Comparativo de latencia — Etapa 4

Medido em 2026-09-12 | 400 inferencias por variante | uma requisicao por vez, CPU, `intra_op_num_threads=1`.

**F1-macro medido nas 2310 linhas completas do conjunto de testes retido (held-out test set)** para evitar vazamento de dados. Por cobrir o held-out inteiro, os valores de sklearn desta tabela reconciliam exatamente com `metadata.json` (mesmo split, mesmas linhas).

| Variante | p50 (ms) | p95 (ms) | p99 (ms) | F1-macro | Tamanho (MB) | Ganho vs baseline |
|---|---|---|---|---|---|---|
| sklearn | 0.580 | 0.826 | 0.933 | 0.5589 | 1.22 | 1.00x |
| onnx-fp32 | 0.174 | 0.225 | 0.243 | 0.5591 | 0.80 | 3.34x |
| onnx-int8 | 0.177 | 0.223 | 0.251 | 0.5591 | 0.80 | 3.28x |

### Operadores do grafo ONNX

- **onnx-fp32**: `{'Reshape': 1, 'StringNormalizer': 1, 'Tokenizer': 1, 'Flatten': 1, 'TfIdfVectorizer': 1, 'Add': 1, 'Log': 1, 'Mul': 1, 'Normalizer': 2, 'LinearClassifier': 1}`
- **onnx-int8**: `{'Reshape': 1, 'StringNormalizer': 1, 'Tokenizer': 1, 'Flatten': 1, 'TfIdfVectorizer': 1, 'Add': 1, 'Log': 1, 'Mul': 1, 'Normalizer': 2, 'LinearClassifier': 1}`
