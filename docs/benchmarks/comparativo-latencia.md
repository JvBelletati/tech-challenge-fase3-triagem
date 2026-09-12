# Comparativo de latencia — Etapa 4

Medido em 2026-09-12 | 400 inferencias por variante | uma requisicao por vez, CPU, `intra_op_num_threads=1`.

**F1-macro medido no conjunto de testes retido (held-out test set)** para evitar vazamento de dados e garantir que os valores sejam comparáveis com `metadata.json`.

| Variante | p50 (ms) | p95 (ms) | p99 (ms) | F1-macro | Tamanho (MB) | Ganho vs baseline |
|---|---|---|---|---|---|---|
| sklearn | 0.570 | 0.697 | 0.831 | 0.5445 | 1.22 | 1.00x |
| onnx-fp32 | 0.173 | 0.225 | 0.269 | 0.5427 | 0.80 | 3.29x |
| onnx-int8 | 0.173 | 0.246 | 0.298 | 0.5427 | 0.80 | 3.29x |

### Operadores do grafo ONNX

- **onnx-fp32**: `{'Reshape': 1, 'StringNormalizer': 1, 'Tokenizer': 1, 'Flatten': 1, 'TfIdfVectorizer': 1, 'Add': 1, 'Log': 1, 'Mul': 1, 'Normalizer': 2, 'LinearClassifier': 1}`
- **onnx-int8**: `{'Reshape': 1, 'StringNormalizer': 1, 'Tokenizer': 1, 'Flatten': 1, 'TfIdfVectorizer': 1, 'Add': 1, 'Log': 1, 'Mul': 1, 'Normalizer': 2, 'LinearClassifier': 1}`
