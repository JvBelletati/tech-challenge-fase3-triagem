# Comparativo de latencia — Etapa 4

Medido em 2026-09-12 | 400 inferencias por variante | uma requisicao por vez, CPU, `intra_op_num_threads=1`.

| Variante | p50 (ms) | p95 (ms) | p99 (ms) | F1-macro | Tamanho (MB) | Ganho vs baseline |
|---|---|---|---|---|---|---|
| sklearn | 0.646 | 1.002 | 1.185 | 0.7662 | 1.22 | 1.00x |
| onnx-fp32 | 0.177 | 0.224 | 0.248 | 0.7662 | 0.80 | 3.66x |
| onnx-int8 | 0.175 | 0.229 | 0.351 | 0.7662 | 0.80 | 3.70x |

### Operadores do grafo ONNX

- **onnx-fp32**: `{'Reshape': 1, 'StringNormalizer': 1, 'Tokenizer': 1, 'Flatten': 1, 'TfIdfVectorizer': 1, 'Add': 1, 'Log': 1, 'Mul': 1, 'Normalizer': 2, 'LinearClassifier': 1}`
- **onnx-int8**: `{'Reshape': 1, 'StringNormalizer': 1, 'Tokenizer': 1, 'Flatten': 1, 'TfIdfVectorizer': 1, 'Add': 1, 'Log': 1, 'Mul': 1, 'Normalizer': 2, 'LinearClassifier': 1}`
