# Truncation of reranker inputs

- Sampled queries: 500
- Candidate documents: 48638
- Tokenized documents: 50000

| Input | n | median | p95 | p99 | max | above 512 | above 300 | above 8192 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bge combined (query + document) | 50000 | 109.0 | 268 | 397 | 1376 | 0.29% | 0.00% | 0.00% |
| ColBERT document | 50000 | 99.0 | 258 | 386 | 1363 | 0.00% | 2.80% | 0.00% |
| jina-reranker-v3 document | 50000 | 112.0 | 292 | 436 | 1576 | 0.00% | 0.00% | 0.00% |

## Query lengths

| Tokenizer | median | p95 | max | above 32 | above 1024 |
| --- | ---: | ---: | ---: | ---: | ---: |
| bge-reranker-v2-m3 | 10.0 | 14 | 25 | 0.00% | 0.00% |
| jina-reranker-v3 | 8.0 | 14 | 28 | 0.00% | 0.00% |
| jina-colbert-v2 | 10.0 | 14 | 25 | 0.00% | 0.00% |
