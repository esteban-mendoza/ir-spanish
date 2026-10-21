| combinación | estrategia | parámetros | nDCG@10 | Recall@100 | MRR@10 | MAP | P@10 | P@50 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| e5-large+bge-m3+jina-v5-small | bordafuse |  | 0.541391 | 0.923417 | 0.477594 | 0.483102 | 0.076436 | 0.018288 |
| qwen3-0.6b+e5-large+bge-m3+jina-v5-small | bordafuse |  | 0.536307 | 0.922839 | 0.472515 | 0.478187 | 0.075928 | 0.018240 |
| splade-v3+qwen3-0.6b+e5-large+bge-m3+jina-v5-small | bordafuse |  | 0.496202 | 0.923658 | 0.425245 | 0.432224 | 0.074304 | 0.018292 |
| splade-v3+e5-large+jina-v5-small | bordafuse |  | 0.465765 | 0.911896 | 0.394143 | 0.402290 | 0.071478 | 0.018046 |
| splade-v3+jina-v5-small | bordafuse |  | 0.385805 | 0.890438 | 0.320443 | 0.331777 | 0.061305 | 0.017326 |
| e5-large+bge-m3+jina-v5-small | combmnz |  | 0.551695 | 0.923079 | 0.487832 | 0.492897 | 0.077490 | 0.018330 |
| qwen3-0.6b+e5-large+bge-m3+jina-v5-small | combmnz |  | 0.548198 | 0.922795 | 0.484280 | 0.489477 | 0.077154 | 0.018281 |
| splade-v3+qwen3-0.6b+e5-large+bge-m3+jina-v5-small | combmnz |  | 0.540022 | 0.922500 | 0.474232 | 0.479636 | 0.076960 | 0.018322 |
| bm25+splade-v3+qwen3-0.6b+e5-large+bge-m3+jina-v5-small | combmnz |  | 0.527708 | 0.925638 | 0.460041 | 0.466007 | 0.076319 | 0.018366 |
| splade-v3+e5-large+jina-v5-small | combmnz |  | 0.513702 | 0.911237 | 0.446500 | 0.452435 | 0.074795 | 0.018044 |
| splade-v3+jina-v5-small | combmnz |  | 0.432576 | 0.890896 | 0.368702 | 0.377692 | 0.065515 | 0.017341 |
| e5-large+bge-m3+jina-v5-small | isr |  | 0.538234 | 0.922969 | 0.474260 | 0.479800 | 0.076312 | 0.018280 |
| qwen3-0.6b+e5-large+bge-m3+jina-v5-small | isr |  | 0.536385 | 0.922180 | 0.472889 | 0.478511 | 0.075962 | 0.018231 |
| splade-v3+qwen3-0.6b+e5-large+bge-m3+jina-v5-small | isr |  | 0.530395 | 0.920012 | 0.468687 | 0.474575 | 0.074844 | 0.018146 |
| bm25+splade-v3+qwen3-0.6b+e5-large+bge-m3+jina-v5-small | isr |  | 0.523380 | 0.921052 | 0.462999 | 0.469322 | 0.073760 | 0.018126 |
| splade-v3+e5-large+jina-v5-small | isr |  | 0.505939 | 0.911020 | 0.444286 | 0.450745 | 0.072474 | 0.017906 |
| splade-v3+jina-v5-small | isr |  | 0.443734 | 0.890475 | 0.376382 | 0.384193 | 0.067760 | 0.017338 |
| e5-large+bge-m3+jina-v5-small | rbc | phi=0.95 | 0.548529 | 0.920561 | 0.483517 | 0.488425 | 0.077539 | 0.018214 |
| qwen3-0.6b+e5-large+bge-m3+jina-v5-small | rbc | phi=0.95 | 0.546213 | 0.919286 | 0.481030 | 0.486037 | 0.077361 | 0.018184 |
| splade-v3+qwen3-0.6b+e5-large+bge-m3+jina-v5-small | rbc | phi=0.95 | 0.539612 | 0.915654 | 0.471997 | 0.476944 | 0.077437 | 0.018108 |
| bm25+splade-v3+qwen3-0.6b+e5-large+bge-m3+jina-v5-small | rbc | phi=0.95 | 0.530871 | 0.916429 | 0.460857 | 0.466049 | 0.077299 | 0.018132 |
| splade-v3+e5-large+jina-v5-small | rbc | phi=0.95 | 0.512892 | 0.907440 | 0.442837 | 0.447751 | 0.075488 | 0.017819 |
| splade-v3+jina-v5-small | rbc | phi=0.95 | 0.429539 | 0.889917 | 0.358757 | 0.366806 | 0.067490 | 0.017306 |
| e5-large+bge-m3+jina-v5-small | rrf | k=60 | 0.545026 | 0.923376 | 0.480628 | 0.485914 | 0.076997 | 0.018299 |
| qwen3-0.6b+e5-large+bge-m3+jina-v5-small | rrf | k=60 | 0.541206 | 0.922888 | 0.476605 | 0.482001 | 0.076679 | 0.018255 |
| splade-v3+qwen3-0.6b+e5-large+bge-m3+jina-v5-small | rrf | k=60 | 0.518224 | 0.923800 | 0.447799 | 0.453640 | 0.076267 | 0.018318 |
| bm25+splade-v3+qwen3-0.6b+e5-large+bge-m3+jina-v5-small | rrf | k=60 | 0.498491 | 0.926960 | 0.424947 | 0.431663 | 0.075299 | 0.018362 |
| splade-v3+e5-large+jina-v5-small | rrf | k=60 | 0.482600 | 0.911892 | 0.410575 | 0.417643 | 0.073263 | 0.018054 |
| splade-v3+jina-v5-small | rrf | k=60 | 0.398941 | 0.890441 | 0.333883 | 0.344408 | 0.062564 | 0.017297 |
