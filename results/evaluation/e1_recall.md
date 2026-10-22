| fusión | Recall@100 | Δ vs. jina | IC 95 % | ¿incluye 0? |
| --- | ---: | ---: | --- | :---: |
| CombMNZ | 0.9231 | +0.0192 | [0.0183, 0.0202] | no |
| RBC | 0.9206 | +0.0167 | [0.0157, 0.0177] | no |
| RRF | 0.9234 | +0.0195 | [0.0185, 0.0205] | no |
| Condorcet | 0.8918 | -0.0120 | [-0.0131, -0.0109] | no |
| BordaFuse | 0.9234 | +0.0195 | [0.0186, 0.0205] | no |
| ISR | 0.9230 | +0.0191 | [0.0181, 0.0201] | no |

Referencia: jina-v5-small (0.9039); Δ = fusión − jina. El IC es el de la t pareada por consulta (170055 consultas). La comparación principal es CombMNZ: p de la t = < 0.0001; p de la permutación = < 0.0001; p de Holm sobre nDCG@10 y Recall@100 = < 0.0001.
Comprobación de sanidad: el IC de nDCG@10 de CombMNZ frente a jina es [0.0391, 0.0413].
