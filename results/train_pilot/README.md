# Piloto de selección sobre la partición de entrenamiento

Este piloto es el **entorno de selección** de los sistemas de la tesis. Usa un corpus cerrado de 10 000 artículos de la partición de entrenamiento de MessIRve; la partición de prueba se usa para evaluar y para el análisis descriptivo de estabilidad, no para elegir. No se usa la partición de entrenamiento completa.

Las métricas absolutas del piloto no son comparables con las de prueba porque el corpus es el 2.3 % del original; la selección se apoya en el **orden** de los sistemas y en el análisis de estabilidad de `stability.md`.

## Composición

- **Artículos:** 10 000, muestreados con semilla 42 de los artículos con documentos relevantes en entrenamiento.
- **Documentos:** 317 750 párrafos (todos los de los artículos muestreados).
- **Consultas:** 156 949, las de entrenamiento cuyos documentos relevantes caen por completo en esos artículos.
- **Juicios:** 161 644.
- **Filtro:** `match_score == 1` y `expanded_search == False`, el mismo criterio que la partición de prueba.
- **Modelos:** los seis de la tesis: `jina-v5-small`, `bge-m3`, `e5-large`, `Qwen3-Embedding-0.6B`, `splade-v3` y `BM25`.

La partición vive en `~/.cache/messirve_embeddings/pilot`; no se versiona.

## Regla de selección

1. **Modelo de primera etapa:** el de mayor nDCG@10 individual en el piloto.
2. **Tres candidatos:** los tres modelos con mayor nDCG@10 individual.
3. **Algoritmo de referencia:** CombMNZ, adoptado *a priori* y comprobado contra los otros cinco sobre el trío.
4. **Confirmación:** los 57 subconjuntos con CombMNZ.
5. **Reordenadores:** se reportan los tres; no se elige uno.
6. **Evidencia:** solo el piloto; la prueba se usa para evaluar y para medir la estabilidad.

## Comandos

```bash
cd ~/ir-spanish
export MESSIRVE_PILOT=$HOME/.cache/messirve_embeddings/pilot
export JAVA_HOME=$HOME/jdk21/jdk-21.0.13+9
export PATH=$JAVA_HOME/bin:$PATH

# 1. Construir la partición cerrada (una vez)
python -m scripts.build_train_pilot --out-dir "$MESSIRVE_PILOT"

# 2. Correr los seis modelos
bash scripts/run_train_pilot.sh

# 3. Barrido de 57 subconjuntos, seis algoritmos, informe y estabilidad
bash scripts/run_train_pilot_analysis.sh
```

Las matrices de nDCG@10 por consulta se guardan en
`~/.cache/messirve_embeddings/train_selection/per_query`; no se versionan.

## Resultados

### Modelos individuales

| # | modelo | nDCG@10 | Recall@100 |
| ---: | --- | ---: | ---: |
| 1 | `jina-v5-small` | 0.6075 | 0.9758 |
| 2 | `bge-m3` | 0.5629 | 0.9477 |
| 3 | `e5-large` | 0.5452 | 0.9513 |
| 4 | `Qwen3-Embedding-0.6B` | 0.5317 | 0.9307 |
| 5 | `splade-v3` | 0.2380 | 0.6928 |
| 6 | `BM25` | 0.2251 | 0.6779 |

### Mejores fusiones con CombMNZ

| # | combinación | nDCG@10 | Recall@100 | Δ vs. la mejor | IC 95 % |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | `e5-large` + `bge-m3` + `jina-v5-small` | **0.6253** | **0.9809** | — | — |
| 2 | `bge-m3` + `jina-v5-small` | 0.6210 | 0.9789 | −0.0043 | [−0.0050, −0.0036] |
| 3 | `Qwen3` + `e5-large` + `bge-m3` + `jina-v5-small` | 0.6207 | 0.9798 | −0.0047 | [−0.0052, −0.0041] |
| 4 | `e5-large` + `jina-v5-small` | 0.6172 | 0.9792 | −0.0081 | [−0.0088, −0.0074] |
| 5 | `Qwen3` + `bge-m3` + `jina-v5-small` | 0.6131 | 0.9774 | −0.0122 | [−0.0130, −0.0113] |

El barrido completo de los 57 subconjuntos está en `sweep_all/results.csv`. La ventaja del trío es distinguible de cero, pero menor que el umbral práctico de 0.01; la regla del top-3 individual y la simplicidad (tres modelos) resuelven el empate práctico.

### Algoritmos sobre el trío

| # | algoritmo | nDCG@10 | Recall@100 |
| ---: | --- | ---: | ---: |
| 1 | CombMNZ | **0.6253** | **0.9809** |
| 2 | RBC | 0.6220 | 0.9799 |
| 3 | RRF | 0.6186 | 0.9806 |
| 4 | Condorcet | 0.6183 | 0.9678 |
| 5 | ISR | 0.6163 | 0.9806 |
| 6 | BordaFuse | 0.6154 | 0.9806 |

### Estabilidad piloto–prueba

| Bloque | Spearman | Kendall | Coincidencia |
| --- | ---: | ---: | --- |
| Modelos individuales | 1.0000 | 1.0000 | mismo orden; top-3 3/3 |
| Fusiones (57) | 0.9726 | 0.8784 | top-1 1/1; top-3 3/3; top-5 3/5 |
| Algoritmos (6) | 0.9429 | 0.8667 | CombMNZ mejor en ambos |

El top-5 de prueba incluye dos combinaciones mixtas de cinco modelos con SPLADE (0.5400 y 0.5382) que el piloto coloca más abajo; quedan por debajo del trío y no afectan la selección.

## Lectura

- El **trío** `e5-large` + `bge-m3` + `jina-v5-small` es la mejor fusión de las 57 y también la mejor en prueba; **CombMNZ** es el mejor de los seis algoritmos en ambos.
- El **top-3 individual** del piloto es el mismo que el de prueba; `jina-v5-small` es el mejor en ambos.
- El piloto reproduce el orden de la evaluación completa (Spearman 0.9726 en las fusiones, 1.0000 en los individuales); por eso puede sostener la selección sin usar prueba.
- **Limitación residual:** el corpus cerrado (2.3 %) y la diferencia del top-5; la transferencia se mide, no se garantiza.

## Archivos

- `selection.md` / `selection.json`: informe de selección, tabla de 57 subconjuntos e IC.
- `stability.md` / `stability.json`: comparación piloto–prueba.
- `sweep_all/`: barrido de 57 subconjuntos con CombMNZ.
- `algorithm_check_six/`: los seis algoritmos sobre el trío.
- `single_models.csv`: métricas individuales del piloto.
- `../test_selection/single_models.csv`: métricas individuales de prueba usadas en la estabilidad.
