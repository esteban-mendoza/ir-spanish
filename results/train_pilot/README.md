# Piloto de selección sobre la partición de entrenamiento

Piloto **exploratorio** para validar la tubería de selección sobre entrenamiento antes de repetirla en el corpus completo. Usa un corpus cerrado de 10 000 artículos de la partición de entrenamiento de MessIRve; las métricas absolutas están infladas por el corpus reducido y **no son evidencia para la tesis**. La decisión definitiva requiere el corpus completo y los seis modelos.

## Composición

- **Artículos:** 10 000, muestreados con semilla 42 de los artículos con documentos relevantes en entrenamiento.
- **Documentos:** 317 750 párrafos (todos los de los artículos muestreados).
- **Consultas:** 156 949, las de entrenamiento cuyos documentos relevantes caen por completo en esos artículos.
- **Juicios:** 161 644.
- **Filtro:** `match_score == 1` y `expanded_search == False`, el mismo criterio que la partición de prueba.
- **Modelos incluidos:** `jina-v5-small`, `bge-m3`, `e5-large` y `Qwen3-Embedding-0.6B`. BM25 y `splade-v3` quedaron fuera por problemas de entorno (véase «Límites»).

La partición vive en `~/.cache/messirve_embeddings/pilot`; no se versiona.

## Comandos

```bash
cd ~/ir-spanish
export MESSIRVE_PILOT=$HOME/.cache/messirve_embeddings/pilot

# 1. Construir la partición cerrada (una vez)
python -m scripts.build_train_pilot --out-dir "$MESSIRVE_PILOT"

# 2. Correr los seis modelos (el runner se detuvo en BM25; ver límites)
bash scripts/run_train_pilot.sh

# 3. Analizar los cuatro modelos densos
bash scripts/run_train_pilot_analysis.sh
```

BM25 requiere JDK 21 en este servidor:

```bash
export JAVA_HOME=$HOME/jdk21/jdk-21.0.13+9
export PATH=$JAVA_HOME/bin:$PATH
```

## Resultados

### Modelos individuales

| # | modelo | nDCG@10 | Recall@100 |
| ---: | --- | ---: | ---: |
| 1 | `jina-v5-small` | 0.6075 | 0.9758 |
| 2 | `bge-m3` | 0.5629 | 0.9477 |
| 3 | `e5-large` | 0.5452 | 0.9513 |
| 4 | `Qwen3-Embedding-0.6B` | 0.5317 | 0.9307 |

### Mejores fusiones con CombMNZ

| # | combinación | nDCG@10 | Recall@100 | Δ vs. la mejor |
| ---: | --- | ---: | ---: | ---: |
| 1 | `e5-large` + `bge-m3` + `jina-v5-small` | **0.6253** | **0.9809** | — |
| 2 | `bge-m3` + `jina-v5-small` | 0.6210 | 0.9789 | −0.0043 |
| 3 | `Qwen3` + `e5-large` + `bge-m3` + `jina-v5-small` | 0.6207 | 0.9798 | −0.0047 |
| 4 | `e5-large` + `jina-v5-small` | 0.6172 | 0.9792 | −0.0081 |
| 5 | `Qwen3` + `bge-m3` + `jina-v5-small` | 0.6131 | 0.9774 | −0.0122 |

El barrido completo de los 11 subconjuntos densos está en `sweep/results.csv`.

### Algoritmos sobre el trío

| # | algoritmo | nDCG@10 | Recall@100 |
| ---: | --- | ---: | ---: |
| 1 | CombMNZ | **0.6253** | **0.9809** |
| 2 | RBC | 0.6220 | 0.9799 |
| 3 | RRF | 0.6186 | 0.9806 |
| 4 | ISR | 0.6163 | 0.9806 |

## Coincidencia con la selección de prueba

- El top-3 individual es el trío denso: **sí**.
- El mejor modelo individual es `jina-v5-small`: **sí**.
- El trío es la mejor fusión con CombMNZ: **sí**.
- CombMNZ es el mejor de los cuatro algoritmos: **sí**.

Es una señal coherente, no una conclusión: el corpus cerrado puede cambiar el orden y faltan BM25 y SPLADE.

## Límites y pendientes

- **Corpus cerrado:** los 317 750 documentos son el 2.3 % del corpus completo; las métricas absolutas no son comparables con las de prueba.
- **BM25:** el servidor tiene Java 11 y Pyserini exige `jdk.incubator.vector`. Se instaló Temurin 21 en `~/jdk21` y Pyserini arranca, pero el *run* del piloto no se guardó. Además, `baselines/bm25.py` construye su ruta de caché sin `cache.cache_base`, así que no respeta el sufijo del piloto; hay que corregirlo antes de la fase definitiva.
- **SPLADE:** la recuperación terminó (156 949 consultas), pero `retrieval_run.lz4` quedó en 0 bytes y el proceso se colgó al guardar; se detuvo y se eliminó el archivo vacío. Los embeddings de documentos y consultas del piloto quedaron en `~/.cache/messirve_embeddings/naver__splade-v3/..._pilot/`. Hay que revisar el guardado del *run* antes de la fase definitiva.
- **Fase definitiva:** repetir la selección con el corpus completo y los seis modelos; solo entonces se puede reescribir la metodología y los resultados de la tesis.
