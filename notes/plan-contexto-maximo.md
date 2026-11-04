# Plan: reordenadores con contexto máximo

**Estado:** código preparado; ejecución pendiente de que el servidor IIMAS recupere las GPU
(véase `notes/incidencia-gpu-2026-11-03.md`).

## Qué cambia

| Modelo | Contexto nativo | Configuración anterior | Configuración nueva |
| --- | --- | --- | --- |
| `bge-reranker-v2-m3` | 8 192 tokens para consulta + documento (`max_position_embeddings = 8194`) | 512 combinados | 8 192 combinados |
| `jina-reranker-v3` | 131 072 compartidos; 8 192 por documento (artículo, tabla 5) | 1 024 consulta / 8 192 documento | sin cambio: ya usa el máximo por documento y la ventana compartida |
| `jina-colbert-v2` | 8 192 tokens por secuencia | 32 consulta / 300 documento | 32 consulta / 8 192 documento |

La consulta de ColBERT se queda en 32 tokens porque es su presupuesto de expansión
con tokens `[MASK]`, no un límite de contexto; la consulta más larga de MessIRve tiene
22 palabras. En `jina-reranker-v3` el tope de 1 024 también supera esa longitud y el
modelo empaqueta los 100 candidatos en su ventana de 131 072; por eso no se reejecuta.

## Memoria

Un lote de 8 192 tokens no cabe con los tamaños anteriores: la atención crece con
`lote × longitud²`. `rerankers/context.py` agrupa los pares por longitud estimada y
cierra cada lote cuando `lote × longitud_max²` supera 2²⁸ tokens² (unos 8.6 GiB de
atención con 16 cabezas en `float32`). Los documentos cortos conservan lotes grandes;
solo los largos usan lotes de 2 a 4. El estimador usa 1.4 tokens por palabra para no
quedarse corto.

## Corrida

En el servidor, con la copia sincronizada:

```bash
cd ~/ir-spanish
git pull --ff-only
setsid nohup bash scripts/run_maxctx_rerankers.sh \
  > logs/maxctx_rerankers.log 2>&1 < /dev/null &
```

El script es reanudable: cada modelo guarda puntos de control cada 200 consultas y
salta la corrida completa si ya existe `retrieval_run.lz4` en la ruta nueva. Las rutas
nuevas son:

- `BAAI__bge-reranker-v2-m3/.../full_v1.2_q8192_d8192_nofilter/`
- `jinaai__jina-colbert-v2/.../full_v1.2_q32_d8192_nofilter/`

Antes de lanzar la corrida completa conviene una prueba de humo con dos consultas y
una ruta aparte:

```bash
RERANKER_LIMIT=2 RERANKER_RUN_TAG=_smoke \
  python -m rerankers.bge_reranker_v2_m3
```

La etiqueta es obligatoria: sin `RERANKER_RUN_TAG` el script se niega a escribir una
corrida parcial en la ruta de producción.

## Costo esperado

Las corridas anteriores, con los topes cortos, tardaron 60.2 h (`bge`, 512), 25.3 h
(`jina-reranker-v3`, 1 024/8 192) y 23.0 h (`ColBERT`, 32/300) en dos A5000. El
aumento no es proporcional al tope: la mediana de los documentos es de 68 palabras y
el P99 de 245, así que solo la cola larga cambia. Aun así, `bge` y `ColBERT` pueden
tardar varios días; hay que planear la corrida como un trabajo de fondo y no bloquear
la sesión.

## Después de la corrida

1. Evaluar efectividad y significancia:

   ```bash
   bash scripts/evaluate_maxctx_e3.sh
   ```

   Escribe `results/evaluation/e3_maxctx.*` y `e3_maxctx_recall.*`; los archivos
   anteriores no se sobrescriben.

2. Rehacer el banco de eficiencia con las longitudes nuevas:
   `evaluation/efficiency.py` ya toma `BGE_MAX_LENGTH`, `BGE_BATCH_SIZE`,
   `COLBERT_MAX_DOC_LENGTH` y `COLBERT_DOC_BATCH` del entorno. Hay que decidir si se
   mantiene el lote 32 para el *throughput* o se reduce a 8, porque a 8 192 tokens el
   lote 32 no cabe.

3. Medir la truncación antes/después:

   ```bash
   python -m scripts.check_reranker_lengths --queries 500
   ```

   Escribe `results/reranker_context/truncation.{json,md}` con el porcentaje de
   documentos por encima de 512, 300 y 8 192 tokens.

4. Actualizar `results.md`, la tesis y las conclusiones. La lista de pasajes está en
   `tesis/notes/todo.md`, entrada «Actualizar toda la tesis con los resultados de
   contexto máximo».

## Por qué no se reejecuta `jina-reranker-v3`

El modelo tiene 131 072 tokens de contexto compartido y un máximo de 8 192 por
documento. La corrida anterior ya usaba 8 192 por documento, y el código del modelo
empaqueta los 100 candidatos en la misma ventana hasta llenar la capacidad; el tope de
1 024 para la consulta excede las 22 palabras de la consulta más larga. Reejecutarlo
con topes nominales más altos produciría la misma entrada efectiva y consumiría otro
día de GPU.
