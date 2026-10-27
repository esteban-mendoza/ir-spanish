# Changelog — `ir-spanish`

Registro cronológico **autoritativo** de cambios y decisiones de `ir-spanish/` (el código de los experimentos). Las rutas de este archivo son relativas a la raíz de `ir-spanish/`.

El trabajo se realiza con varios agentes que se alternan: **asume que tu contexto de conversación puede estar descontinuado**. Ante cualquier duda entre tu memoria y este archivo (o el contenido real del repositorio), este archivo y el repositorio mandan.

**Alcance.** Este changelog registra **solo** cambios de `ir-spanish/`. Los cambios de `tesis/` van en `tesis/notes/changelog.md`. Si un cambio toca los dos repositorios, se anota en ambos.

Protocolo obligatorio para todo agente:

1. Leer este archivo (y `AGENTS.md`) al **inicio** de cada sesión de trabajo sobre `ir-spanish/`.
2. Contrastar tu contexto con el changelog y los archivos del repositorio antes de actuar.
3. Al terminar un cambio **importante**, **añadir una entrada al inicio** (orden cronológico inverso) con: archivos tocados, decisiones tomadas y estado/pendientes.

**Qué es «importante».** Una entrada se justifica cuando una sesión futura necesitaría leerla para no repetir trabajo, no deshacer una decisión o no malinterpretar el estado del proyecto: decisiones de método o de datos, cambios de estructura, scripts o cuadernos nuevos o reescritos, y resultados que cambian lo que se reporta. **No** hace falta anotar erratas, typos, reformateos ni el detalle de las verificaciones: el historial de git ya los guarda y el changelog debe seguir siendo corto para poder leerse entero.

## Plantilla de entrada

## YYYY-MM-DD — Título breve
- **Cambios:** archivos modificados y qué se hizo.
- **Decisiones:** qué se decidió y por qué.
- **Estado / pendientes:** qué quedó por hacer.

---

## 2026-10-27 — Selección completa sobre el subconjunto piloto
- **Cambios:** seis modelos en el piloto (BM25 con JDK 21 y SPLADE corregido), 57 subconjuntos, seis algoritmos sobre el trío, `results/train_pilot/` con `selection.md`, `stability.md`, `sweep_all/` y `algorithm_check_six/`, y `results/test_selection/` con las métricas de prueba.
- **Decisiones:** el subconjunto de 10 000 artículos es el **entorno de selección**; la partición de prueba se usa para evaluar y para el análisis descriptivo de estabilidad; no se usa la partición de entrenamiento completa. Regla: top-3 individual + CombMNZ; coinciden con los sistemas reportados.
- **Estado / pendientes:** estabilidad individual 1.0, fusiones 0.9726 y algoritmos 0.9429; top-1/top-3 coinciden; top-5 3/5 por combinaciones mixtas con SPLADE. La tesis debe reescribirse; la limitación residual es el corpus cerrado.

## 2026-10-26 — Piloto denso de selección sobre entrenamiento
- **Cambios:** `utils/data.py`, `utils/cache.py`, `utils/retrieval.py`, `rerankers/fuse_sweep.py`, `evaluation/train_selection.py`, `scripts/build_train_pilot.py`, `scripts/run_train_pilot.sh` y `scripts/run_train_pilot_analysis.sh`; `results/train_pilot/` y logs del piloto. El piloto usa 10 000 artículos y los cuatro modelos densos.
- **Decisiones:** la regla de selección es el top-3 individual y CombMNZ como referencia; BM25 y SPLADE quedan fuera del piloto por Java 11 y por un *run* de SPLADE que no se guardó. El piloto no se usa como evidencia.
- **Estado / pendientes:** el piloto coincide con el trío y CombMNZ; la fase definitiva (corpus completo, seis modelos) requiere JDK 21 para BM25, corregir la ruta de caché de `bm25.py` y revisar el guardado de SPLADE.

## 2026-10-26 — Se retira el pipeline que fusionaba y luego reordenaba
- **Cambios:** se eliminan `rerankers/hybrid.py` y su log `logs/hybrid.log`, el comando de `scripts/cmds.sh` y la sección «Hybrid» de `notes/results.md`; la nota de eficiencia de E3 en `tesis/` deja de excluirlo.
- **Decisión:** la variante que fusionaba con CombMNZ y reordenaba con `jina-reranker-v3` no está en los objetivos de investigación y nunca se reportó; el historial de git conserva el código y la corrida.

## 2026-10-23 — Limpieza de logs de evaluación
- **Cambios:** se eliminan los seis logs `e1_significance.log`, `e1_recall.log`, `e2_significance.log`, `e2_recall.log`, `e3_significance.log` y `e3_recall.log`, previos a la migración estadística; quedan los `*_statsmodels.log`.
- **Decisión:** los CSV y logs regenerados con `scipy`/`statsmodels` son la referencia; el historial de git conserva los antiguos.
- **Estado / pendientes:** sin pendientes.

## 2026-10-23 — Pruebas estadísticas sobre scipy/statsmodels
- **Cambios:** `evaluation/significance.py` pasa a envoltorios de `scipy.stats` y `statsmodels` (intervalo con `DescrStatsW`, Holm/BH con `multipletests`, permutación con `permutation_test`); `requirements.txt` declara `scipy` y `statsmodels`; el README lo refleja.
- **Decisión:** se elimina el código propio de intervalo, correcciones y permutación; la API pública se conserva para no tocar los experimentos.
- **Estado / pendientes:** CSV de E1–E3 regenerados; las columnas deterministas no cambian y solo se movió `p_permutation`.

## 2026-10-22 — Dos métricas primarias y resultados de Recall@100
- **Cambios:** `evaluation/significance.py` añade `paired_mean_ci`; `compare.py` reporta IC; `e2.py` y `e3.py` aceptan `--metric`/`--suffix`; nuevo `e1_recall.py`; `results/evaluation/` con `e1_recall.*`, `e2_*_recall.*` y `e3_recall.*`, y sus logs.
- **Resultados:** E1 Recall@100 CombMNZ +0.0192 [0.0183, 0.0202] y Condorcet −0.0120; E2 Recall@100 no distingue a los cuatro primeros tras Holm y solo RBC (0.0025) y Condorcet (0.0312) se separan; E3 la fusión +0.0192 [0.0183, 0.0202] sobre los tres reordenadores, estructural.
- **Pendientes:** ninguno de cómputo; el análisis por consulta del E5 y la actualización de las conclusiones quedan en la tesis.

## 2026-10-21 — Índices y almacenamiento
- **Cambios:** `notes/results.md` añade el cálculo de los tres índices planos y su viabilidad; el análisis completo está en `tesis/knowledge/indices-almacenamiento-fusion.md`.
- **Resultado:** 160.8 GiB en RAM o 80.4 GiB en GPU para tres índices; no caben en el servidor. Un índice del reranker sí (53.6/26.8 GiB).

## 2026-10-21 — E5 por algoritmo
- **Cambios:** `analysis/rbo_gain.py` acepta `--strategy` (uno o `all`); `results/e5/` con los cuatro pares de archivos y `summary_all.md`/`.json`.
- **Resultados:** la correlación RBO–ganancia en nDCG@10 es estable: 0.678 (CombMNZ), 0.661 (RBC), 0.749 (RRF) y 0.681 (ISR), con intervalos enteros por encima de 0.5. En Recall@100 es débil (0.25–0.31).
- **Decisión:** la conclusión del experimento 5 deja de depender de CombMNZ; se reportan las cuatro correlaciones como análisis descriptivo.

## 2026-10-21 — Etapa A completa con cuatro algoritmos
- **Cambios:** `results/fuse_sweep/results.csv` y `results.md` con 228 fusiones; `notes/results.md` añade la comparación entre algoritmos.
- **Resultados:** el mejor subconjunto es `e5-large+bge-m3+jina-v5-small` en los cuatro (0.5517 CombMNZ, 0.5485 RBC, 0.5450 RRF, 0.5382 ISR). La media por tamaño crece con el número de modelos, pero el máximo está en un trío.
- **Pendiente:** recalcular la correlación del E5 para los cuatro algoritmos.

## 2026-10-21 — Barrido de etapa A con RRF e ISR
- **Cambios:** `results/fuse_sweep/results.csv` crecerá de 114 a 228 pares; log en `logs/fuse_sweep_rrf_isr.log`.
- **Decisión:** se añaden RRF e ISR al barrido porque su costo es el mismo que el de los dos algoritmos ya ejecutados y así el mejor subconjunto no depende de CombMNZ.

## 2026-10-21 — Memoria de índices medida con trazas
- **Cambios:** `evaluation/mem_trace.py` (muestreo a 1 Hz con capas torch, dispositivo y proceso propio) y `evaluation/scaled_efficiency.py`; `results/scaled_efficiency/memtrace_{2000000,4000000}.csv` y `scale_*.json` con 2 M y 4 M, tres índices, tres repeticiones.
- **Resultados:** el incremento por índice y por GPU es 1 543 + 980·N_millones MiB; a 14 M son 44.7 GiB por GPU y 89.4 GiB en total. `torch.cuda.memory_reserved` marca 0: FAISS no es visible para PyTorch y la atribución se hace por PID con nvidia-smi. Sumando búsqueda y codificación, la fusión queda en ~191 ms por consulta frente a 482–2 319 ms de los rerankers: 2.5–12× más barata, no dos órdenes de magnitud.
- **Decisión:** se reporta la medición por escalamiento, no una corrida a 14 M con tres índices residentes, porque no caben.

## 2026-10-21 — Limpieza de artefactos y Condorcet definitivo
- **Cambios:** se eliminan `logs/e5_rbo_gain.log`, `logs/fuse-reranker-combnz-2-3.log`, `logs/fuse-reranker-combmnz-rbc-rrf-isr.log` y `logs/fuse-all-strategies.log`; también los crudos del piloto de eficiencia salvo `pilot/metadata/bge_rep0.json`, y `results/scaled_efficiency/candidates_*.json`.
- **Decisión:** el registro de Condorcet es el de la etapa B (0.5422/0.8918); se descarta el 0.5415/0.8913 del barrido antiguo, que queda en el historial de git. `logs/hybrid.log` se conserva, pero la variante no se reporta.

## 2026-10-21 — Jina completa la primera etapa
- **Cambios:** `evaluation/first_stage.py` mide también `jina-v5-small`; `results/first_stage/` incluye `jina.json`, `jina_batch1.csv`, el resumen y el log.
- **Resultados:** jina 29.57 ms de mediana en lote 1 y 1.12 ms en lote 32. La codificación total de la primera etapa es 57.53 ms en lote 1 y 2.41 ms en lote 32; los dos recuperadores extra son 27.96/1.28 ms.
- **Decisión:** con jina incluida, la ventaja total de la fusión baja a ~38×/~14×/~7× (lote 1) y ~41×/~35×/~17× (lote 32); sigue sin incluir búsqueda ni índices.

## 2026-10-21 — Codificación de la primera etapa
- **Cambios:** `evaluation/first_stage.py`, `results/first_stage/` y `logs/first_stage.log`; 500 consultas, e5-large y bge-m3, lote 1/32.
- **Resultados:** 14.65 y 13.31 ms de mediana en lote 1; 1.28 ms/consulta en lote 32 para los dos recuperadores extra. La ventaja de la fusión baja de 1 182× a ~75× en lote 1, pero sigue siendo de decenas de veces.
- **Pendiente:** búsqueda en los índices y almacenamiento; la cifra es una cota inferior.

## 2026-10-21 — Alcance de la eficiencia: segunda etapa
- **Cambios:** `notes/results.md` distingue el costo marginal de la segunda etapa del costo total del pipeline y advierte que la fusión usa tres recuperadores y los rerankers uno.
- **Decisión:** los factores 1 182× y 417× no se presentan como costo de extremo a extremo; la primera etapa queda fuera hasta medirla o declararla como limitación.

## 2026-10-21 — Banco de eficiencia ejecutado
- **Cambios:** `evaluation/efficiency.py` y `results/efficiency/` con 500 consultas, tres repeticiones, lote 1/32, crudos, metadatos y resumen; logs en `logs/efficiency_run.log` y `logs/efficiency_pilot.log`.
- **Resultados:** fusión 1.90 ms de mediana; `bge-reranker-v2-m3` 2 247.75 ms (1 182×), `jina-reranker-v3` 793.45 ms (417×) y `jina-colbert-v2` 410.76 ms (216×). VRAM pico: 4 401, 3 153 y 1 385 MiB.
- **Decisión:** la comparación principal es mediana/p95 y razones; el throughput del lote 1 se reporta como complemento porque la media de la fusión tiene valores atípicos.

## 2026-10-20 — Scripts durables en `scripts/`
- **Cambios:** `setup_env.sh` y `notes/cmds.sh` se mueven a `scripts/`; se elimina del control de versiones `run_b2_b3_after_b6.sh`; README y `scripts/README.md` documentan la regla.
- **Decisión:** solo se versionan scripts reproducibles; los lanzadores de una sola corrida son temporales y no se commitean.

## 2026-10-20 — Etapa B: B2 muere por OOM, B3 queda al límite
- **Cambios:** `notes/results.md` añade la tabla de BordaFuse por combinación y la sección de límites de recursos; se incorporan a git los logs de B2/B3, el muestreo de RSS y `results/fuse_stage_b/`.
- **Resultados:** B6 846.5 s (0.5363/0.9228); B3 1 259.8 s (0.4962/0.9237) con un pico de 100.2 GiB y 19.4 GB disponibles; B2 con BordaFuse murió por OOM a los 1 519 s (RSS 117.5 GiB, `oom_kill=1`, salida 137). Condorcet B1: 6 567 s y 0.8918 de Recall@100.
- **Decisiones:** B2-BordaFuse no es viable en 125 GiB y no se imputa; los cuatro algoritmos rápidos sobre B2 sí terminaron. Condorcet no se corre en combinaciones mayores: se descarta por tiempo y recall, sin afirmar un OOM que no se midió.

## 2026-10-20 — E4 y E5: líneas base y RBO–ganancia
- **Cambios:** `results/e4/baselines.md`; `analysis/rbo_gain.py`, `results/e5/rbo_gain.csv`, `results/e5/summary.json` y logs.
- **Resultados:** la mejor fusión supera a la línea base propietaria en nDCG@10 y Recall@100; el mejor reranker la supera en nDCG@10, no en recall. En E5, RBO correlaciona positivamente con la ganancia en nDCG@10 (Pearson 0.678, IC 95 % 0.521–0.808); con Recall@100 la correlación es débil (0.253, IC −0.025–0.553).
- **Pendientes:** completar BordaFuse en B2–B5, cerrar la etapa B y escribir el capítulo 5.

## 2026-10-20 — Sin ablación de longitudes fijas
- **Cambios:** `notes/results.md` pierde las tablas de q512/d512, q64/d256 y 400 palabras; se eliminan los seis `logs/rerankers-fuse-*.log` de la corrida q512; `rerankers/fuse_sweep.py` deja de citar esa corrida; `rerankers/fuse.py` y `analysis/top_n_per_query.ipynb` ya no mencionan al modelo descartado.
- **Decisión:** se reporta solo la configuración de longitudes máximas nativas; la ablación de longitudes fijas no se conserva.

## 2026-10-20 — Limpieza de logs y harrier fuera del estudio
- **Cambios:** se eliminan `logs/rbo_full_handrolled.log`, `logs/harrier-oss-v1-0.6b.log` y `logs/fuse-reranker-combnz-3-4.log`; `logs/qwen3-embedding-0.6b` pasa a `.log`; `notes/results.md` deja de reportar harrier.
- **Decisiones:** harrier no se considera en el estudio; el RBO reportado es el de `ranked-overlap`, no el artesanal.
- **Estado / pendientes:** `baselines/harrier.py` y la línea comentada de `rerankers/fuse.py` siguen ahí. La diferencia de Condorcet (0.5415 frente a 0.5422) se resolvió el 2026-10-21: se conserva 0.5422 de la corrida de etapa B.

## 2026-10-20 — Logs y resultados en control de versiones
- **Cambios:** `logs/fuse_sweep_nosave.log`, `logs/rbo_full_handrolled.log` y `logs/rbo_ranked_overlap.log` se versionan; `README.md` documenta la convención.
- **Decisión:** los logs y resultados de cada corrida se agregan al repositorio; se pueden limpiar después si crecen.
- **Pendientes:** etapa B, comparaciones de significancia y análisis RBO–ganancia.

## 2026-10-20 — RBO vuelve a ranked-overlap
- **Cambios:** `evaluation/rbo.py` y `requirements.txt` vuelven a `ranked-overlap`; `README.md` y `notes/results.md` lo reflejan.
- **Decisión:** se descarta PyTerrier para RBO por su costo (1000 profundidades y DataFrames, ~30 veces más lento); `ranked-overlap` es MIT, sin dependencias y produce la corrida rápida ya validada.
- **Estado / pendientes:** los 15 pares de `results/rbo/rbo_pairs.csv` y `logs/rbo_ranked_overlap.log` siguen siendo válidos; no hay que recalcular.

## 2026-10-20 — E2: comparación de los seis algoritmos
- **Cambios:** `results/evaluation/e2_holm.*` y `e2_bh.*`; `logs/e2_significance.log`.
- **Resultados:** los 15 pares sobre B1 son significativos; CombMNZ queda primero (0.5517), seguido de RBC (0.5485), RRF (0.5450), Condorcet (0.5422), BordaFuse (0.5414) e ISR (0.5382). La única diferencia que no baja de 1e-4 es BordaFuse–Condorcet (p = 0.0252).
- **Pendientes:** completar BordaFuse en B2–B5, E4 descriptivo y análisis RBO–ganancia del E5.

## 2026-10-20 — E1 y E3: significancia
- **Cambios:** `results/evaluation/e1.*` y `e3.*`; `logs/e1_significance.log`, `logs/e3_significance.log`; corrida de Condorcet B1 con run guardado.
- **Resultados:** E1, fusión 0.5517 frente a jina 0.5115 (Δ = +0.0402, p < 0.0001); E3, la fusión pierde con bge-reranker y jina-reranker y gana a jina-colbert.
- **Pendientes:** etapa B, E2, E4 descriptivo y análisis RBO–ganancia del E5.

## 2026-10-20 — README y terminología de pruebas
- **Cambios:** `README.md` reduce «Statistical Comparison» a la lista de pruebas y separa RBO en «Complementarity»; `evaluation/__init__.py` y `evaluation/significance.py` hablan de «paired permutation test».
- **Decisión:** no se mencionan pruebas de signos; la confirmación es una permutación pareada.
- **Pendientes:** corrida de RBO con PyTerrier y etapa B.

## 2026-10-20 — RBO con PyTerrier
- **Cambios:** `evaluation/rbo.py` usa `pyterrier_alpha.rbo`; `requirements.txt`; `README.md`; corrida completa en curso.
- **Decisión:** se usa PyTerrier, la implementación reconocida. RBO no está en el núcleo de PyTerrier 1.1.2, sino en su extensión oficial `pyterrier-alpha`; `ranked-overlap` se retira.
- **Estado / pendientes:** la corrida de 15 pares con PyTerrier está en curso; al terminar se actualizan `results/rbo/` y `notes/results.md`.

## 2026-10-20 — RBO con ranked-overlap
- **Cambios:** `evaluation/rbo.py` delega el cálculo a `ranked-overlap`; `requirements.txt`; `results/rbo/rbo_pairs.csv` recomputado; `notes/results.md` y `README.md` actualizados.
- **Decisiones:** se descarta PyTerrier para RBO porque su implementación trabaja sobre DataFrames y recorre 1000 profundidades por consulta; `ranked-overlap` es MIT, sin dependencias, e implementa la ecuación de Webber et al. (2010).
- **Estado / pendientes:** los 15 pares están recomputados; los pares densos coinciden y los de BM25 cambian en la cuarta decimal.

## 2026-10-20 — Resultados del barrido y de RBO
- **Cambios:** `results/fuse_sweep/results.csv`/`.md` (114 pares) y `results/rbo/rbo_pairs.csv` (15 pares) versionados; `notes/results.md` con las mejores fusiones y la matriz de RBO.
- **Decisiones:** la mejor fusión en nDCG@10 es CombMNZ sobre el trío denso (0.5517); los pares léxico–denso tienen RBO bajo (~0.15–0.18) frente a los densos (~0.38–0.48), lo que respalda la hipótesis de complementariedad del E5.
- **Estado / pendientes:** etapa B de fusión, comparaciones de significancia y análisis RBO–ganancia.

## 2026-10-19 — RBO para el E5
- **Cambios:** `evaluation/rbo.py` (RBO extrapolado, pares de runs, CLI y CSV); README con el uso.
- **Decisiones:** p = 0.95, profundidad 100 y promedio por consulta; la corrida completa de los 15 pares queda encolada tras el barrido de fusiones.
- **Estado / pendientes:** prueba de humo jina-v5 vs. bge-m3 con 1 000 consultas (RBO = 0.4075); falta la corrida completa y el análisis contra la ganancia.

## 2026-10-19 — Módulo `evaluation`: t pareada, permutación, Holm y BH
- **Cambios:** `evaluation/` nuevo (`significance.py`, `compare.py`); prueba de humo con jina-v5 vs bge-m3 en 20 000 consultas; corrección del bug de `ranx.evaluate` con una métrica parametrizada y de la importación perezosa del CLI.
- **Decisiones:** E1/E3 usan t pareada + permutación; E2 usa Holm-Bonferroni principal y Benjamini-Hochberg como sensibilidad; no Tukey. La implementación es independiente de `ranx.compare` para controlar las correcciones.
- **Estado / pendientes:** probado; falta aplicarlo a los runs definitivos y añadir el análisis de eficiencia.

## 2026-10-19 — Barrido exhaustivo de fusiones y normalización explícita
- **Cambios:** `rerankers/fuse_sweep.py` nuevo (todas las combinaciones de 2 a 6 modelos, CSV/Markdown y runs fusionados); `rerankers/fuse.py` pasa `norm="min-max"` explícito. Corrida lanzada en el servidor con CombMNZ y RBC 0.95 sobre las longitudes nativas.
- **Decisiones:** la corrida usa dos estrategias (CombMNZ y RBC) para no multiplicar por seis el costo; los runs fusionados quedan en `~/.cache/messirve_embeddings/fusions` para las pruebas pareadas. La normalización de CombMNZ es la min-max por consulta de `ranx`, verificada en su código.
- **Estado / pendientes:** el barrido está en curso (114 pares); al terminar, versionar `results/fuse_sweep/results.csv` y `.md` e implementar `ranx.compare`.

## 2026-09-25 — Extractos largos y sin recorte en las tablas del cuaderno top-n
- **Cambios:** `analysis/top_n_per_query.ipynb` — parámetro nuevo `EXTRACTO_CARACTERES` en la §1 (2000 por omisión; `None` = párrafo completo), que alimenta a `_extracto`, y `pd.set_option("display.max_colwidth", None)` en la celda de configuración. Cuaderno re-ejecutado en el servidor (0 errores).
- **Decisiones:** el límite de `_extracto` no era el único recorte: **pandas recorta las celdas a 50 caracteres en el `display`** (en el HTML y en el texto), así que subir el límite no bastaba. Medido sobre los párrafos que la recuperación devuelve para la consulta del usuario (7494312, 32 párrafos distintos): mediana 487 caracteres, P90 916, máx 1597; con el límite de 330 solo se leían completos 11 de 32. Con 2000 se leen completos todos y el tope queda como red de seguridad para párrafos anómalos (`None` lo quita). Verificado: 121 extractos en las tablas de la §6 y de la «Versión interactiva», **0 recortados** por pandas y 0 por el tope.
- **Estado / pendientes:** el cuaderno ejecutado pasa de 124 KB a 312 KB, porque el texto de cada párrafo va dos veces en cada tabla (en el HTML y en el texto plano de la salida). Es el costo de poder leer el documento dentro del cuaderno; si molesta en el historial de git, la vía sería mostrar las tablas con `to_markdown` (una sola representación) o bajar `EXTRACTO_CARACTERES`.

---

## 2026-09-25 — Corregido el estado global del cuaderno top-n (la versión interactiva mostraba la consulta vieja)
- **Cambios:** `analysis/top_n_per_query.ipynb` — las funciones de visualización dejan de leer la consulta y los textos del espacio global: ahora son `cargar_textos(resultados, numero)`, `mostrar_resultados(resultados, numero, textos)`, `mostrar_relevantes(numero, textos)` y `resumen_relevantes(resultados, numero)`. La sección «Versión interactiva» (celdas del usuario) queda con un único punto de entrada, `analizar_consulta(numero, n)`, que hace el recorrido completo; la celda del ejemplo (§4) usa `CONSULTA` para colgar de la misma resolución que la §3. Cuaderno re-ejecutado en el servidor (0 errores).
- **Decisiones:** el cuaderno se escribió para una consulta fija (§1), de modo que `CONSULTA` (resuelta en la §3) y `textos` (cargados en la §5) eran globales. Al cambiar `NUMERO_CONSULTA` en la versión interactiva, `top_n_por_modelo` devolvía los documentos de la consulta nueva, pero `mostrar_relevantes()` imprimía los relevantes de la vieja, `resumen_relevantes(resultados)` comparaba contra el `docid` viejo («—» en los seis modelos, `modelos = 0`) y las columnas `titulo`/`extracto` salían vacías, porque los textos se habían cargado solo para los `docid` de la consulta anterior. Se arregla pasando la consulta **explícitamente** (no leyendo variables globales) y con un solo punto de entrada interactivo, que evita olvidar un paso. Se conserva el límite de 330 caracteres de `_extracto` que había puesto el usuario.
- **Estado / pendientes:** verificado con la consulta interactiva del usuario, 7521003 («cual fue el ultimo presidente de la urss»): su relevante (`303630#0`, «Presidente de la Unión Soviética») sale 1.º en qwen3-0.6b y jina-v5-small, 9.º en bge-m3 y no aparece en los otros tres (`modelos = 3`), y las tablas salen con título y extracto; el ejemplo de la §6 sigue con 8101866. **Regla del cuaderno:** toda función que dependa de la consulta la recibe como argumento (`numero`); no leer `CONSULTA` ni `textos` del espacio global.

---

## 2026-09-25 — Cuaderno nuevo: top n de documentos por modelo para una consulta de prueba
- **Cambios:** `analysis/top_n_per_query.ipynb` — cuaderno nuevo (17 celdas) que, dado un número de consulta del split de prueba, devuelve el top n (10 por omisión) de documentos de cada modelo de recuperación inicial, leído de los runs cacheados. Ejecutado de extremo a extremo en el servidor con el kernel `proyecto` (0 errores) sobre la consulta de ejemplo 8101866 («que moneda circula en aruba»).
- **Decisiones:**
  - **Seis modelos**, los de la tabla principal de resultados: `bm25_pyserini`, `naver/splade-v3`, `intfloat/multilingual-e5-large-instruct`, `BAAI/bge-m3`, `Qwen/Qwen3-Embedding-0.6B` y `jinaai/jina-embeddings-v5-text-small-retrieval`. **`microsoft/harrier-oss-v1-0.6b` queda excluido** por indicación del usuario; en el cuaderno aparece como línea comentada, para que la exclusión sea explícita y no un olvido.
  - El «número de consulta» es el `id` numérico del split de prueba (p. ej. 8101866), que es la clave de los runs y de los qrels. `resolver_consulta()` falla con un mensaje claro si el número no existe, y el cuaderno trae `listar_consultas()` y `buscar_consultas()` para encontrar uno.
  - **Los runs se leen con `ranx.io.load_lz4`**, no con `Run.from_file`: `load_lz4` devuelve el mismo diccionario `{consulta: {docid: score}}` que escribió `Run.save` y tarda ≈4.7 s por run, mientras que `Run.from_file` construye las estructuras de numba de 170,055 consultas × 100 documentos y tardaba **más de 60 s**. Cada run se libera antes de cargar el siguiente (≈2.5 GB de pico, ≈30 s los seis).
  - Las rutas se derivan con `utils.cache`, la misma que usa el pipeline, a partir de `max_query_length`/`max_doc_length` de cada run. BM25 no tiene longitudes de secuencia, así que se aceptan la ruta sin longitudes (la que escribe `baselines/bm25.py`) y la que usan los scripts de fusión; los dos archivos son idénticos (`md5` verificado).
  - Los **textos** (título y extracto, más el texto completo de los relevantes) se sacan del corpus con una máscara de pyarrow (`pc.is_in`) sobre `corpus.data.table`, solo para los ≈40 `docid` que aparecen en las tablas (≈2 s); se desactivan con `INCLUIR_TEXTO = False`. Se silencian los registros INFO de `httpx`/`httpcore` que `utils` deja activos, porque llenaban la salida de peticiones HTTP.
  - La metadata usa la kernelspec **`proyecto`** (a diferencia de `distributions.ipynb`, que conserva `python3`): el cuaderno necesita `ranx`, `datasets` y `pandas` del entorno conda, y con `python3` JupyterLab arrancaría el python del sistema.
- **Estado / pendientes:** verificado: los seis runs devuelven 10 documentos para la consulta de ejemplo y el relevante (`328242#0`, «Florín arubeño») aparece 1.º en los cuatro densos, 5.º en SPLADE-v3 y 10.º en BM25 — buen ejemplo de la diferencia léxico/semántico para la tesis. **Nota:** BM25 devuelve a veces menos de 100 documentos (solo los que comparten términos con la consulta; p. ej. 58 para la consulta 8199550) y el cuaderno lo avisa cuando devuelve menos de *n*. Pendiente: si se re-ejecuta el pipeline con otras longitudes de secuencia, hay que actualizar `MODELOS` (§2) o el cuaderno falla indicando las rutas que probó.

---

## 2026-09-24 — Eliminada `docs/`: la figura del sistema se hace en TikZ dentro de la tesis
- **Cambios:** `docs/` — **eliminada** (los tres archivos: `arquitectura-recuperacion.mmd`, su PNG y el `README.md`). La figura del sistema vive ahora en la tesis, como código TikZ: `tesis/figuras/tikz/arquitectura-sistema.tex` (ver el changelog de `tesis/`, mismo día).
- **Decisiones:** el diagrama Mermaid se descartó por un problema **medido**, no de gusto: mide 1741 × 429 pt y la caja de texto de la tesis es de 443.6 pt, así que hay que reducirlo a ≈0.25 y su texto queda a ≈7.5 pt, ilegible. La versión TikZ se compone a 439.7 × 168 pt, **cabe sin reducir** y su texto sale a `\footnotesize`/`\scriptsize` con la tipografía del documento. El usuario pidió además no conservar el borrador de Mermaid en este repositorio. **No reintentar el diagrama en Mermaid para la tesis**; si se quiere un diagrama para documentación del repositorio (por ejemplo en el `README.md`), sería otro caso y con otro fin.
- **Estado / pendientes:** los detalles de la decisión, las medidas y los tropiezos (los *shorthands* de `babel`, el `text width` de los nodos) están en el changelog de `tesis/` y en `AGENTS.md`, que es donde vive ahora la convención de diagramas.

---

## 2026-09-24 — El servidor deja de ser obligatorio para todo `ir-spanish`
- **Cambios:** `AGENTS.md` (en el paraguas `titulación/`, no en este repositorio) — la sección «Fuente de verdad por repositorio» se sustituye por **«Git → Dónde trabajar `ir-spanish` (servidor o copia local)»**: el servidor solo hace falta cuando el trabajo **necesita ejecutar** algo (cuadernos de `analysis/`, scripts de `baselines/`/`rerankers/`, embeddings, cualquier cosa que dependa del entorno conda `proyecto`, las GPU o los datos de MessIRve); para documentación, diagramas, notas y changelog **se trabaja en la copia local del Mac**. En ambos casos, `git pull --ff-only` antes de empezar y `push` al terminar, para que las dos copias queden al día. También se actualizaron la estructura del directorio raíz y la viñeta de «Dónde se edita el código» de la sección de Jupyter, que apuntaba a la sección renombrada.
- **Decisiones:** la regla anterior («una sola máquina escritora»: todo `ir-spanish` en el servidor, el clon del Mac como espejo de solo lectura) **queda derogada** por indicación del usuario (2026-09-24). El motivo original era evitar que los dos clones acumularan el mismo cuaderno reescrito sin commitear; se conserva la parte que lo prevenía —una sola máquina escribiendo a la vez y sincronización obligatoria— sin exigir SSH para trabajo que no ejecuta nada. El diagrama de `docs/` es justamente ese tipo de trabajo: se hizo en el Mac.
- **Estado / pendientes:** la copia del Mac y la del servidor estaban ambas en `d40255a` y limpias salvo por este cambio, así que no hubo conflicto. A partir de ahora, **antes de tocar `ir-spanish` en cualquiera de las dos máquinas, `git pull --ff-only`**, y `push` al terminar.

---

## 2026-09-24 — Diagrama del sistema de recuperación en `docs/`
- **Cambios:** `docs/` — carpeta nueva con `arquitectura-recuperacion.mmd` (el diagrama) y `README.md` (índice, decisiones del diagrama e instrucciones de renderizado).
- **Decisiones:** es un **`flowchart LR`** con las tres claves de apariencia fijadas en el *front matter* (`theme: default`, `look: classic`, `layout: dagre`) para que se vea igual en cualquier versión de Mermaid, porque a partir de la v12 el aspecto por defecto pasó a `redux-color` con motor ELK. Los almacenes de datos (documentos, consultas, rankings en caché y verdad de referencia) van en forma de cilindro `[( )]`; los procesos, en rectángulos. **La fusión y el reordenamiento son ramas hermanas, no pasos consecutivos**: ambos consumen los rankings cacheados y ambos entregan a la evaluación, que es lo que se está comparando. Por acuerdo con el usuario el diagrama **no** muestra los modelos concretos, **no** incluye la variante híbrida (`rerankers/hybrid.py`, sin justificar en los objetivos de investigación) y **no** lleva parámetros.
- **`subgraph` por sección (a petición del usuario):** cinco recuadros —Datos, Embeddings, Recuperación, Alternativas y Evaluación— con los identificadores `S1`…`S5` para que los títulos se lean limpios. Los nodos de cada recuadro se renombraron por claridad: dentro de «Embeddings» las etiquetas dicen «Embeddings de documentos/consultas» (antes solo «Documentos»/«Consultas», que se leían como datos y no como la etapa de codificación). **La verdad de referencia queda fuera de los recuadros a propósito:** al meterla en el `subgraph` «Evaluación», el recuadro se extiende a la izquierda y las dos flechas de las alternativas entran cruzando el título, que queda ilegible; se comprobó renderizando las dos variantes. Tampoco funcionó envolverla en un `subgraph` anidado.
- **Descartado antes de llegar aquí (no reintentar sin motivo):** una primera versión `flowchart` listaba los seis modelos de primera etapa, los seis algoritmos de fusión y los tres rerankers: el usuario la rechazó por sobrecargada. Una segunda, con **diagrama de arquitectura** (`architecture-beta`, íconos nativos `database`/`server`/`disk`), también fue rechazada: además de que su layout fcose **superpone** «Documentos» y «Consultas» en la misma coordenada si no se añade `align row` (requiere Mermaid ≥ 11.16), el usuario prefiere la simplicidad del `flowchart`.
- **Estado / pendientes:** validado con `mermaid.parse` y renderizado con `mermaid-cli` 12.0.0; **el PNG renderizado se versiona** en `docs/arquitectura-recuperacion.png` (3200 × 926 px, generado con `-b white -s 2 --size 1600` desde la raíz del repositorio; el comando está en `docs/README.md`) y **se muestra en el chat** al proponer un cambio, por indicación del usuario (2026-09-24): a partir de ahora todo diagrama se revisa sobre la imagen renderizada, no solo sobre el código o el parseo. El diagrama corresponde a la figura del pipeline de la §4.2 de la tesis, hoy marcada como `[PENDIENTE: Fig. 4.1]` en `tesis/capitulos/capitulo4.tex`; conviene decir en el pie de figura que la fusión y el reordenamiento son alternativas que se comparan. Pendiente: el visto bueno del usuario y, después, la entrada en el changelog de `tesis/` si el diagrama se incorpora como figura.

---

## 2026-09-21 — Bug corregido: las «consultas de prueba» eran en realidad las de entrenamiento
- **Cambios:** `analysis/distributions.ipynb` — la celda 7 ya no reutiliza el nombre `unique_queries`: define `test_queries` y `train_queries` por separado, y la celda 15 (longitudes de consulta de prueba) lee `test_queries`. La figura `analysis/figures/messirve-longitud-consultas-test.pdf` se regeneró ejecutando **solo las celdas necesarias** (1, 2, 7 y 15) en una copia parcial y transplantando después las salidas de la celda 15 al cuaderno real, sin repetir el `.map()` sobre los 14 M de documentos del corpus.
- **Decisiones:** el arreglo va en la **causa raíz** —la reutilización del nombre en la celda 7, introducida en `3e2a317`— y no en la celda 15, que era el síntoma. Se ejecutaron solo las celdas necesarias, a petición del usuario, en lugar de re-ejecutar el cuaderno completo (~5 min). El bug pasó inadvertido porque la figura **no fallaba**: producía una imagen plausible, y solo el `print` de la propia celda delataba el número de elementos (708 373, que es train).
- **Estado / pendientes:** verificado: la celda 15 ahora reporta 170 055 elementos, mín 3, mediana 5.0, P90 8, P95 9, P99 12, máx 22 y desv 1.88, **idénticos a los que se habían calculado aparte** con el mismo criterio y validados contra los controles del cuaderno (train consultas 6.02/1.91 y prueba documentos 79.35/48.49). `messirve-longitud-consultas-test.pdf` ya no es un duplicado de la de entrenamiento: muestra «mediana = 5» con el eje hasta 40 000, frente a «mediana = 6» y 160 000 de train. **Nota cosmética:** las celdas 1, 2, 7 y 15 quedaron con `execution_count` 1-4 (de la ejecución parcial) mientras el resto conserva los de la última ejecución completa.

---

## 2026-09-21 — Las figuras pierden el título interno y el texto pasa a `print()`
- **Cambios:** `analysis/distributions.ipynb` — en las 7 celdas de gráficas se elimina `ax.set_title(...)` y, justo antes de cada figura, se añaden dos `print()`: el primero con el texto que tenía el título, con el prefijo «Figura: », y el segundo con el archivo que genera la celda (`        -> figures/<archivo>.pdf`). La explicación va **una sola vez**, en la celda 1, y los comentarios que ya tenían las celdas se conservan (en las 17 y 20 anotaban la escala logarítmica). Las 7 figuras de `analysis/figures/` se regeneraron con una ejecución completa del cuaderno en el servidor.
- **Decisiones:** el título dentro de la imagen **duplicaba el pie de figura** de LaTeX al incluirla en la tesis, así que se quita de la imagen pero **no se pierde**: queda en el `print()` como referencia para redactar. Los `print()` **no numeran** las figuras a propósito, porque la numeración la asigna LaTeX («Figura 4.x») y una numeración interna entraría en conflicto. La segunda línea (el archivo) se añadió para poder localizar la figura al citarla.
- **Estado / pendientes:** verificado: 0 `set_title` en el cuaderno; `pdftotext` no encuentra el texto del título en ninguna de las 7 figuras; las 7 siguen con `CLATDO+LMRoman10-Regular` en `CID TrueType`; 0 errores y 0 avisos de glifos en el cuaderno ejecutado. Las copias de `tesis/figuras/` se actualizaron en la misma sesión (ver el changelog de `tesis/`).

---

## 2026-09-21 — Las figuras pasan a componerse en Latin Modern (la tipografía de la tesis)
- **Cambios:** `analysis/fonts/` — carpeta nueva con `lmroman10-regular.ttf` (Latin Modern Roman en TrueType), `GUST-FONT-LICENSE.txt` y `README.md` con la procedencia y la receta de conversión. `analysis/distributions.ipynb` — la celda 1 registra la fuente (`font_manager.addfont`) y fija los `rcParams` (`font.family = "Latin Modern Roman"`, `mathtext.fontset = "cm"`, `pdf.fonttype = 42`, `font.size = 10`, `axes.titlesize = 11`) y define `FIG_ANCHO, FIG_ALTO = 6.14, 4.39`; las 7 celdas de gráficas ya no usan `figsize=(7, 5)` sino esas constantes. Las 7 figuras de `analysis/figures/` se regeneraron con una ejecución completa del cuaderno en el servidor.
- **Decisiones:** las figuras iban en **DejaVu Sans** (la fuente por defecto de matplotlib), que no se parece al serif del cuerpo de la tesis; ahora usan **Latin Modern**, la misma de `lmodern`. Descartado `text.usetex` (el servidor **no tiene LaTeX**) y descartado también el `cmr10` que trae matplotlib, al que le faltan los glifos acentuados (á, ó, ú). El OTF de Latin Modern sí funciona tal cual en matplotlib, pero con él el PDF incrusta la fuente como **Type 3** y sin mapa Unicode, así que el texto de la figura no se puede buscar; convertida a **TTF** con `fontTools` y con `pdf.fonttype = 42` queda como **CID TrueType** con mapa Unicode. El ancho de las figuras se ata a la **caja de texto de la tesis** (15.59 cm = 6.14 in) para que `width=\textwidth` no las reescale; el alto conserva la proporción 7:5 anterior. La fuente se registra **desde el repositorio**, de modo que el cuaderno no depende de que esté instalada en el sistema ni de que exista LaTeX, y si el archivo falta falla con un error explícito en vez de caer en silencio a DejaVu.
- **Estado / pendientes:** verificado en las 7 figuras regeneradas: `CLATDO+LMRoman10-Regular`, `CID TrueType`, `Identity-H`, `uni=yes`, ancho 435.3 pt (la caja mide 442.0 pt, es decir un reescalado del 1.5 % al incluir), acentos correctos al extraer el texto y 0 avisos de glifos en el cuaderno ejecutado. La ejecución completa tardó ~5 min. Las copias de `tesis/figuras/` se actualizaron en la misma sesión (ver el changelog de `tesis/`). **Observación sin atender, fuera de lo pedido:** el formato de miles no coincide entre las marcas del eje (`175000`) y las anotaciones (`166,201`).

---

## 2026-09-14 — Figuras de MessIRve exportadas a PDF; `ir-spanish` pasa a trabajarse en el servidor
- **Cambios:** `analysis/distributions.ipynb` — el cuaderno escribe las 7 figuras en `analysis/figures/*.pdf` (`FIG = Path("figures")` en la celda 1; `savefig` en las celdas 10, 12, 14, 16, 20, 22 y 24: distribución de relevantes por consulta, longitud de consultas, de documentos relevantes y del corpus completo, en train y test). Esas figuras y la reescritura del cuaderno —que hasta ahora solo existían sin commitear, idénticas, en los dos clones— quedaron versionadas en `4b0e291`, junto con las dos líneas del túnel de Jupyter en `notes/cmds.sh`, que solo estaban en el clon del Mac.
- **Decisiones:** el export de figuras es **reproducible** (volver a ejecutar el cuaderno regenera los 7 PDF; no es un paso manual) y aun así se **versionan** en `analysis/figures/`, para fijar el artefacto exacto que consume la tesis (`tesis/figuras/` es la copia del repo de la tesis). **`ir-spanish` se trabaja en el clon del servidor** (`~/ir-spanish`), que es la fuente de verdad: los agentes entran por `ssh iimas` a editar, ejecutar, commitear y hacer `push`, y el clon del Mac queda como espejo de lectura (`git pull --ff-only`, sin editar ni commitear). Queda documentado en `AGENTS.md`, «Git → Fuente de verdad por repositorio».
- **Estado / pendientes:** `origin/main` = `4b0e291`, y el clon del servidor quedó limpio y sincronizado. Los números siguen siendo los de la entrada anterior (170 055 consultas; 69 001 documentos relevantes; 97.73 % con un único relevante). El `push` exigió un `rebase` porque el servidor estaba en `d2b1c87` y `ec82173` (solo changelog) ya estaba publicado: de ahí la regla operativa de hacer `git pull --ff-only` **antes** de commitear en el servidor.

---

## 2026-09-14 — Cuaderno de distribuciones de MessIRve reescrito y ejecutado
- **Cambios:** `analysis/distributions.ipynb` — reescrito como 7 secciones numeradas (consultas únicas; documentos relevantes únicos; artículos relevantes únicos; distribución de relevantes por consulta; longitud en palabras de consultas, de documentos relevantes y del corpus completo), cada una con su texto impreso y su gráfica en español; eliminadas las celdas exploratorias y el bloque suelto de `pyarrow`. Ejecutado en el servidor (helena, env miniconda `proyecto`) con `jupyter nbconvert --execute`; 17/17 comprobaciones de salida verificadas.
- **Decisiones:** el conteo de palabras de un documento es **título + texto**, el mismo criterio del pipeline de recuperación. En el servidor se creó la kernelspec `proyecto` (ruta absoluta al python de miniconda) porque el kernel «python3» resuelve al python del sistema —sin `datasets`— cuando nbconvert se lanza desde una terminal (desde JupyterLab sí funciona). El cuaderno conserva la metadata de kernel «python3» por portabilidad.
- **Estado / pendientes:** los números quedan medidos y disponibles para la tesis: 170,055 consultas; **69,001** documentos relevantes únicos y **51,190** artículos; el **97.73 %** de las consultas tiene **1** documento relevante (media 1.02, máx 4); longitudes medias en palabras: consultas 5.7, documentos relevantes 79.4, corpus 59.4 (14,047,759 párrafos; 1,946,205 artículos). **Implicación para la tesis:** resuelve el pendiente de §4.1 P3 y condiciona §4.4 — con ~1 relevante por consulta, Recall@100 se comporta como recall del único relevante, P@50 queda acotada por ~0.02 y MAP se aproxima a MRR.
