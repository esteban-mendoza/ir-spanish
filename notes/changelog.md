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
