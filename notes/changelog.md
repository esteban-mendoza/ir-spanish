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

## 2026-09-14 — Cuaderno de distribuciones de MessIRve reescrito y ejecutado
- **Cambios:** `analysis/distributions.ipynb` — reescrito como 7 secciones numeradas (consultas únicas; documentos relevantes únicos; artículos relevantes únicos; distribución de relevantes por consulta; longitud en palabras de consultas, de documentos relevantes y del corpus completo), cada una con su texto impreso y su gráfica en español; eliminadas las celdas exploratorias y el bloque suelto de `pyarrow`. Ejecutado en el servidor (helena, env miniconda `proyecto`) con `jupyter nbconvert --execute`; 17/17 comprobaciones de salida verificadas.
- **Decisiones:** el conteo de palabras de un documento es **título + texto**, el mismo criterio del pipeline de recuperación. En el servidor se creó la kernelspec `proyecto` (ruta absoluta al python de miniconda) porque el kernel «python3» resuelve al python del sistema —sin `datasets`— cuando nbconvert se lanza desde una terminal (desde JupyterLab sí funciona). El cuaderno conserva la metadata de kernel «python3» por portabilidad.
- **Estado / pendientes:** los números quedan medidos y disponibles para la tesis: 170,055 consultas; **69,001** documentos relevantes únicos y **51,190** artículos; el **97.73 %** de las consultas tiene **1** documento relevante (media 1.02, máx 4); longitudes medias en palabras: consultas 5.7, documentos relevantes 79.4, corpus 59.4 (14,047,759 párrafos; 1,946,205 artículos). **Implicación para la tesis:** resuelve el pendiente de §4.1 P3 y condiciona §4.4 — con ~1 relevante por consulta, Recall@100 se comporta como recall del único relevante, P@50 queda acotada por ~0.02 y MAP se aproxima a MRR.
