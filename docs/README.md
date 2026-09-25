# Diagramas de `ir-spanish`

Diagramas [Mermaid](https://mermaid.js.org/) que documentan la arquitectura del sistema
de recuperación implementado en este repositorio.

| Archivo | Qué muestra |
| --- | --- |
| `arquitectura-recuperacion.mmd` | El sistema de recuperación de principio a fin, **con cada sección enmarcada en un `subgraph`**: datos, embeddings, recuperación (con la caché de rankings), las dos alternativas (fusión de rangos / reordenamiento neuronal) y la evaluación contra la verdad de referencia. |
| `arquitectura-recuperacion.png` | El mismo diagrama renderizado con `mermaid-cli`, para verlo sin herramienta adicional. **Es un artefacto generado: no se edita a mano.** |

## Cómo renderizarlo

GitHub, GitLab, mermaid.live, la extensión de VS Code u Obsidian renderizan el contenido
del `.mmd` tal cual.

Para regenerar el PNG versionado (o exportar a SVG o PDF), con
[mermaid-cli](https://github.com/mermaid-js/mermaid-cli), **desde la raíz del repositorio**:

```bash
npx -p @mermaid-js/mermaid-cli mmdc -i docs/arquitectura-recuperacion.mmd \
  -o docs/arquitectura-recuperacion.png -b white -s 2 --size 1600
```

Cambiar `-o` a `docs/arquitectura-recuperacion.svg` (o `.pdf`) produce el otro formato.
Para que el PNG se vea igual que aquí, **fija la apariencia en el *front matter*** del
`.mmd` (`theme: default`, `look: classic`, `layout: dagre`): a partir de Mermaid v12 el
aspecto por defecto cambió a `redux-color` con motor ELK.

## Secciones y formas

| Sección (`subgraph`) | Contenido | Formas |
| --- | --- | --- |
| **Datos** | documentos, consultas | cilindros (almacenes) |
| **Embeddings** | embeddings de documentos, embeddings de consultas | rectángulos (procesos) |
| **Recuperación** | recuperación, rankings en caché | rectángulo y cilindro |
| **Alternativas** | fusión de rangos, reordenamiento neuronal | rectángulos |
| **Evaluación** | evaluación | rectángulo |

La **verdad de referencia** queda deliberadamente **fuera** de los recuadros, como recurso
externo que alimenta a la evaluación: es el único nodo que no forma parte de una etapa del
sistema. *No la metas en el `subgraph` «Evaluación»*: si lo haces, el recuadro se extiende a
la izquierda, las dos flechas de las alternativas entran cruzando el título y este queda
ilegible (verificado renderizando ambas variantes).

## Decisiones del diagrama

- **Es un `flowchart LR`** (flujo de izquierda a derecha), con las tres apariencias fijadas
  en el *front matter* (`theme: default`, `look: classic`, `layout: dagre`) para que se vea
  igual en cualquier versión de Mermaid: a partir de la v12 el aspecto por defecto cambió a
  `redux-color` con motor ELK.
- **Forma de cilindro `[( )]`** para todo lo que es un almacén de datos: documentos,
  consultas, rankings en caché y verdad de referencia. Los procesos van en rectángulos.
- **La fusión y el reordenamiento son hermanos, no pasos consecutivos.** Están en la misma
  sección «Alternativas» porque comparten la entrada (los rankings cacheados) y comparten la
  salida (la evaluación): el diagrama no dice «fusión y luego reordenamiento».
- **Detalle deliberadamente ausente:** los modelos concretos de cada etapa, los algoritmos de
  fusión y los parámetros. Ese detalle va en el texto y las tablas de la tesis, no en la figura.

## Estado

- **Borrador en revisión.** El diagrama se propone en el chat antes de darlo por definitivo.
- **Numeración de figura.** Cuando el diagrama se incorpore a la tesis, corresponde a la
  figura del pipeline de la §4.2 (`tesis/capitulos/capitulo4.tex`, plan P1), que hoy está
  marcada como `[PENDIENTE: Fig. 4.1]`. El `.mmd` no numera la figura: el número lo asigna
  LaTeX. **Conviene decir en el pie de figura** que la fusión y el reordenamiento son
  alternativas que se comparan, no etapas sucesivas.
