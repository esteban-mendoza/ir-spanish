# Tipografía de las figuras

`lmroman10-regular.ttf` es la fuente con la que se componen las figuras de
`analysis/distributions.ipynb`: **Latin Modern Roman**, la misma que usa el cuerpo de la
tesis (`\usepackage{lmodern}` en `tesis/tesis.tex`).

## Por qué la fuente vive aquí

El cuaderno la registra con `matplotlib.font_manager.addfont()`, así que las figuras no
dependen de que la fuente esté instalada en el sistema ni de que exista una instalación
de LaTeX (el servidor de experimentos no la tiene).

## Procedencia

1. Paquete `lm` de CTAN: <https://mirrors.ctan.org/fonts/lm.zip>
2. Archivo de origen: `lm/fonts/opentype/public/lm/lmroman10-regular.otf`, con contornos
   CFF (curvas cúbicas).
3. Conversión a TrueType (curvas cuadráticas) con `fontTools`, **sin tocar los contornos**:

```python
from fontTools.ttLib import TTFont, newTable
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen

src, dst = "lmroman10-regular.otf", "lmroman10-regular.ttf"
f = TTFont(src)
orden, glifos = f.getGlyphOrder(), f.getGlyphSet()
glyf = newTable("glyf"); glyf.glyphOrder = orden; glyf.glyphs = {}
for n in orden:
    pluma = TTGlyphPen(glifos)
    glifos[n].draw(Cu2QuPen(pluma, 1.0, reverse_direction=True))
    glyf.glyphs[n] = pluma.glyph()
f["loca"] = newTable("loca"); f["glyf"] = glyf
for t in ("CFF ", "VORG"):          # tablas propias de CFF
    if t in f: del f[t]
f.sfntVersion = "\x00\x01\x00\x00"
m = f["maxp"]; m.tableVersion = 0x00010000
for k, v in dict(maxZones=1, maxTwilightPoints=0, maxStorage=0, maxFunctionDefs=0,
                 maxInstructionDefs=0, maxStackElements=0, maxSizeOfInstructions=0,
                 maxComponentElements=0, maxComponentDepth=0).items():
    setattr(m, k, v)                # sin esto, guardar falla con KeyError: 'maxZones'
glyf.compile(f)
p = f["post"]; p.formatType = 3.0; p.glyphOrder = None
p.extraNames = []; p.mapping = {}; p.names = []
f.save(dst)
```

## Por qué convertida a TTF y no usar el OTF

matplotlib no puede incrustar una fuente CFF: guardando el PDF con el `.otf`, la fuente
sale como **Type 3** y sin mapa Unicode, así que el texto de la figura no se puede buscar
ni copiar. Con el `.ttf` y `pdf.fonttype = 42` sale como **CID TrueType** con mapa
Unicode. Comprobado con `pdffonts` sobre las figuras de `analysis/figures/`.

## Licencia

Latin Modern se distribuye bajo la **GUST Font License**: la LaTeX Project Public License
1.3c o posterior, más una cláusula 1 que *pide* —pero **no obliga legalmente**— renombrar
las obras derivadas. Aquí los contornos están intactos y solo cambia el formato. El texto
completo está en `GUST-FONT-LICENSE.txt`, copiado del mismo paquete de CTAN.
