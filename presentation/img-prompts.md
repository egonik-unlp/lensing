# Imágenes del deck — fuentes y lenguaje visual

El deck es **vector-first**: casi todas las ilustraciones están construidas en
SVG/CSS (anillo de Einstein animado de la marca, diagramas de flujo, terminal,
heatmap del scan, mapa de comps, mock del PDF, galaxia del showcase, rejilla de
espacio-tiempo deformada). Las **únicas imágenes rasterizadas** son **4 fotos
reales de telescopio** que ya vienen commiteadas en `assets/`, una por cada
diapositiva a sangre completa.

No hay nada que generar: las fotos ya están. Este archivo documenta de dónde
salieron (atribución) y las reglas del lenguaje visual por si hay que
reemplazarlas.

## Las 4 fotos de fondo (reales, ya incluidas)

Cada una ilustra una manifestación distinta de la **lente gravitacional**, el
hilo conductor del deck: anillo → tránsito que curva la luz de una galaxia →
campo profundo → agujero negro.

| Archivo | Slide | Qué es | Fuente |
|---|---|---|---|
| `assets/hero-cosmos.jpg` | 1 (Portada) | Anillo de Einstein real | NASA / Hubble (dominio público) |
| `assets/blackhole-milkyway.jpg` | 9 (divisor "Primera parte · Modelar con lensing") | Simulación de un agujero negro pasando frente a la Vía Láctea — su gravedad curva la luz de la galaxia de fondo en un anillo | Ute Kraus (fondo de Axel Mellinger), CC BY-SA |
| `assets/deepfield-smacs.jpg` | 16 (divisor "Bajo el capó") | Primer campo profundo de JWST, SMACS 0723 (2022) — galaxias de fondo deformadas en arcos por la masa del cúmulo | NASA / ESA / CSA / STScI |
| `assets/blackhole-m87.jpg` | 20 (Demo) | El agujero negro supermasivo M87\* (2019) | Event Horizon Telescope (CC BY 4.0, ESO) |

Las cuatro se montan como fondo a sangre completa (`.hero-bg .img`), atenuadas y
paneadas para dejar lugar al texto y a la animación vectorial al lado. En los
divisores 1 y 9 la foto se aparea con un esquema vectorial del mismo fenómeno
(el anillo de Einstein animado de la marca y el tránsito `bhlens`,
respectivamente). El deck trae *fallbacks* en CSS (gradientes radiales) por si
falta algún archivo.

## Reglas del lenguaje visual (si reemplazás una foto)

- **Sin texto, sin logos, sin UI.** Solo atmósfera.
- Estética **científica/precisa**, no pictórica.
- Formato **JPG/WebP/PNG**, **16:9**, ≥ 2400×1350, fondo oscuro.
- Dejá el lado donde va el texto/la animación más oscuro/vacío; la foto va
  atenuada (opacity ~0.5–0.9 según el caso) para que no compita con el contenido.
- El feature brillante (anillo, cúmulo, agujero negro) debe quedar hacia un lado,
  no centrado bajo el texto.

## Lo que NO hay que generar (ya está, en vector)

Anillo de Einstein animado (la marca) · logo · íconos · diagramas de flujo y
pipeline · terminal del predictor · heatmap del scan · mock del PDF (registro) ·
mapa de comps (heatmap inferno sobre mapa oscuro) · galaxia del showcase
(Pathfinder) · rejilla de espacio-tiempo deformada · semillas de marca.
