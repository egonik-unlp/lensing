# Presentación — ¿Qué es lensing?

Deck interno (en español) que explica qué es lensing: un laboratorio de
predicción que se adapta a tus datos. Autocontenido, sin dependencias, sin
build. Reúne el lenguaje visual real del proyecto ("The Control Room"): chrome
de pizarra, papel blanco, un solo naranja para lo vivo, monoespaciada tabular
para cada número.

## Presentar

Abrí `index.html` en cualquier navegador (Chrome/Edge/Firefox). Para pantalla
completa: tecla **F**.

| Tecla | Acción |
|---|---|
| `→` `Espacio` | siguiente paso / diapositiva |
| `←` | anterior |
| `↑` `↓` | diapositiva anterior / siguiente (salta pasos) |
| `Inicio` `Fin` | primera / última |
| `1`…`9` | ir a la diapositiva N |
| `O` | panorama (todas las diapositivas) |
| `S` | notas del orador |
| `F` | pantalla completa |
| `?` | ayuda · `Esc` cierra |

También: click en la mitad derecha avanza, en la izquierda retrocede; swipe en
táctil. El número de diapositiva queda en la URL (`#5`), así podés retomar donde
quedaste.

## Imágenes de fondo

El deck usa **4 fotos reales de telescopio** (dominio público / CC), ya incluidas
en `assets/`, una por cada diapositiva a sangre completa. Cada una ilustra una
manifestación distinta de la lente gravitacional:

- `assets/hero-cosmos.jpg` — portada (slide 1): anillo de Einstein real (Hubble).
- `assets/blackhole-milkyway.jpg` — divisor "Primera parte · Modelar con lensing"
  (slide 9): un agujero negro pasando frente a la Vía Láctea, su gravedad curva la
  luz en un anillo (Ute Kraus, CC BY-SA).
- `assets/deepfield-smacs.jpg` — divisor "Bajo el capó" (slide 16): primer campo
  profundo de JWST (SMACS 0723) con galaxias deformadas en arcos.
- `assets/blackhole-m87.jpg` — Demo (slide 20): el agujero negro M87* (EHT).

Hay *fallbacks* en CSS (gradientes) por si falta algún archivo, pero las imágenes
ya vienen commiteadas; no hace falta generar ni tocar nada. Ver `img-prompts.md`
para fuentes/atribución y el resto del lenguaje visual (todo vectorial).

## PDF de respaldo

`Ctrl/Cmd + P` con **"gráficos de fondo"** activado → guardar como PDF
(una diapositiva por página, apaisado). Para que el PDF muestre todo el
contenido revelado, abrí con `index.html?all` antes de imprimir, o usá
`index.html?print`.

## Estructura

```
presentation/
  index.html      las 20 diapositivas + notas del orador (inline)
  deck.css        sistema de diseño (tokens Control Room) + impresión
  deck.js         navegación, pasos, panorama, notas, lattices animados
  fonts/          Sora · Inter · JetBrains Mono (self-hosted, woff2)
  assets/         SVGs de marca reutilizados + 3 fotos de telescopio de fondo
  img-prompts.md  fuentes de las fotos de fondo + lenguaje visual
```

## Contenido (20 diapositivas, en 2 partes)

**Parte 1 — Estructura (el framework genérico, slides 1–15):** portada · modelar
es un pipeline · ¿cómo lo hacemos hoy? · un programa genérico que la IA adapta ·
un template que produce instancias · de Bootstrap a showcase (el recorrido) · la
interfaz es doble (GUI + harness) · `/bootstrap` · **divisor "Primera parte ·
Modelar con lensing"** · preprocesamiento (dos transformaciones, una fila) · el
contrato de los predictores · entrenar y comparar · el registro científico en PDF
(con heatmap de un scan) · qué es un modelo · showcase.

**Parte 2 — Bajo el capó (la ingeniería + demo, slides 16–20):** divisor "Bajo el
capó" · ¿por qué Rust? (el núcleo) · cómo está hecho y dónde corre (crates +
deployment) · correr el modelo en el cliente (tecnología del showcase) · **Demo**.

Tres secciones en el footer: **Estructura · Ingeniería · Demo**. El deck se abre y
se quiebra con diapositivas a sangre completa que muestran una lente gravitacional
real (anillo de Einstein → tránsito que curva una galaxia → campo profundo →
agujero negro); el divisor "Modelar con lensing" (slide 9) separa la introducción
del recorrido detallado del modelado.

Texto mínimo en pantalla a propósito: el detalle está en las **notas del orador**
(tecla `S`). Todos los números son reales, tomados de las instancias del workspace.
El Pathfinder es un **showcase** de spotify-engagement, no otra instancia.
