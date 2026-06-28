# Detección de duplicados — Resumen del plan

> Para quien conoce el scraper, los embeddings y la base vectorial, pero **nunca
> escuchó hablar de *lensing*** y no necesita el detalle interno ni la teoría de
> los modelos. El objetivo es que se entienda **qué vamos a hacer y por qué**.
>
> Versiones más detalladas: `DUPLICATES.md` (técnica, asume *lensing*) y
> `DUPLICATES-EXPLICADO.md` (explica todo desde cero).

---

## 1. El problema

El scraper trae avisos inmobiliarios de muchos proveedores, todo el tiempo. Hay
una base **cruda** (con duplicados) y una base **productiva curada** (ya sin
duplicados). Hoy los duplicados se sacan con **reglas escritas a mano + la opinión
de un LLM**, y consultar al LLM cuesta plata en cada caso.

Queremos un sistema que, **antes de guardar un aviso nuevo**, decida si ya existe;
que use el LLM **solo cuando hay duda real** (y no en cada caso); que funcione
tanto en barridos por lotes como al vuelo cuando entra el aviso; y que **agrupe los
duplicados** para que una persona los revise.

**Qué cuenta como duplicado** (acordado con el negocio):

1. Mismo inmueble publicado por **proveedores distintos** (texto/precio/fotos
   diferentes).
2. **Reposts** del mismo proveedor.
3. Mismo inmueble **re-publicado más tarde con otro precio** → no se descarta: se
   **actualiza el aviso y se guarda el precio anterior** como historial.
4. **A evitar:** dos unidades **distintas pero casi idénticas** (mismo edificio,
   otro depto) **no** son el mismo inmueble.

Y tenemos una ventaja clave: del proceso de curación sabemos **qué avisos fueron
eliminados por duplicados y a cuál corresponden**. O sea, tenemos **ejemplos reales
de duplicados** para que el sistema aprenda de ellos.

---

## 2. Sobre qué lo construimos

Lo vamos a construir sobre **lensing**, un framework interno que ya se usó en otros
dos proyectos sobre esta misma base de avisos. Para este plan alcanza con saber una
sola cosa de *lensing*:

> *lensing* sabe **entrenar, comparar y servir modelos que hacen una predicción por
> aviso** ("dado este aviso, ¿cuál es su precio?", "¿es anómalo, sí/no?"). Ya trae
> hecho lo de armar los datos, probar varios modelos, quedarse con el mejor y
> exponerlo por una API web para responder en vivo.

Reutilizar todo eso nos ahorra muchísimo trabajo. **El único detalle a tener en
cuenta:** *lensing* razona **de a un aviso por vez** — le das uno, te responde sobre
ese uno.

---

## 3. La idea central: convertir cada par en un "registro"

La pregunta de duplicados es **de a pares**: "¿el aviso A y el aviso B son el mismo
inmueble?". Pero *lensing* piensa de a uno. El truco del plan es simple:

> Tratamos **cada par de avisos (A, B)** como si fuera **un registro**. Para ese
> registro calculamos varias **comparaciones entre A y B** (ver punto 5) y le
> pedimos a *lensing* que prediga **una sola etiqueta: ¿son el mismo inmueble?
> (sí/no)**.

Así el problema "de a pares" entra en la maquinaria "de a uno" que ya existe, sin
reinventarla.

---

## 4. No comparamos todos contra todos

Comparar cada aviso con todos los demás es inviable. En su lugar, **para cada aviso
buscamos solo unos pocos candidatos plausibles** y comparamos contra esos. Para
encontrarlos usamos lo que ya tenemos:

- la base vectorial nos devuelve rápido **los avisos más parecidos** (por
  embedding) a uno dado;
- y además exigimos que estén en la **misma zona** y en una **banda de precio
  parecida**.

Esto es lo que más código nuevo necesita: hoy *lensing* no hace esta búsqueda de
"parecidos", así que esa pieza se programa desde cero. Y es **lo primero que hay
que validar**: si un duplicado verdadero no aparece entre los candidatos, ya no hay
forma de detectarlo después. La prueba concreta: *¿esta búsqueda recupera al menos
~98% de los duplicados que ya conocemos?*

---

## 5. Qué mira el modelo de cada par

Para cada par calculamos ~20 comparaciones, por ejemplo:

- qué tan parecidos son los textos (similitud de embeddings),
- diferencia de **precio** y de **precio por m²**,
- diferencia de **superficie**, **ambientes**, **dormitorios**, **baños**,
- **distancia geográfica**,
- parecido de **dirección** (incluyendo piso/unidad),
- si son del **mismo proveedor** y cuánto tiempo pasó entre publicaciones.

**Por qué no alcanza con "qué tan parecido es el texto":** el caso difícil son dos
deptos distintos del mismo edificio — textos y ubicación casi idénticos. Lo que los
distingue es que **la superficie o los ambientes no coinciden**, o que el **piso/
unidad es otro**. Por eso comparamos esos detalles finos, y por eso conviene un
modelo que los combine en lugar de una sola regla de "si se parecen, son el mismo".

---

## 6. Qué se hace con el resultado

El modelo responde una sola cosa: **¿es el mismo inmueble?** Lo que se hace después
lo decide una **regla simple y clara** mirando datos que ya tenemos del par:

- mismo inmueble + precio parecido + (otro proveedor o repost) → **descartar
  duplicado**;
- mismo inmueble + precio distinto + fecha posterior → **actualizar el aviso +
  guardar precio anterior en el historial**;
- no es el mismo → **es nuevo, se inserta**.

**El LLM, solo en la zona de duda:** ponemos dos umbrales sobre el puntaje del
modelo. Puntaje alto → se actúa solo; puntaje bajo → se inserta solo; puntaje **en
el medio** → recién ahí **se le pregunta al LLM**. Como la mayoría de los casos son
claros, el LLM se usa mucho menos que hoy → más barato.

---

## 7. Cómo nos aseguramos de medir bien

Para saber si el sistema sirve, lo evaluamos con avisos que no usamos para
construirlo. Hay un cuidado importante: **todos los avisos de un mismo inmueble
tienen que quedar del mismo lado** (todos en "entrenamiento" o todos en "prueba").
Si se mezclan, la evaluación se engaña sola y da resultados buenísimos que después
no se sostienen en producción. Esto exige un pequeño ajuste en cómo se separan los
datos, y es **innegociable**.

---

## 8. ¿Hace falta un modelo que aprenda "en vivo"?

**No.** La recomendación:

- **Entrenar cada tanto, por lotes** (las correcciones de la curación llegan así, no
  en un chorro continuo).
- **Responder en vivo:** el modelo ya entrenado contesta al instante cuando entra
  un aviso. Eso *lensing* ya lo hace por su API; "en vivo" no significa "aprender en
  tiempo real".
- **Reentrenar de forma programada** y **reemplazar el modelo viejo solo si el
  nuevo mide mejor**.
- **Vigilar si los avisos cambian con el tiempo** (un proveedor nuevo, otro estilo)
  para disparar un reentrenamiento cuando haga falta.
- Bonus: las respuestas del LLM en la zona de duda se reaprovechan como ejemplos
  nuevos en cada reentrenamiento, así esa zona se achica y el LLM se usa cada vez
  menos.

---

## 9. Qué hay que construir (alto nivel)

**Se reutiliza de *lensing*:** entrenar/comparar/servir los modelos, y las
herramientas para armar los datos. Casi sin tocar.

**Se construye nuevo:**
1. la **búsqueda de candidatos** (parecidos + filtros) y el **cálculo de las
   comparaciones del par**, en un **único módulo** usado igual al entrenar, al
   barrer en lote y al chequear en vivo;
2. el **armado de ejemplos** a partir de la procedencia (qué fue duplicado de qué);
3. la **regla de acción** + la **compuerta al LLM**;
4. el **agrupador** de duplicados para revisión + la **pantalla de revisión**;
5. el **enganche con el scraper** (chequear antes de insertar);
6. el **reentrenamiento programado** y el **monitoreo** de cambios.

---

## 10. Etapas y verificación

> Los ⛔ son puntos donde, si falla, no se sigue hasta arreglarlo.

1. **Sacar los ejemplos de la procedencia.** Verificar cuántos duplicados hay de
   cada tipo y revisar algunos a mano.
2. ⛔ **Búsqueda de candidatos.** ¿Recupera ≥ ~98% de los duplicados conocidos? Si
   no, ajustar antes de seguir.
3. **Cálculo de comparaciones del par.** Verificar que separen bien, **incluido el
   caso del mismo edificio**.
4. **Armar los datos y la configuración del proyecto.**
5. ⛔ **Separación de datos por inmueble** (punto 7). Verificar que ningún inmueble
   quede en los dos lados.
6. **Comparar modelos** y medirlos contra el sistema actual (reglas + LLM) sobre los
   mismos pares.
7. ⛔ **Elegir el mejor + calibrar umbrales + conectar la regla.** Que al descartar
   automático **casi nunca se equivoque** (descartar de más borra datos buenos), y
   estimar cuánto se ahorra en LLM.
8. **Barrido por lotes + agrupar + pantalla de revisión.** Probar sobre un lote ya
   curado: ¿reproduce las eliminaciones correctas?
9. **Enganche en vivo con el scraper.** Que responda rápido y acierte en casos
   preparados (un duplicado y un "mismo edificio, otra unidad").
10. **Reentrenamiento programado + monitoreo.**

---

## 11. Riesgos

- **Medición engañada** si se mezclan avisos del mismo inmueble → se evita con la
  separación del punto 7.
- **Entrenar y producir con cálculos distintos** → se evita usando un único módulo
  compartido para las comparaciones.
- **Búsqueda de candidatos floja** → le pone techo a todo; por eso se valida
  primero.
- **El caso del mismo edificio** → incluir esos ejemplos difíciles a propósito.
- **Costo del LLM** → calibrar los umbrales para que la zona de duda sea chica.
