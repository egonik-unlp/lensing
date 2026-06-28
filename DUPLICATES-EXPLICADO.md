# Detección de duplicados — Plan explicado desde cero

> Esta es la versión **autoexplicativa** del plan de `DUPLICATES.md`. No asume
> que sepas qué es *lensing*, qué es una base de datos vectorial ni qué es un
> modelo de machine learning. Cada término técnico se define la primera vez que
> aparece (y todos están juntos en el **Glosario**, al final).
>
> Si ya conocés *lensing*, leé directamente `DUPLICATES.md` (versión técnica y
> más compacta).

---

## 0. ¿De qué se trata todo esto, en una frase?

Queremos un sistema que, cada vez que el scraper trae un aviso inmobiliario
nuevo, **decida automáticamente si ese aviso ya existe** (es un duplicado) antes
de guardarlo en la base de datos "limpia" — y que cuando dude, le pase el caso a
una persona (o a un LLM) en vez de equivocarse.

---

## 1. El contexto, con todos los términos explicados

### 1.1 ¿Qué tenemos hoy?

- **Un scraper:** un programa que recorre sitios de inmobiliarias (varios
  proveedores distintos) y baja avisos: precio, ambientes, superficie,
  ubicación, descripción, fotos, etc. Corre **continuamente** (todo el tiempo
  entran avisos nuevos).
- **Una base de datos vectorial (Qdrant):** una base de datos pensada para
  guardar, además de los datos normales de cada aviso, un **embedding**.
  - **Embedding:** la descripción de texto del aviso se pasa por un modelo de
    inteligencia artificial (en este caso de OpenAI) que la convierte en una
    lista de 1536 números (un "vector"). La gracia es que **dos textos que
    significan algo parecido producen vectores parecidos** (cercanos en el
    espacio de 1536 dimensiones). Esto permite buscar "avisos parecidos a este"
    aunque no usen exactamente las mismas palabras.
  - Hay una base **cruda** (todo lo que trae el scraper, con duplicados) y una
    base **productiva curada** (ya sin duplicados). Hoy alguien/algo ya hizo el
    trabajo de sacar duplicados de la curada.
- **El deduplicador actual (el "baseline" a superar):** un sistema que **no usa
  estadística/aprendizaje**: combina **reglas escritas a mano (heurísticas)** con
  **la opinión de un LLM** (un modelo de lenguaje grande, tipo ChatGPT, al que se
  le pregunta "¿estos dos avisos son el mismo inmueble?"). El problema: preguntarle
  al LLM **cuesta plata y tiempo en cada llamada**.

### 1.2 ¿Qué es *lensing*?

*lensing* es un **framework** (una base de código reutilizable) para armar
"laboratorios" de modelos predictivos sobre una base vectorial como la de arriba.
Ya existen dos proyectos hechos con él:

- **price-guesser-models:** estima el **precio** de un aviso.
- **listing-anomaly-detection:** detecta avisos **anómalos** (raros/sospechosos).

Lo importante de *lensing* para este plan: **sabe trabajar muy bien con un
problema de la forma "para cada aviso, predecí UN número o UNA etiqueta".** Por
ejemplo: "para este aviso, ¿el precio es X?" o "para este aviso, ¿es anómalo (sí/no)?".

Mecánicamente, *lensing* ofrece, ya hecho y probado:

- armar **datasets** (tablas de números) a partir de la base vectorial,
- entrenar muchos tipos de **modelos** (árboles, redes neuronales, etc.) y
  compararlos en una tabla de resultados,
- **servir** un modelo entrenado para que responda predicciones por una API web
  (una URL a la que le mandás un aviso y te devuelve la respuesta),
- mantener automáticamente un grupo con "los mejores modelos".

> **La pieza clave a entender:** *lensing* razona **de a un aviso por vez**. Le
> das un aviso, te da una respuesta sobre ESE aviso.

### 1.3 ¿Qué nos pidieron construir?

Un **detector de duplicados** como nuevo proyecto *lensing*, acoplado al scraper,
que:

- **filtre o agrupe** los avisos duplicados **antes** de que entren a la base
  curada,
- **abarate** el sistema actual: usar el LLM (caro) **solo cuando el caso es
  dudoso**, y resolver lo evidente con un modelo barato,
- funcione **tanto en lote (batch) como en vivo (online)** desde la primera
  versión,
- **agrupe los duplicados** encontrados para que una persona los revise.

### 1.4 Definiciones ya acordadas con el negocio

- **Etiquetas disponibles:** tenemos **procedencia**. Es decir, sabemos qué avisos
  de la base cruda fueron eliminados por duplicados **y a qué aviso "ganador"
  (canónico) corresponden**. Esto es oro: nos da **ejemplos reales** de "esto es
  un duplicado de esto otro" para enseñarle al modelo. (En aprendizaje
  supervisado, las **etiquetas** son las respuestas correctas conocidas con las
  que se entrena.)
- **Qué cuenta como "duplicado" (lo que queremos detectar):**
  1. **Mismo inmueble publicado por proveedores distintos** (con texto, precio o
     fotos diferentes). Es el caso difícil e importante.
  2. **Reposts del mismo proveedor** (el mismo aviso vuelto a publicar).
  3. **Mismo inmueble re-publicado más tarde con el precio cambiado.** Ojo: esto
     **también es un match**, pero la **acción** no es "tirar el duplicado", sino
     **"actualizar el aviso existente y guardar el precio viejo en un historial de
     precios"**.
  4. **Trampa a evitar (falso positivo):** dos departamentos **distintos pero casi
     idénticos** en el mismo edificio/emprendimiento. Se parecen muchísimo pero
     **NO son el mismo inmueble** → el sistema **no** debe fusionarlos.

---

## 2. El obstáculo conceptual (y cómo lo resolvemos)

Hay un choque entre el problema y la herramienta:

- **El problema es "de a pares":** la pregunta natural es "¿el aviso A y el aviso
  B son el mismo inmueble?". Involucra **dos** avisos a la vez.
- **lensing piensa "de a uno":** está hecho para "dado UN aviso, predecí algo de
  ESE aviso".

**La solución (el truco central del plan):** convertimos cada **par de avisos
candidatos** en un "registro" único, y le pedimos a *lensing* que prediga, para
ese registro, **una sola etiqueta: `is_match` = ¿son el mismo inmueble? (sí/no)**.

Es decir: tratamos "el par (A, B)" como si fuera "el aviso" del que habla
*lensing*. Para cada par calculamos un puñado de **características comparativas**
(ver 3.2) y eso se convierte en la fila de la tabla. Así reutilizamos casi toda la
maquinaria ya hecha y probada (la misma que usa el proyecto de anomalías, que
también predice un sí/no).

---

## 3. La arquitectura, explicada

### 3.1 No podemos comparar todos contra todos

Si tenemos, digamos, 200.000 avisos, comparar **cada aviso contra todos los
demás** son ~20.000 millones de pares: imposible en la práctica. La técnica
estándar para evitarlo se llama **blocking** (o "generación de candidatos"):

> **Blocking:** en vez de comparar todo contra todo, para cada aviso buscamos solo
> un puñado de **candidatos plausibles** a ser su duplicado, y comparamos
> únicamente contra esos.

¿Cómo encontramos candidatos plausibles barato? Usando lo que ya tenemos:

- **Búsqueda por similitud (ANN):** la base vectorial sabe responder rápido "dame
  los 30 avisos cuyo embedding es más parecido al de este aviso". (**ANN** =
  *Approximate Nearest Neighbors*, "vecinos más cercanos aproximados": encontrar
  los vectores más parecidos sin revisar todos uno por uno.)
- **Filtros de metadata:** además exigimos que el candidato esté en la **misma
  zona** (misma ciudad/barrio o coordenadas cercanas) y en una **banda de precio
  parecida**, etc.

> **Dato importante:** el código actual de *lensing* **no sabe hacer búsqueda por
> similitud** (solo sabe "traeme todos los avisos"). Así que esta parte hay que
> **programarla nueva** sí o sí. Es el corazón del trabajo nuevo.

El blocking define el **techo de recall** del sistema: si un duplicado verdadero
**no aparece** entre los candidatos, ya no hay forma de detectarlo después. (**Recall** =
de todos los duplicados que existen, qué porcentaje logramos siquiera considerar.)
Por eso lo primero que hay que medir es: *¿el blocking encuentra al menos el ~98%
de los duplicados que ya conocemos por la procedencia?*

### 3.2 Qué mira el modelo de cada par (las "características")

Para cada par candidato (A, B) calculamos ~20 números que comparan A con B. Por
ejemplo:

- qué tan parecidos son sus textos (la **similitud de embeddings**),
- cuánto difieren en **precio** y en **precio por m²**,
- cuánto difieren en **superficie**, en **ambientes/dormitorios/baños**,
- qué tan **cerca geográficamente** están (distancia entre coordenadas),
- qué tan parecidas son sus **direcciones** (incluyendo piso/unidad),
- si son del **mismo proveedor**, cuánto tiempo pasó entre publicaciones, etc.

**Por qué esto y no solo "qué tan parecido es el texto":** el caso trampa son dos
departamentos distintos del mismo edificio. Sus textos y su ubicación son casi
idénticos → la similitud sola **no alcanza** para distinguirlos. Lo que los
delata es que **la superficie o la cantidad de ambientes no coinciden**, o que el
**número de unidad/piso es distinto**. Por eso necesitamos las características
comparativas finas, y por eso un modelo que las combina le gana a una regla simple
de "si se parecen mucho, son el mismo".

### 3.3 Un detalle técnico que, si se ignora, arruina todo: el "leakage"

Para saber si un modelo es bueno, se lo entrena con una parte de los datos
(**train**) y se lo evalúa con otra parte que **nunca vio** (**test**). Esto se
llama **train/test split**.

El problema: si un mismo inmueble aparece en varios pares, y algunos de esos pares
caen en *train* y otros en *test*, el modelo "se sabe la respuesta de memoria" y
los resultados de evaluación salen **falsamente buenos** (puede dar 99% en el
laboratorio y ser inútil en producción). A eso se le llama **leakage** (fuga de
información entre train y test).

**La regla de oro:** todos los pares que tocan a un mismo inmueble tienen que
quedar **del mismo lado** (o todos en train, o todos en test). Esto requiere un
pequeño agregado al código (un "split por grupo", donde el "grupo" es el inmueble).
Es la corrección **obligatoria** del plan.

### 3.4 Un solo modelo + reglas simples para decidir la acción

El modelo aprende a responder **una sola cosa**: `is_match` (¿A y B son el mismo
inmueble?). Eso es exactamente lo que nos dice la procedencia, así que es lo más
limpio de enseñar.

**La acción** que se toma después la decide una **regla simple y auditable** (no
otro modelo), mirando datos que ya tenemos del par:

- es match + precio parecido + (otro proveedor o repost) → **descartar el
  duplicado**,
- es match + precio cambiado + fecha posterior → **actualizar el aviso + guardar
  el precio anterior en el historial**,
- no es match → **es un aviso nuevo, se inserta**.

**Dónde entra el LLM (el ahorro):** ponemos dos umbrales sobre el "puntaje" del
modelo.

- Puntaje muy alto → es duplicado, se actúa automático.
- Puntaje muy bajo → no es duplicado, se inserta automático.
- Puntaje en el **medio (zona de duda)** → **recién ahí** se le pregunta al LLM.

Como la mayoría de los casos son claros, el LLM se usa **mucho menos** → más
barato que hoy, que (según entendemos) se lo consulta siempre.

### 3.5 ¿Hace falta un "modelo vivo" (que aprenda en tiempo real)?

Esta fue una pregunta explícita. **Respuesta corta: no, no hace falta aprendizaje
en tiempo real.** La recomendación es:

- **Entrenar en lote (offline), cada tanto.** Las etiquetas vienen del proceso de
  curación, que ya es por lotes; no hay un chorro continuo de respuestas
  correctas que justifique aprender en vivo.
- **Servir en vivo (online).** Esto es distinto de *aprender* en vivo: el modelo
  ya entrenado responde al instante cada vez que llega un aviso nuevo. *lensing*
  ya ofrece esto por su API.
- **Reentrenar de forma programada** (por ejemplo, una tarea automática periódica)
  con los datos más nuevos, y **reemplazar el modelo viejo solo si el nuevo es
  mejor**.
- **Monitorear "drift"** (deriva): vigilar si el tipo de avisos cambia con el
  tiempo (un proveedor nuevo, otro estilo de redacción). Si las cosas se "corren",
  se dispara un reentrenamiento.
- **Bonus:** las respuestas que da el LLM en la zona de duda son etiquetas nuevas
  de buena calidad → se reaprovechan en cada reentrenamiento, y con el tiempo la
  zona de duda se achica (cada vez menos llamadas al LLM).

En resumen: **modelo entrenado por lotes, servido en vivo, reentrenado
periódicamente.** No "vivo" en el sentido de aprender solo en tiempo real.

---

## 4. ¿Qué ya existe y qué hay que construir?

**Ya hecho en *lensing* (se reutiliza casi sin tocar):**
- Los distintos tipos de modelos y la forma de entrenarlos y compararlos.
- El soporte para problemas de **sí/no** y sus métricas de calidad (ya probado en
  el proyecto de anomalías).
- Guardar modelos, **servirlos por API** (responder predicciones en vivo) y
  mantener el grupo de "mejores modelos".
- Las herramientas para armar e inspeccionar datasets.

**Hay que construir (inevitable):**
1. **El blocking + la búsqueda por similitud + el cálculo de características del
   par.** Tiene que ser **un único módulo** usado igual en entrenamiento, en el
   barrido por lotes y en el chequeo en vivo (si se calcularan distinto, el modelo
   fallaría en producción).
2. **El extractor de etiquetas** a partir de la procedencia (qué fue duplicado de
   qué, y agrupar por inmueble).
3. **El armador de la "tabla de pares"** que alimenta a *lensing*.
4. El pequeño **agregado de "split por grupo"** (para evitar el leakage de 3.3).
5. La **regla de acción** + la **compuerta al LLM** con sus umbrales.
6. El **agrupador** que junta todos los duplicados de un mismo inmueble para
   revisión.
7. El **enganche con el scraper** (chequear antes de insertar).
8. La **pantalla de revisión** para que una persona vea los grupos marcados.
9. El **monitoreo de drift** y el **reentrenamiento programado**.

---

## 5. Plan por etapas (con puntos de control que frenan todo)

> "Frenar todo" = si esa verificación falla, no tiene sentido seguir hasta
> arreglarla.

1. **Sacar las etiquetas de la procedencia.** *Verificar:* cuántos duplicados hay
   de cada tipo, qué tan grandes son los grupos, revisar 20 casos a mano.
2. **Probar el blocking y medir el recall.** ⛔ **Control:** ¿encuentra ≥ ~98% de
   los duplicados conocidos? Si no, ajustar antes de seguir. *(Si el blocking no
   los encuentra, nada de lo de abajo importa.)*
3. **Programar el cálculo de características del par.** *Verificar:* que esas
   características de verdad separen duplicados de no-duplicados, **incluido el caso
   trampa del mismo edificio**.
4. **Armar la tabla de pares y la configuración del proyecto.** *Verificar:* que
   *lensing* logre construir un dataset con esa tabla.
5. **Aplicar el split por grupo.** ⛔ **Control:** ningún inmueble aparece a la vez
   en train y en test.
6. **Primera comparación de modelos.** *Verificar:* qué tan bien detectan en el
   test "honesto", y **comparar contra el sistema actual (heurísticas + LLM)**
   sobre los mismos pares.
7. **Elegir el mejor + calibrar los umbrales + conectar la regla de acción.** ⛔
   **Control:** que cuando descarta automáticamente, **casi nunca se equivoque**
   (descartar de más borra datos buenos = el peor error), y estimar cuánto se
   ahorra en llamadas al LLM.
8. **Barrido por lotes + agrupar + pantalla de revisión.** *Verificar:* sobre un
   lote ya curado, ¿reproduce las eliminaciones que sabemos que correspondían?
9. **Enganche en vivo con el scraper.** *Verificar:* que responda lo
   suficientemente rápido y que acierte en casos preparados (un duplicado conocido
   y un "mismo edificio, otra unidad" conocido).
10. **Monitoreo de drift + reentrenamiento programado.** *Verificar:* que solo
    reemplace el modelo si mejora, y que avise si los datos cambian.

---

## 6. Riesgos principales

- **Leakage (resultados falsamente buenos):** mitigado con el split por grupo.
- **Que entrenamiento y producción calculen distinto:** mitigado usando **un solo
  módulo** compartido para las características.
- **Blocking flojo:** si no encuentra los candidatos, pone un techo a todo el
  sistema; por eso se mide y ajusta primero.
- **El caso trampa (mismo edificio, distinta unidad):** asegurarse de incluir esos
  ejemplos difíciles al entrenar.
- **Costo del LLM:** calibrar los umbrales para que la zona de duda (donde se lo
  consulta) sea chica.

---

## 7. Glosario rápido

- **Aprendizaje supervisado:** enseñarle a un modelo dándole ejemplos con la
  respuesta correcta conocida (las "etiquetas").
- **Embedding:** lista de números que representa el significado de un texto; textos
  parecidos → vectores parecidos.
- **Base de datos vectorial / Qdrant:** base de datos que guarda esos vectores y
  sabe buscar "los más parecidos" rápido.
- **lensing:** el framework con el que se arman estos proyectos de modelos.
- **Blocking / generación de candidatos:** quedarse solo con unos pocos pares
  plausibles en vez de comparar todo contra todo.
- **ANN (vecinos más cercanos aproximados):** búsqueda rápida de los vectores más
  parecidos a uno dado.
- **`is_match`:** la etiqueta sí/no que predice el modelo: ¿estos dos avisos son el
  mismo inmueble?
- **Train/test split:** partir los datos en una parte para entrenar y otra,
  nunca vista, para evaluar.
- **Leakage (fuga):** cuando información del test se "cuela" en el entrenamiento y
  los resultados salen falsamente buenos.
- **Recall:** de todos los duplicados reales, qué fracción logramos detectar (o al
  menos considerar).
- **Precisión:** de los que el sistema marca como duplicados, qué fracción lo son
  de verdad.
- **Batch (lote) vs. online (en vivo):** procesar de a tandas grandes cada tanto
  vs. responder al instante caso por caso.
- **Drift (deriva):** cuando los datos nuevos cambian con el tiempo y el modelo
  viejo empieza a quedar desactualizado.
- **LLM:** modelo de lenguaje grande (tipo ChatGPT); acá, el "juez" caro que se
  reserva para los casos dudosos.
- **Canónico:** el aviso "ganador" que se conserva cuando se detecta un grupo de
  duplicados.

---

*Versión técnica y compacta del mismo plan: `DUPLICATES.md`.*
