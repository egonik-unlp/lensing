# Plan de implementación — Detección de duplicados (instancia *lensing*)

> Documento de **diseño/plan**, no de implementación. Redactado en español según lo pedido.
> Repos de referencia: `lensing` (framework), `price-guesser-models` y `listing-anomaly-detection` (instancias hermanas).

---

## 1. Contexto

Snappler mantiene una base de datos vectorial (Qdrant) de avisos inmobiliarios alimentada por **scraping continuo de múltiples proveedores**. La base **productiva está curada** (se le quitaron duplicados, varios no obvios). Hoy la deduplicación corre con un clasificador **no estadístico en producción = heurísticas + opinión de un LLM** (caro por llamada).

El objetivo es construir un **detector de duplicados estadístico** como nueva instancia *lensing*, acoplado al scraper, que:

- filtre/agrupe resultados duplicados **antes** de insertarlos en la base productiva,
- abarate el baseline actual reservando el LLM solo para la **franja ambigua**,
- sirva **tanto en barrido batch como en línea (al momento del scraping)** desde v1,
- **agrupe los duplicados para revisión manual**.

**Decisiones ya confirmadas con el usuario:**

- **Etiquetas:** existe **procedencia** (sabemos qué puntos crudos se eliminaron y a qué punto canónico sobreviviente mapean) → tenemos **pares/grupos de duplicados reales** para entrenamiento supervisado y evaluación.
- **Alcance de "duplicado" (clase positiva):**
  1. **mismo inmueble entre proveedores distintos** (texto/precio/fotos diferentes),
  2. **reposts del mismo proveedor**.
  - Una tercera relación — **mismo inmueble re-listado más tarde con precio cambiado** — es también un *match*, pero su **acción es "actualizar el aviso canónico + agregar al historial de precios"**, no descartar.
  - **Negativo difícil a respetar:** unidades **distintas casi idénticas** (mismo edificio/emprendimiento, otro departamento) **NO deben matchear**.
- **Modo operativo:** batch **y** online desde v1.
- **Baseline a superar:** heurísticas + LLM.

**Tensión arquitectónica central y su resolución.** *lensing* predice **un target por punto**; la deduplicación es **pareada/relacional** ("¿A y B son el mismo inmueble?"). La estrategia es **reducir el problema pareado a clasificación binaria por punto**, donde cada "punto" es un **par candidato** con features de par y target `is_match ∈ {0,1}`. Así se reutiliza casi todo el framework (datasets, predictores, runs, best-models, métricas binarias) tal como ya lo hace la instancia de anomalías. El paso **pareado/relacional (blocking) vive aguas arriba**, fuera del pipeline Rust.

---

## 2. Arquitectura recomendada

### 2.1 Reducción: el par como "punto"

Se materializa una **colección Qdrant `candidate-pairs`**. Cada punto = un **par candidato** (listing A, listing B) con:

- **payload = features de par ingenierizadas** (numéricas + categóricas planas),
- **`is_match`** (target binario, derivado de procedencia),
- **`group_id`** (id de cluster de inmueble, ver split),
- los dos `row_id` fuente + datos de procedencia (para la UI de revisión),
- **sin embedding real de par**: se apaga PCA (`pca_dims = 0`; si el path 0 falla en el spike, usar `pca_dims = 1` con vector constante como workaround). La señal discriminante está en las features ingenierizadas, **no** en concatenar/restar dos embeddings de 1536-d (eso es caro y redundante una vez que se tiene el coseno como escalar).

`domain.toml` apunta `[corpus].collection = "candidate-pairs"` y *lensing* lo trata como un corpus binario común — clonando el patrón de `listing-anomaly-detection/domain.toml` (target binario, `primary = "RMSE"`, columnas `["RMSE","ROC-AUC","AP","P@pos","R²"]`). Las reglas de calidad por punto (`duplicate-content`, `nonpositive-price`, `price-*`) se **desactivan** (no tienen sentido sobre un par).

**Por qué colección Qdrant y no emitir artefactos directos:** así se conservan gratis el preflight de `dataset-design`, el análisis de redundancia, `items.json`/inspector del server y el almacenamiento de procedencia para la UI. (Se descarta extender el pipeline Rust para construir pair-datasets nativamente — invasivo, solo revisar si el builder Python se vuelve cuello de botella.)

### 2.2 Features de par (núcleo del valor)

~20 features, calculadas por un **único módulo compartido** (ver 2.4). Deben incluir las que separan *mismo inmueble* de *mismo edificio, otra unidad*:

- coseno de embeddings; `precio` ratio/Δ; `precio/m²` consistencia; ratios de `totalArea`/`coveredArea`;
- (des)acuerdo de `bedrooms`/`bathrooms`/`rooms`; distancia **haversine** de coordenadas;
- **similitud de dirección** (Jaccard/Levenshtein) sobre la dirección **completa incl. piso/unidad**; **booleano "match de token de unidad"**;
- flag **mismo proveedor**; **Δ tiempo** (`createdAt`); acuerdo de `propertyType`/`neighborhood`/`city`.

> Insight clave: haversine ≈ 0 tanto para mismo-inmueble como para mismo-edificio → necesario pero no suficiente. Lo que rompe el empate (negativo difícil) es **ratio de área ≠ 1 + desacuerdo de ambientes + token de unidad distinto**. Por eso un modelo aprendido supera un umbral geométrico.

### 2.3 Generación de candidatos / *blocking* (código nuevo, recall-driven)

El pipeline *lensing* **no tiene primitiva de búsqueda ANN** (solo `scroll_all`) → el blocking es **código nuevo** sí o sí.

- **ANN:** por cada listing, `query` vectorial top-k en la colección viva, **filtrada por claves de blocking** metadata.
- **Claves de blocking:** celda geográfica (geohash ~250–500 m) o `city`+`neighborhood`; banda de precio en log (con **adyacentes**, y **relajada/desactivada** para la relación 3 "re-listado con precio cambiado" — el precio es *feature*, no compuerta dura); acuerdo grueso de `propertyType`/`bedrooms`.
- **k:** arrancar en 20–50; **ajustar por recall de positivos de procedencia** (elegir el menor k que recupere ≥ ~98 % de positivos conocidos).

### 2.4 Contrato compartido entrenamiento↔servicio (anti train/serve skew)

Factorizar blocking + featurización en **una sola función** `candidates_and_features(listing, live_index) -> [pair_feature]`, importada por **(1)** la construcción del set de entrenamiento, **(2)** el barrido batch y **(3)** el hook online. Misma filosofía que el contrato congelado de featurización de *lensing* (`inference.rs`). **Batch = el run síncrono iterado** sobre el corpus existente (el listing nuevo aún no está en la DB, así que online = "query contra vivo").

### 2.5 Split consciente de grupos (corrección obligatoria)

El split estándar (`shuffle.rs::train_test_split`) es Fisher-Yates por fila y **filtra (leakage)**: si pares del mismo cluster de inmueble caen a ambos lados, las métricas se inflan (AUC ~0.99 en lab, inútil en prod).

- **Fix recomendado (1 cambio Rust, ~30 líneas, *upstreamable*):** agregar a `build_dataset` un modo `split = "by_group"` que lea un `group_field` y haga Fisher-Yates sobre **`group_id` distintos** (no sobre filas), expandiendo luego a índices. `group_id` = componente conexa (union-find) del grafo de positivos `(raw_id, canonical_id)`.
- **Stopgap sin tocar Rust:** emitir dos colecciones `candidate-pairs-train` / `-test` ya particionadas por grupo. Más feo; solo si se rechaza el cambio Rust en v1.
- **Guard:** ningún `group_id` debe aparecer en ambos lados del split (chequeo sobre `row_ids`).

### 2.6 Salida de tres vías: un modelo + router determinístico

**Un solo modelo binario aprendido (`is_match`)** + **router determinístico** para la acción (no tres modelos):

- el modelo responde solo "¿A y B son el mismo inmueble?" — exactamente lo que supervisa la procedencia;
- la **acción** es función determinística y auditable de `(is_match, mismo_proveedor, Δtiempo, Δprecio)`:
  - match & precio≈ & (cross-proveedor o repost) → **descartar duplicado**,
  - match & precio cambiado & timestamp posterior → **actualizar canónico + historial de precios**,
  - no-match → **único (insertar)**.
- **Franja ambigua → LLM:** dos umbrales sobre el score. `> τ_high` auto-acción; `< τ_low` auto-rechazo; en `[τ_low, τ_high]` se llama al **LLM existente solo en esa rebanada**. Calibrar `τ` contra procedencia en el test split para una **precisión alta de auto-descarte** (un descarte equivocado pierde datos — es el error más costoso).
- **Negativos difíciles:** minar de procedencia los **sobrevivientes NO colapsados** que son vecinos cercanos (alto coseno, misma celda) → `is_match=0` gold-standard; asegurarse de que el blocking los exponga como candidatos.

### 2.7 Pregunta del "modelo vivo": NO hay aprendizaje online

**Recomendación: entrenamiento batch + servicio online + reentrenamiento programado + monitoreo de drift.** Justificación:

- *lensing* es un entrenador batch con **contratos congelados** promovidos a modelos inmutables servidos por consenso; el aprendizaje online pelearía con toda esa arquitectura (split sembrado, PCA congelada, contrato train==serve) y tiraría la reproducibilidad.
- Las **etiquetas vienen de la procedencia de curación**, que es batch → no hay flujo continuo que consumir con SGD online.
- **Servicio online** ya lo da `POST /api/best-models/predict` (consenso, scoring instantáneo) → cubre el requerimiento síncrono sin infra nueva.
- **Reentrenamiento programado** (cron): rearmar `candidate-pairs` con la procedencia + corpus al día, reconstruir dataset, lanzar el zoo de predictores, **promover solo si supera al titular** en la métrica primaria.
- **Monitoreo de drift:** distribución de features de par + histograma de scores en el tiempo; alarmar si crece la fracción en la franja-LLM (proveedor nuevo / cambio de estilo) o si cae el recall de blocking sobre una muestra de procedencia.
- **Bonus (active learning batch):** las decisiones del LLM en la franja ambigua son etiquetas nuevas de alta calidad → realimentarlas en cada reentrenamiento; la franja se **achica con el tiempo** (más barato).

---

## 3. Componentes: reutilizado vs. nuevo

**Reutilizado de *lensing* tal cual (cero/casi-cero cambio):**
- Todo el zoo de predictores + contrato stdin/stdout JSON-lines (xgboost, lightgbm, random-forest, burn-mlp, ridge, blend, baseline-median).
- Path binario `Task::Binary` + métricas RMSE/ROC-AUC/AP/P@pos/R² (probado por la instancia de anomalías).
- Runs, promoción, MODELS, contrato congelado, `POST /api/models/{name}/predict`, `POST /api/best-models/predict`.
- Preflight de `dataset-design`, análisis de redundancia, inspector manifest/`items.json`, export ONNX.
- Mecanismo `domain.toml`, patrón de scripts standalone en `analysis/`, skills `bootstrap`/`model-definitions`.
- `features::assemble`/`Encoder` para las features de par como numéricas + categóricas planas.

**Nuevo (inevitable):**
1. **Módulo de blocking + cliente ANN + featurizador de par** compartido (`candidates_and_features`). Requiere una primitiva nueva `search`/`query` en Qdrant (hoy no existe).
2. **Extractor procedencia → etiquetas + `group_id`** (union-find; minería de negativos difíciles).
3. **Builder de la colección `candidate-pairs`** (estilo `analysis/`): corre 1+2, `upsert` de pair-points.
4. **Split por grupo** — el único cambio Rust recomendado en `build_dataset` (`split = "by_group"`).
5. **Router de acción** determinístico + **compuerta LLM** de franja media con umbrales calibrados.
6. **Agrupamiento union-find** sobre las aristas de match del barrido → grupos de revisión.
7. **Hook de integración con el scraper** (scoring síncrono pre-inserción).
8. **UI de revisión** de grupos marcados (partir de `analysis/ui` de la instancia de anomalías + inspector del server).
9. **Monitor de drift + orquestación de reentrenamiento** (cron + watcher de métricas; realimenta etiquetas del LLM).

---

## 4. Archivos críticos

**Framework (a leer / cambiar mínimamente):**
- `crates/lensing-pipeline/src/qdrant.rs` — **agregar** primitiva ANN `search`/`query` (hoy solo `scroll_all`/`count`/`fetch_points`/`upsert_raw`).
- `crates/lensing-pipeline/src/build.rs` — path de build; sitio del cambio `split="by_group"` y donde verificar `pca_dims = 0`.
- `crates/lensing-core/src/shuffle.rs` — `train_test_split` es por fila (filtra); el split por grupo lo envuelve.
- `crates/lensing-pipeline/src/inference.rs` — contrato congelado de featurización que el servicio reutiliza; el featurizador de par debe coincidir.
- `crates/lensing-pipeline/src/features.rs` / `quality.rs` — encoder reutilizado; reglas a desactivar.

**Plantillas a clonar (instancia hermana):**
- `listing-anomaly-detection/domain.toml` — plantilla de target binario (RMSE primario, ROC-AUC/AP/P@pos, reglas por punto desactivadas).
- `listing-anomaly-detection/analysis/anomaly_scan.py` + `tag_anomalies.py` — patrón standalone para el extractor de procedencia y el builder de `candidate-pairs`.
- `listing-anomaly-detection/analysis/ui` — base para la UI de revisión.

**Nuevos (instancia de duplicados):**
- `domain.toml` (corpus `candidate-pairs`, target `is_match`, métricas binarias).
- `analysis/dedup_blocking.py` (módulo compartido `candidates_and_features`).
- `analysis/provenance_labels.py` (procedencia → pares + `group_id`).
- `analysis/build_pairs_collection.py` (upsert a Qdrant).
- `analysis/action_router.py` (router + compuerta LLM).
- `analysis/group_review.py` (union-find + UI).
- Hook del scraper + cron de reentrenamiento + monitor de drift.

---

## 5. Orden de implementación y verificación (con *stop-the-line*)

1. **Extracción de etiquetas de procedencia (offline).** Set de positivos + grupos union-find. **Verif:** conteo de positivos por relación (cross-proveedor / repost / re-listado-precio), distribución de tamaño de grupo, spot-check manual de 20 pares; cuantificar desbalance de clase.
2. **Spike de blocking + medición de recall.** ANN + metadata contra la DB viva. ⛔ **Gate:** recall de blocking ≥ ~98 % de positivos de procedencia al k elegido; registrar candidatos-por-listing (costo). Ajustar k y claves acá, antes de modelar.
3. **Módulo `candidates_and_features` + diseño de features.** **Verif:** separación univariada en muestra etiquetada (coseno, ratio de área, token de unidad); confirmar presencia y distinguibilidad de negativos difíciles (mismo edificio).
4. **Materializar `candidate-pairs` + `domain.toml`.** Stub de embedding / `pca_dims=0`, `is_match`, `group_id`, ids fuente, procedencia. **Verif:** preflight de `dataset-design` construye un dataset; layout de columnas = features de par; conteos de clase = paso 1.
5. **Split por grupo.** **Verif (gate):** ningún `group_id` en ambos lados del split; construir train/test.
6. **Primer bake-off de modelos.** xgboost/lightgbm/random-forest/blend vía `model-definitions`. **Verif:** ROC-AUC/AP/P@pos en test honesto; comparar con precisión/recall del baseline heurísticas+LLM sobre los mismos pares; sanity-check contra un control de split *con* leakage (confirmar que el split por grupo importa).
7. **Promoción + calibración de umbrales + router.** Promover el mejor run; calibrar `τ_low`/`τ_high` contra procedencia. ⛔ **Gate:** precisión de auto-descarte ≥ objetivo; fracción de franja media aceptable; estimar volumen de llamadas LLM vs. baseline all-LLM (la ganancia de costo).
8. **Barrido batch + union-find + UI de revisión.** **Verif:** end-to-end sobre un batch de curación conocido — ¿reproduce las eliminaciones conocidas?; revisar muestra de auto-descartes por falsos merges (mismo edificio, otra unidad).
9. **Hook síncrono del scraper.** Mismo módulo, query-contra-vivo, router. **Verif:** presupuesto de latencia (ANN + featurizar + predict) bajo el SLA del scraper; correctitud sobre listings inyectados duplicado-conocido y unidad-distinta-conocida.
10. **Monitor de drift + reentrenamiento programado (con feedback de etiquetas LLM).** **Verif:** un reentrenamiento forzado promueve solo si mejora la métrica; la alarma de drift dispara ante un shift sintético; la franja media se achica tras incorporar etiquetas adjudicadas.

**Gates que detienen la línea:** (2) recall de blocking — si no se exponen los positivos barato, nada aguas abajo importa; (5) leakage de split — si los grupos filtran, toda métrica es ficción; (7) precisión de auto-descarte — un descarte equivocado destruye datos, el único error realmente costoso.

---

## 6. Riesgos

- **Leakage por grupo** (mitigado por split por grupo + guard).
- **Train/serve skew** (mitigado por módulo de featurización único compartido).
- **Recall de blocking insuficiente** → techo de recall del sistema; mitigar ajustando k/claves antes de modelar.
- **`pca_dims=0` no soportado end-to-end** → workaround `pca_dims=1` con vector constante (verificar en el spike del paso 4).
- **Negativos difíciles (mismo edificio)** sub-representados → minería explícita desde procedencia.
- **Costo/latencia del LLM** en franja media → calibrar umbrales para acotar volumen; achicar franja vía active learning.
