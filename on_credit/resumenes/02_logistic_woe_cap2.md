# Capítulo 2 — *Logistic Regression: From Maximum Likelihood to Machine Learning*

**Resumen conceptual (foco WOE ↔ regresión logística) + comparación con la metodología propia + 4 preguntas finales**

Libro: *On Credit: Scoring Foundations in the Age of AI* — Denis Burakov (Berlín, 2026).
Fuente del capítulo: `on_credit/book/2.pdf` (36 páginas). Continúa a [`01_woe_cap1.md`](01_woe_cap1.md).

> Nota: este documento **no traduce** el capítulo. Resume las partes necesarias para entender **cómo el WOE se conecta con la regresión logística** (el núcleo §2.1–2.4), y lo contrasta con el código de `on_credit/woe_example/`. Las secciones avanzadas (multinomial §2.5, incertidumbre/conformal §2.6) van como mención breve.

Si el cap. 1 respondía "¿qué es el WOE?", el cap. 2 responde **"¿qué hago con el WOE una vez que lo tengo?"** → alimentar una regresión logística. Y demuestra que ambos son, matemáticamente, **la misma cosa**.

---

## Parte 1 — Resumen conceptual del Capítulo 2

### 2.1 Máxima verosimilitud: el marco de Fisher

En los 1920s la estadística usaba "probabilidad inversa" (Bayes aplicado a estimar parámetros), en descrédito por depender de priors. Fisher (1922) dio vuelta la pregunta:

- Probabilidad inversa: *"¿cuál es la probabilidad de que el parámetro valga θ, dados los datos?"*
- **Verosimilitud (likelihood):** *"¿cuál es la probabilidad de observar estos datos, si el parámetro vale θ?"*

Elegimos el θ que **maximiza la verosimilitud** (MLE). Para el caso binomial (x éxitos en n), la MLE es la respuesta intuitiva: `p̂ = x/n`, la **proporción observada**. Como Turing, Fisher trabajó con el **logaritmo** de la verosimilitud (multiplicar → sumar), que además es más cómodo de optimizar.

**Propiedad clave de la MLE (semilla de la calibración).** En el óptimo, si el modelo tiene intercepto:

```
Σ yᵢ = Σ p̂ᵢ
```

es decir, **el total de eventos observados = el total de eventos predichos**. Si tu modelo predice 100 defaults y observaste 150, algo está mal ajustado. Volvemos a esto en la Parte 3, pregunta 3.

### 2.2 Por qué logit y no probit

En los 1930s el problema era ajustar curvas dosis-respuesta con forma de "S". Bliss propuso el **probit** (basado en la normal acumulada); Berkson (1944) propuso el **logit** (la función logística `1/(1+e^−(α+βx))`). El logit ganó por tres razones:

- **Simplicidad computacional:** derivada cerrada, sin tablas de la normal.
- **Odds-ratios interpretables:** `log(p/(1−p))` da log-odds que se exponencian a odds-ratios (cap. 1).
- **Tratabilidad matemática.**

Lo importante para hoy: **la log-verosimilitud negativa del logit ES la log loss / cross-entropy** que se usa en todo el ML moderno:

```
Log Loss = − (1/n) Σ [ yᵢ·log(p̂ᵢ) + (1−yᵢ)·log(1−p̂ᵢ) ]     (ec. 2.9-2.10)
```

Corolario del libro: **cada red neuronal o modelo de boosting entrenado para clasificar está usando el principio de Fisher de 1922 — máxima verosimilitud disfrazada.** La log loss castiga **exponencialmente** los errores confiados: predecir 0.19 para un caso que era 1 cuesta 1.66; predecir 0.79 cuesta solo 0.23.

### 2.3 Regresión logística a mano — y WOE como logística "centrada" (LA SECCIÓN CLAVE)

El libro toma un ejemplo 2×2: predecir default según mora (delincuencia > 5 días).

| Mora | Default (y=1) | No default (y=0) | Total |
|---|---:|---:|---:|
| > 5 días (x=1) | 80 | 100 | 180 |
| ≤ 5 días (x=0) | 20 | 500 | 520 |
| **Total** | **100** | **600** | 700 |

**Solución cerrada de la logística** `log(p/(1−p)) = β₀ + β₁·x`:

```
Grupo x=1:  p̂₁ = 80/180 = 0.444   → odds₁ = 0.80   → log-odds₁ = −0.223
Grupo x=0:  p̂₀ = 20/520 = 0.0385  → odds₀ = 0.04   → log-odds₀ = −3.219

β₀ = log-odds₀ = −3.219          ← el intercepto es el log-odds del grupo de referencia (x=0)
β₁ = log-odds₁ − log-odds₀ = −0.223 − (−3.219) = 2.996
odds-ratio = e^2.996 = 20        ← mora alta ⇒ 20× los odds de default
```

**El puente formal: WOE como logística centrada.** En vez de medir cada grupo "relativo a la referencia", el WOE lo mide **relativo al log-odds poblacional** (la tasa de default global):

```
log-odds poblacional = log(100/600) = −1.792

WOE₁ (mora alta) = log-odds₁ − (−1.792) = −0.223 − (−1.792) = +1.569
WOE₀ (mora baja) = log-odds₀ − (−1.792) = −3.219 − (−1.792) = −1.427

ΔWOE = WOE₁ − WOE₀ = 1.569 − (−1.427) = 2.996 = β₁     ← ¡el coeficiente de la logística!
```

**Esto es la equivalencia central del capítulo: la diferencia de WOE entre dos bins es exactamente el coeficiente β₁ de la regresión logística.** WOE y logística no son dos pasos distintos; son la misma recta en log-odds, vista con dos orígenes distintos (la referencia vs. la media poblacional).

**Por qué el centrado importa (footnote 6 del libro).** Con **una** variable es solo otra parametrización. Pero con **múltiples** categóricas, los coeficientes de one-hot **dependen de qué categoría elijas como referencia** (cambiás la referencia, cambian todos los números); los coeficientes WOE, al estar centrados en la tasa poblacional, **quedan comparables entre variables** sin esa arbitrariedad. Esta es una ventaja real del WOE-encoding, no un detalle cosmético.

**Errores estándar (para inferencia).** El SE del WOE de un grupo es el de su log-odds (restar una constante no cambia la varianza):

```
Var(WOE_grupo) = 1/n_eventos + 1/n_no_eventos
SE(β₁) = √( SE(WOE₀)² + SE(WOE₁)² )
```

### 2.4 Optimización iterativa (cuando no hay forma cerrada)

El ejemplo 2×2 tiene solución cerrada porque es un predictor binario con datos agrupados. En el mundo real (muchas variables, continuas, interacciones) **no hay fórmula cerrada** y se optimiza iterando:

- **Gradient descent:** predecir `p̂ = sigmoide(Xβ)`, calcular el residual `r = p̂ − y`, actualizar `β ← β − α·Xᵀ(p̂−y)`. Simple pero lento y sensible al learning rate `α`.
- **Fisher Scoring / IRLS** (*iteratively reweighted least squares*): usa información de segundo orden (la matriz de información de Fisher `I = XᵀWX`, con `W = diag(p̂(1−p̂))`). Converge mucho más rápido y **sin** tunear learning rate. Es lo que usan por dentro `statsmodels`, el `glm()` de R, SAS y (con variantes) `sklearn`.

#### Recuadro clave: "WOE es solo un encoding" es un malentendido

Este recuadro del libro es el corazón conceptual para tu práctica. Desmonta la idea de que "el WOE es un paso de encoding y la logística hace el trabajo real":

1. **Bajo independencia (naive-Bayes), el WOE ya ES el modelo.** Sumando WOEs directamente obtenés el log-odds posterior (esto es exactamente el cap. 1):
   ```
   log( P(Y=1|X) / P(Y=0|X) ) = log-odds_poblacional + Σⱼ WOEⱼ(xⱼ)      (ec. 2.44)
   ```
   Cada `WOEⱼ` es una **regresión logística univariada** de esa variable contra el target.

2. **Pero las features correlacionan** → sumar WOEs crudos sobre-cuenta evidencia (predicciones sobre-confiadas). La solución (Spiegelhalter & Knill-Jones) es usar el WOE **como input** de una logística que **pondera**:
   ```
   log( P(Y=1|X) / P(Y=0|X) ) = β₀ + Σⱼ βⱼ · WOEⱼ(xⱼ)                    (ec. 2.45)
   ```
   Los `βⱼ` son **factores de ajuste** por dependencia entre variables. Si las features fueran realmente independientes, esperaríamos `βⱼ ≈ 1`; las desviaciones miden cuánto se viola la independencia (más en la Parte 3, pregunta 2).

3. **Lectura meta-learner / ensemble.** Cada WOE es un "voto" (una logística univariada); la logística final es un **meta-learner** que aprende cuánto pesar cada voto, descontando redundancia. *"Por esto el método sigue siendo competitivo con el ML moderno: combina feature engineering univariado fuerte con calibración multivariada flexible."*

> Footnote a del recuadro: en textos de crédito la fracción del WOE suele invertirse para que puntajes altos = mejor calidad crediticia, lo que **invierte el signo de los coeficientes** — exactamente la convención de tu `utils.py` (ver Parte 2).

### 2.5 y 2.6 — Mención breve

- **Multinomial / softmax (§2.5).** El logit binario se generaliza a K clases con la función **softmax**; la pérdida sigue siendo cross-entropy. El libro lo ejemplifica con default DPD (*days past due*) vs UTP (*unlikely to pay*) vs no-default, y muestra que el WOE multiclase también funciona (razones de verosimilitud por clase). Relevante si modelás **estados de mora / collections**. Nota ordinal: para categorías con orden natural (ratings AAA→D, buckets 30/60/90 dpd) conviene **regresión logística ordinal**.
- **Incertidumbre (§2.6).** Cómo poner intervalos sobre la PD: **Wald** (clásico, de la matriz de covarianza) vs **conformal prediction** (distribution-free, vía residuos de Pearson). Detalle en la Parte 3, pregunta 4.

### 2.7 — Cierre histórico

GLM (Nelder-Wedderburn, 1972) unifica la logística con otros modelos vía *link functions* e IRLS; McFadden (discrete choice, Nobel 2000); regularización (ridge/LASSO, 1990s); ensembles (AdaBoost y gradient boosting con **pérdida logística**); y el despliegue a escala web (Google/Facebook). Puente hacia los caps. 3-4 (redes neuronales y boosting/SHAP).

---

## Parte 2 — Comparación con tu metodología (`utils.py` + `z_bivariados.ipynb`)

### 2.1 Tu pipeline ES el de §2.4

Tu flujo (bivariados → binning → `aplicar_woe` → alimentar una regresión logística) es **literalmente** el `logit = β₀ + Σ βⱼ·WOEⱼ` de la ec. 2.45. El cap. 2 le da dos cosas que antes eran implícitas:
- El **fundamento formal**: `ΔWOE = β₁` (§2.3). Tu encoding no es un truco previo al modelo; es el modelo escrito en otra base.
- La **lectura meta-learner**: cada columna `_woe` es una logística univariada; tu logística final pondera esos votos.

### 2.2 El signo, otra vez (cierra el loop del cap. 1)

El libro define WOE y log-odds **a favor de default**: `WOE₁` (mora alta) = **+1.569**, positivo = más riesgo. Tu `calcular_metricas_bin` ([utils.py:82-86](../woe_example/utils.py#L82-L86)) calcula `ln(porcbuenostotal/porcmalostotal)` = `ln(%buenos/%malos)`, que **invierte el signo** → en tu escala, WOE positivo = **menos** riesgo. Consecuencia práctica en la logística:
- Con tu convención, un WOE bien construido y monótono tiende a dar **βⱼ positivos** (más "bondad" → mayor score de bueno).
- El **intercepto** se interpreta en tu escala (log-odds de bueno, no de default).
- Lo único que importa: **consistencia intercepto ↔ features** en todo el pipeline. No mezclar convenciones.

### 2.3 Tu WOE ya está "centrado en la población"

Como calculás el WOE con `%buenos_total` y `%malos_total` (distribuciones sobre el total de buenos y de malos), tu WOE **ya está centrado en la tasa poblacional** — es el análogo exacto del "overall log-odds" del libro. Por eso tus coeficientes son **comparables entre variables** y no dependen de una categoría de referencia arbitraria: la ventaja de footnote 6, que ya estás aprovechando sin nombrarla.

### 2.4 Lo que falta en el repo: el paso logístico

Hoy `woe_example` llega hasta **generar el encoding WOE** (`calcular_woe_mapping` / `aplicar_woe`). El cap. 2 es el "qué viene después": **ajustar la logística** sobre las columnas `_woe`, revisar los `βⱼ` (¿cercanos a 1?, ¿algún signo raro?), y validar calibración. Aclaración importante: la fórmula cerrada del libro (`β₁ = ΔWOE`) **solo vale para el caso agrupado 2×2**; con tus decenas de variables continuas/categóricas el ajuste es **iterativo** (IRLS de `statsmodels`, o `sklearn.LogisticRegression`). El valor de `ΔWOE = β₁` te sirve como **intuición y sanity check** univariado, no como el estimador final multivariado.

---

## Parte 3 — Las 4 preguntas

### 3.1 ¿Por qué WOE-encoding + logística, en vez de one-hot + logística?

| | WOE-encoding | One-hot |
|---|---|---|
| Columnas por variable | **1** | tantas como categorías (−1) |
| Alta cardinalidad | compacto | explota la dimensionalidad |
| Escala | log-odds, centrada en la tasa poblacional | binaria, relativa a una referencia |
| Coeficientes | **comparables**, no dependen de la referencia | dependen de la categoría de referencia elegida |
| Relación con el target | **linealizada** en log-odds (monotonicidad) | escalonada por categoría |
| Missing / outliers | absorbidos por el binning | requieren categoría/tratamiento aparte |

En resumen: el WOE mete el conocimiento univariado (dirección y fuerza de cada bin) **dentro de la escala del modelo**, dando una logística compacta, estable e interpretable. Es la razón por la que domina en scorecards regulados. **Costos** del WOE (no gratis): (a) riesgo de **leakage** si no congelás el mapeo y los cortes de train (cap. 1 §2.5); (b) **pérdida de interacciones** dentro del bin (todo lo que cae en el mismo bin recibe el mismo valor); (c) un paso más de mantenimiento (recalcular WOE si cambia la población).

### 3.2 ¿Qué significa `βⱼ ≈ 1`?

En `logit = β₀ + Σ βⱼ·WOEⱼ`, como cada `WOEⱼ` ya viene en unidades de log-odds, el coeficiente `βⱼ` te dice **cuánto de la evidencia univariada conserva el modelo multivariado**:

- **`βⱼ ≈ 1`** → la variable aporta su evidencia univariada **tal cual**; se cumple la independencia naive-Bayes, no hay redundancia con otras variables.
- **`βⱼ < 1`** (lo más común) → parte de su señal ya la aportan otras variables correlacionadas; la logística la **descuenta** para no doble-contar.
- **`βⱼ > 1`** → sinergia: la variable vale más en presencia de las otras.
- **`βⱼ ≤ 0`** → **alarma**: multicolinealidad fuerte, WOE mal construido (no monótono), o una **inversión de signo** accidental (mezclaste convenciones). Revisar.

Es un **diagnóstico práctico y barato** de tu WOE-logistic: mirá la tabla de coeficientes y esperá valores positivos y en un rango razonable alrededor de 1 (con tu convención good-to-bad, positivos). Coeficientes muy grandes o negativos son banderas rojas.

### 3.3 ¿Mi scorecard queda calibrado?

**A nivel poblacional, sí, por construcción.** La propiedad MLE `Σy = Σp̂` (§2.1, exige intercepto) garantiza que **la PD media predicha = la tasa de default observada en train**, y el intercepto absorbe el log-odds poblacional. Es calibración "en promedio" gratis. Matices que no cubre esa garantía:

- **Calibración global ≠ por segmento.** Que el promedio cierre no implica que cada decil de score esté bien calibrado. Verificá con un gráfico de calibración / tabla observado-vs-esperado por deciles de score.
- **Vale en train, no en OOT.** Si hay *drift* entre train y la población futura, la calibración se degrada; por eso se re-calibra periódicamente (a veces solo el intercepto).
- **Requiere intercepto.** Sin término `β₀` la propiedad no se cumple.

### 3.4 ¿Cómo pongo intervalos de confianza sobre la PD?

Dos caminos (§2.6):

- **(a) Wald (clásico).** Propagás la incertidumbre de los coeficientes: `Cov(β̂)` = inversa de la información de Fisher; para un caso `x₀`, `Var(ĝ₀) = x₀ᵀ Cov(β̂) x₀` en escala **logit**; armás el intervalo `ĝ₀ ± z·SE(ĝ₀)` y lo pasás por la **sigmoide** para llevarlo a probabilidad. Barato y estándar (`statsmodels` lo da con `get_prediction`). Limitación: es **asintótico** — poco fiable con muestras chicas o PD cerca de 0/1.
- **(b) Conformal prediction (moderno).** *Distribution-free*: usás **residuos de Pearson** en un **set de calibración** aparte para construir bandas con **cobertura garantizada en muestra finita**, sin asumir normalidad. Cuesta reservar datos de calibración. Nota fina: en crédito querés un intervalo sobre la **probabilidad p̂** (p. ej. "la PD está entre 10% y 20%"), no el *prediction set* de clases del conformal clásico — el libro usa la variante "conformalizada" para probabilidades.

Regla práctica: **Wald** para el día a día (rápido, integrado); **conformal** cuando la cobertura es crítica (capital regulatorio, decisiones de pricing) y tenés datos de sobra para calibrar.

---

## Puentes con el resto del material

- **Con el cap. 1 ([01_woe_cap1.md](01_woe_cap1.md)):** el cap. 1 definió el WOE (Bayes en log-odds); el cap. 2 muestra que **sumar WOEs y ponderarlos = regresión logística** (`ΔWOE = β₁`). La inversión de signo de tu `utils.py` reaparece idéntica.
- **Hacia adelante (caps. 3-4):** la log loss de §2.2 es la misma que minimizan redes neuronales y gradient boosting → el "WOE es solo el caso lineal" de una familia más amplia. Conecta con tu pregunta del cap. 1 sobre XGBoost.

## Referencias del capítulo (bibliografía del libro)

- R. A. Fisher, *On the Mathematical Foundations of Theoretical Statistics* (1922) — máxima verosimilitud.
- J. Berkson, *Application of the Logistic Function to Bio-assay* (1944) — el logit.
- C. I. Bliss, *The Calculation of the Dosage-Mortality Curve* (1935) — el probit.
- D. Spiegelhalter & R. Knill-Jones (1984) — WOE como input de logística en diagnóstico.
- Nelder & Wedderburn, *Generalized Linear Models* (1972); McCullagh & Nelder (1989).
- Hosmer & Lemeshow, *Applied Logistic Regression* (2000) — Wald / inferencia.
- V. Manokhin, *Practical Guide to Applied Conformal Prediction in Python* (2023).
