# Capítulo 1 — *Weight of Evidence: From Bletchley Park to Credit Scoring*

**Resumen conceptual (foco WOE) + comparación con la metodología propia + respuestas finales**

Libro: *On Credit: Scoring Foundations in the Age of AI* — Denis Burakov (Berlín, 2026).
Fuente del capítulo: `on_credit/book/1.pdf` (el capítulo 1 completo).

> Nota: este documento **no traduce** el capítulo. Resume solo las partes necesarias para entender el **concepto** de Weight of Evidence (WOE), con foco en la sección 1.3 (ejemplo de credit scoring), y lo contrasta con el código de `on_credit/woe_example/`.

---

## Parte 1 — Resumen conceptual del Capítulo 1

El capítulo cuenta una sola idea, vista desde tres ángulos: **acumular evidencia sumando logaritmos**. Turing lo usó para romper Enigma; los bancos lo usan hoy cada vez que alguien aprieta "Solicitar crédito".

### 1.1 El origen: Turing, Banburismus y los *decibans*

En Bletchley Park (1941–1943) Turing necesitaba decidir cuál de millones de configuraciones de Enigma había cifrado los mensajes navales del día. No podía probarlas todas, así que diseñó **Banburismus**: un sistema de puntajes para **acumular evidencia** hasta que una hipótesis superara a las demás.

La innovación clave fue **convertir la inferencia bayesiana de multiplicativa en aditiva**:

- Bayes en forma de *odds* exige **multiplicar** razones de verosimilitud (likelihood ratios), algo costoso a mano.
- Turing tomó **logaritmos**: así multiplicar se vuelve **sumar**.
- Escaló esos logaritmos (`×10`, en base 10) y los redondeó a enteros → unidades llamadas **decibans** (`10·log₁₀`).
- El **medio-decibán** es, según Jack Good, aproximadamente el menor cambio de evidencia perceptible por la intuición humana.

Resultado: todo el proceso se reducía a **sumar y restar números enteros** con lápiz y papel. La misma idea reaparece hoy como el clasificador **naive Bayes** (filtros de spam, diagnóstico médico, scoring).

### 1.2 El fundamento matemático: el *Factor Principle*

El "Factor Principle" de Turing es Bayes en forma de odds:

```
Odds a posteriori = Odds a priori × Bayes factor
```

donde

```
                P(evidencia | hipótesis verdadera)
Bayes factor = ────────────────────────────────────
                P(evidencia | hipótesis falsa)
```

El factor es el ratio de verosimilitudes (cuánto más probable es la evidencia si la hipótesis es cierta vs. si es falsa). Si hay varias evidencias **independientes**, sus factores **se multiplican**.

Tomando logaritmos:

```
log(odds posteriori) = log(odds a priori) + log(Bayes factor)
```

**Ese `log(Bayes factor)` es el Weight of Evidence (WOE).** Es decir: el WOE es el aporte aditivo, en escala log-odds, de una pieza de evidencia.

#### Relación con el teorema de Bayes "de manual"

El Bayes que se enseña hoy es:

```
           P(B | A) · P(A)
P(A | B) = ────────────────
                P(B)
```

Leído en clave de scoring: `A` = "el cliente hace default" (la **hipótesis**), `B` = "observo cierto valor de la variable" (la **evidencia**, p. ej. utilización >90%). `P(A|B)` es lo que queremos: la probabilidad de default **dado** lo que observamos.

El problema de esta forma es el denominador `P(B)` (la probabilidad marginal de la evidencia), que es incómodo de calcular. **El truco de Turing/Good es escribir Bayes para la hipótesis y para su complemento, y dividir** — así `P(B)` se cancela:

```
P(A | B)     P(B | A) · P(A)          P(B | A)     P(A)
────────  =  ───────────────────  =  ──────────  · ──────
P(Ā | B)     P(B | Ā) · P(Ā)          P(B | Ā)     P(Ā)
└────────┘                            └────────┘   └────┘
odds posteriori                       Bayes factor  odds priori
```

Eso **es** el Factor Principle: `odds posteriori = Bayes factor × odds priori`. El paso a logaritmos convierte el producto en suma y aparece el WOE. Dicho de otro modo: **trabajar en odds (en vez de probabilidades) es lo que hace que el molesto `P(B)` desaparezca** y que la evidencia se vuelva aditiva.

#### ¿Qué son los *odds*?

Los **odds** (cuotas/momios) son una forma alternativa de expresar una probabilidad, como ratio "a favor : en contra" en vez de "casos / total":

```
odds = p / (1 − p)            p = odds / (1 + odds)
```

- `p = 0.10` (10% de default) → `odds = 0.10/0.90 = 1/9 ≈ 0.111` → "1 a 9 en contra del default".
- `p = 0.50` → `odds = 1` (1 a 1, parejo).
- `p = 0.80` → `odds = 4` (4 a 1 a favor).

Propiedades que importan acá:
- Probabilidad vive en `[0, 1]` (acotada); los odds viven en `[0, ∞)`; y el **log-odds** (`logit`) vive en `(−∞, +∞)`. Esta última escala, **sin topes**, es la que permite **sumar** evidencia libremente sin chocar contra el 0 o el 1.
- El **logit** (`log-odds`) es exactamente la escala en la que trabaja la regresión logística: `logit(p) = log(p/(1−p))`. Por eso el WOE (que ya está en log-odds) se suma directamente al output del modelo.

**Conexión con teoría de la información (Shannon, 1948).** El "contenido de información" de un suceso es `h(x) = −log P(x)`: los sucesos improbables sorprenden más. WOE, log-verosimilitud y contenido de información son **la misma operación matemática**: cuantifican, en logaritmos de probabilidad, cuánto cambia nuestra creencia. La propiedad central que las une es que **la evidencia de fuentes independientes se suma** (en lugar de multiplicarse).

### 1.3 WOE en credit scoring — la sección clave

El libro traslada el Factor Principle al riesgo crediticio. La hipótesis ahora es "el cliente entra en *default*" y la evidencia es el valor de una variable (un *feature*).

**Ejemplo del libro (utilización de tarjeta > 90% vs. default).** Tabla de contingencia sobre 1000 clientes:

| Utilización | No default (0) | Default (1) | Total |
|-------------|---------------:|------------:|------:|
| ≤ 90%       | 810            | 20          | 830   |
| > 90%       | 90             | 80          | 170   |
| **Total**   | **900**        | **100**     | 1000  |

Para el grupo de **alta utilización (>90%)**:

- De los 100 *defaulters*, 80 tienen utilización >90% → `P(>90% | default) = 80/100 = 0.80`.
- De los 900 *no-defaulters*, 90 tienen utilización >90% → `P(>90% | no default) = 90/900 = 0.10`.

**Bayes factor (likelihood ratio):**

```
Factor = (80/100) / (90/900) = 0.80 / 0.10 = 8.0
```

Los clientes de alta utilización son **8 veces más propensos** a ser *defaulters* que *no-defaulters*.

**Weight of Evidence (formulación del libro, "bad-to-good", logaritmo natural):**

```
WOE = ln(8.0) = 2.08
```

**WOE solo NO da una probabilidad.** El WOE mide *cuánto mueve* la evidencia la balanza, pero el punto de partida (qué tan riesgosa es la cartera **antes** de mirar la variable) lo aporta el **prior**. Por eso hay que **sumar el WOE al log-odds a priori** y recién ahí volver a probabilidad. El recorrido completo, paso a paso:

```
1) Prior (tasa base de default en la cartera):  p₀ = 100/1000 = 0.10
2) Prior en odds:        odds₀ = p₀/(1−p₀) = 0.10/0.90 = 1/9 ≈ 0.111
3) Prior en log-odds:    log(odds₀) = log(1/9) = −2.20      ← punto de partida
4) Sumo la evidencia:    log(odds post) = −2.20 + 2.08 = −0.128   (WOE = +2.08)
5) Vuelvo a probabilidad (sigmoide):
   PD = σ(−0.128) = 1 / (1 + e^(−(−0.128))) = e^(−0.128)/(1 + e^(−0.128)) = 0.47
```

**Qué es la "PD del grupo".** PD = *Probability of Default*: la probabilidad de default **del subgrupo** definido por la evidencia, es decir `P(default | utilización > 90%) = 0.47`. **El 47% de los clientes con utilización >90% terminan en default.** No es la probabilidad de un cliente individual (eso requeriría sumar los WOE de *todas* sus variables), sino la del bin.

Y esto coincide con el cálculo **directo/frecuentista**: en el grupo >90% hay 80 defaults sobre 170 clientes → `80/(80+90) = 0.47`. Que el camino bayesiano (prior + WOE + sigmoide) y el conteo directo den **exactamente lo mismo** es justamente el punto del capítulo: el WOE no es una aproximación, es Bayes reescrito en escala aditiva.

**Por qué necesitás el prior — un contraejemplo.** Imaginá que la cartera tuviera 50% de default base (`odds₀ = 1`, `log-odds₀ = 0`). El **mismo** WOE de +2.08 daría `PD = σ(0 + 2.08) = σ(2.08) = 0.89`. Misma evidencia (factor 8), pero PD muy distinta (0.47 vs 0.89), porque cambió el punto de partida. **El WOE es relativo; la probabilidad final depende del prior.** Por eso un WOE aislado no es interpretable como probabilidad: solo dice "esta evidencia multiplica los odds por 8".

**Dos formas equivalentes de calcular el WOE** (ambas dan factor 8.0):
1. Desde la **tabla de contingencia / probabilidades condicionales** (la de arriba).
2. Desde las **tasas de default** por grupo y los odds (forma de Bayes en odds de Good).

**Cierre del capítulo — el límite de la independencia.** El marco aditivo asume que las evidencias son **independientes** (la hipótesis naive-Bayes). Peirce ya lo había intuido en 1878, pero cometió un error: sumaba evidencias **sin** incluir los odds a priori (el intercepto). Turing y Good corrigieron el álgebra: la condición de independencia correcta es sobre las **verosimilitudes**, no sobre las posteriori. La consecuencia práctica: cuando los features están **correlacionados o interactúan**, el WOE puro se queda corto; por eso el scoring moderno sumó gradient boosting y redes neuronales. Aun así, el WOE sigue siendo valioso por **interpretabilidad**, **cumplimiento regulatorio** y **eficiencia**.

#### Idea para llevarse
> Un scorecard de crédito es Banburismus: cada variable aporta un WOE (un puntaje en log-odds), se **suman** todos los WOE más el log-odds base (intercepto), y al pasar por la sigmoide se obtiene la probabilidad de default. Mismo motor matemático que rompió Enigma.

---

## Parte 2 — Comparación con tu metodología (`utils.py` + `z_bivariados.ipynb`)

Tu flujo: leés una tabla de variables crudas, corrés `calcular_bivariados` por quintiles/deciles, ves si la variable **ordena** monótonamente contra el target, agrupás (quintiles, deciles, nivel cero, etc.) cuando hace falta, y finalmente convertís las categóricas a numéricas con el **WOE** que sale de `calcular_woe_mapping` / `aplicar_woe`. Conceptualmente **es exactamente lo que describe el cap. 1**. Hay un punto fino de convención y un par de notas técnicas.

### 2.1 Diferencia de signo en la definición de WOE (lo más importante)

| | Fórmula | WOE positivo significa |
|---|---|---|
| **Libro** (bad-to-good) | `ln(%malos / %buenos)` | **más** riesgo |
| **Tu `calcular_metricas_bin`** (good-to-bad, Siddiqi) | `ln(%buenos / %malos)` | **menos** riesgo |

En [utils.py:82-86](../woe_example/utils.py#L82-L86):

```python
df_grouped['woe'] = np.where(
    (df_grouped['porcmalostotal'] == 0) | (df_grouped['porcbuenostotal'] == 0),
    0,
    np.log(df_grouped['porcbuenostotal'] / df_grouped['porcmalostotal'])   # ln(%buenos / %malos)
)
```

Las dos fórmulas son **negativas una de la otra** (`WOE_libro = −WOE_tuyo`). **No es un error**: vos usás la convención de Naeem Siddiqi (*Credit Risk Scorecards*), la más habitual en la banca, donde un WOE alto = cliente "bueno". El propio libro lo aclara en su **footnote 2 (pág. 11)**: avisa que la formulación "good-to-bad" alinea con scorecards tradicionales (puntaje alto = más solvente) pero que puede generar inconsistencia de signo con el **intercepto** (que representa el log-odds base de *default*). La formulación "bad-to-good" del libro hace que WOE>0 ⇒ más riesgo.

**Qué hacer con esto:** ninguna acción urgente, pero tené presente el signo al (a) leer la tabla bivariada —en tu tabla, el grupo más riesgoso tiene WOE **negativo**— y (b) montar la regresión logística: el signo de los coeficientes y la interpretación del intercepto dependen de qué convención uses. Lo crítico es **ser consistente** en todo el pipeline.

### 2.2 Coincidencias conceptuales (vas bien)

- Tu "asignar el WOE de cada bin a la variable" es literalmente lo que el libro llama *"logarithmic scoring models assign WOE values to features"*.
- Tu paso siguiente (WOE → regresión logística que pondera y suma) es el *maximum likelihood sobre features WOE-transformadas* del capítulo. El feature WOE ya viene en **unidades de log-odds**, que es la escala natural del logit.
- Tu IV por variable (`iv_total_variable`) es el criterio estándar de poder predictivo univariado, perfectamente alineado con el espíritu del capítulo (medir cuánta "evidencia" aporta cada variable).

### 2.3 Monotonicidad: por qué tu requisito tiene sentido

Exigís que la variable "ordene" monótonamente (creciente o decreciente) con el target antes de aceptarla o agruparla. Eso **no es un capricho estético**: la regresión logística es **lineal en log-odds**, y el WOE **lineariza** la relación variable↔log-odds. Si el WOE de los bins es monótono, el logit puede aprovechar la variable con un único coeficiente de signo claro e interpretable. Una relación no-monótona "rota" esa linealidad y por eso la reagrupás hasta que ordene. El marco aditivo de log-odds del cap. 1 es justamente la justificación teórica de esta práctica.

### 2.4 Nota técnica — escala del IV (×100)

En [utils.py:68-92](../woe_example/utils.py#L68-L92), `porcbuenostotal` y `porcmalostotal` están en **puntos porcentuales** (multiplicados por 100), no en proporciones. Como `iv = (%buenos − %malos) · woe`, tu IV queda en una escala **~100×** la convencional. Por eso ves `iv_total_variable` de ~105 o ~113, cuando el IV "de libro" (con proporciones 0–1) daría ~1.05 o ~1.13.

- No afecta el **ranking** de variables (es un factor de escala constante), así que para ordenar/seleccionar funciona igual.
- Pero **sí importa** si comparás contra los umbrales clásicos de IV: `<0.02` inútil · `0.02–0.1` débil · `0.1–0.3` medio · `0.3–0.5` fuerte · `>0.5` sospechoso (posible leakage). Para usarlos, dividí tu IV por 100 (o pasá los porcentajes a proporciones en el cálculo).

### 2.5 Nota técnica — congelar los **cortes** de binning, no solo el WOE

Tu práctica real (aclarada): calculás el **WOE en train** y aplicás ese mismo valor en test — perfecto, eso evita el *leakage* del target, que es el riesgo principal. **Pero** el binning en test lo recalculás con los **quintiles del propio test**, no con los cortes del train. Ahí hay un desajuste sutil que conviene corregir.

**El problema.** El WOE se asocia a una **etiqueta de bin** (`percentil_25`, `percentil_50`, …), pero esa etiqueta solo significa lo mismo si representa el **mismo rango de valores crudos**. Si los cortes de quintil se recalculan sobre test:

- El corte del quintil 1 en train puede ser, p. ej., `edad ≤ 35`, pero en test (otra distribución) puede ser `edad ≤ 38`.
- Un cliente de 37 años cae en `percentil_25` en train pero en `percentil_50` en test → se le aplica un WOE **distinto** al que le correspondería por su valor crudo.
- Resultado: el WOE de train (estimado para "edad ≤ 35") se aplica a un rango de test que **no es el mismo**. El mapeo deja de ser consistente y se contamina la evaluación out-of-sample (especialmente si hay *drift* entre train y test).

**La regla correcta (en scoring se llama "congelar la grilla").** Los **cortes** (bordes de quintil/decil) y el **mapeo WOE** son **dos artefactos del train** que se transportan juntos a test/OOT:

1. En **train**: calculás los cortes de quintil → asignás bins → calculás el WOE por bin. Guardás *ambos*: cortes y tabla WOE.
2. En **test/OOT**: aplicás los **cortes del train** (un cliente de 37 cae donde caería según train), y sobre ese bin aplicás el **WOE del train**.

Así, "estar en el quintil 1" significa exactamente el mismo rango de valores en train y test, y el WOE asignado es coherente con cómo se estimó.

**Lo bueno: tu código ya soporta esto.** `discretizar_variables` ([utils.py:238](../woe_example/utils.py#L238)) permite `calcular_percentiles=False` + `percentiles_precalculados=<los de train>`, justamente para **reutilizar los cortes del train** en test. El ajuste es de **flujo**, no de código: calcular cortes y WOE en train, y en test pasar los cortes precalculados (en vez de dejar que `pd.qcut` recalcule los quintiles sobre test). Hoy estás fijando el WOE pero dejando que el binning se recalcule; falta fijar también la grilla de cortes.

> En síntesis: **dos cosas se aprenden en train y se aplican congeladas en test** → (a) los **bordes de los bins** y (b) la **tabla WOE por bin**. Vos ya congelás (b); falta congelar (a).

### 2.6 El mismo ejemplo del libro, en las dos notaciones (y por qué el IV NO es el factor principle)

Esta es la traducción exacta, término a término, entre el cálculo del libro (§1.3) y tus columnas de `calcular_metricas_bin`. Tomamos el bin **utilización > 90%** (buenos=no default=90, malos=default=80; totales 900 buenos / 100 malos).

**Paso 1 — Tus columnas SON las verosimilitudes del libro.**

```
porcbuenostotal = cantbuenos / cant_buenos_total = 90/900 = 0.10 = P(evidencia | good) = P(>90% | no default)
porcmalostotal  = cantmalos  / cant_malos_total  = 80/100 = 0.80 = P(evidencia | bad)  = P(>90% | default)
```

**Paso 2 — El factor (Bayes factor) es el ratio de tus dos columnas.**

```
Factor = P(ev|bad) / P(ev|good) = porcmalostotal / porcbuenostotal = 0.80 / 0.10 = 8.0
```

**Paso 3 — El WOE es el log del factor; tu `.py` toma el log invertido.**

```
WOE_libro = ln(porcmalostotal / porcbuenostotal) = ln(8)   = +2.08
WOE_tuyo  = ln(porcbuenostotal / porcmalostotal) = ln(1/8)  = −2.08     ← lo que calcula utils.py
```

Relación cerrada: **`WOE_tuyo = −WOE_libro`** y **`Factor = e^(WOE_libro) = e^(−WOE_tuyo)`**. Mismo número, signo invertido (§2.1). El *factor principle* es la actualización `odds posteriori = Factor × odds priori = 8.0 × odds priori`; da igual en qué convención lo escribas, siempre que el prior use la misma dirección (odds de default vs. odds de no-default).

**Tabla de equivalencia (bin >90%):**

| Concepto del libro | En tus columnas | Valor |
|---|---|---|
| `P(evidencia \| bad)` | `porcmalostotal` (proporción) | 0.80 |
| `P(evidencia \| good)` | `porcbuenostotal` (proporción) | 0.10 |
| `Factor` (Bayes factor) | `porcmalostotal / porcbuenostotal` | 8.0 |
| `WOE` (bad-to-good) | `−woe` (el negativo de tu columna) | +2.08 |
| — (good-to-bad, tu convención) | `woe` | −2.08 |

**¿Es el IV el factor principle? No — son cosas distintas.**

| | Qué es | Alcance | Para qué sirve |
|---|---|---|---|
| **Factor / WOE** | `factor = 8.0`; `WOE = ln(8) = 2.08` | **por bin** | Actualizar los odds de un caso (Bayes) |
| **IV** | `Σ_bins (%buenos − %malos) · WOE` | **variable entera** (suma de bins) | Medir poder predictivo / seleccionar variables |

El factor principle es una operación **por bin** (multiplicás los odds por el factor de ese bin). El **IV agrega toda la variable**: suma la contribución de todos los bins para decir "qué tan bien separa buenos de malos en conjunto" (formalmente, la **divergencia KL simétrica** entre la distribución de buenos y la de malos). No actualiza la probabilidad de ningún cliente.

IV de este ejemplo (2 bins):

```
Bin ≤90%: (0.90 − 0.20) · ln(0.90/0.20) =  0.70 · (+1.504) = 1.053
Bin >90%: (0.10 − 0.80) · ln(0.10/0.80) = (−0.70)·(−2.079) = 1.456
IV total = 2.51     (≈ 251 en tu escala ×100, ver §2.4)
```

Detalle revelador: el **IV es invariante al signo de la convención** — al invertir buenos↔malos se dan vuelta *tanto* `(%buenos−%malos)` *como* `WOE`, y su producto no cambia. Por eso el IV siempre es positivo e idéntico en ambos enfoques, mientras que el WOE sí cambia de signo. Eso prueba que WOE/factor (por bin, **direccional**) e IV (agregado, **no direccional**) miden cosas diferentes.

---

## Parte 3 — Las dos preguntas finales

### 3.1 ¿Es necesario agrupar/binnear variables cuando se usan algoritmos no-logit, como árboles de *boosting* (XGBoost)?

**No, en general no — y muchas veces es contraproducente.**

- El binning + WOE existe para **linealizar** features para un modelo **lineal** (regresión logística), que solo captura relaciones lineales en log-odds. Es una muleta para una limitación del logit.
- Los árboles (XGBoost, LightGBM, CatBoost) **parten por umbrales** y hacen su propio "binning" **en cada split**, de forma supervisada y adaptada al target. Capturan **no-linealidades, relaciones no-monótonas e interacciones** de forma nativa.
- Pre-binnear a quintiles/deciles y reemplazar por WOE **descarta granularidad** e impone monotonía. Eso puede **degradar** la performance del árbol: le quitás los cortes finos que él mismo encontraría y le prohibís descubrir patrones no-monótonos (p. ej. riesgo alto en los extremos y bajo en el centro).

**Cuándo el binning/WOE sigue siendo útil con árboles (excepciones):**
- **Categóricas de alta cardinalidad:** WOE / target encoding las vuelve numéricas y compactas. (Aunque LightGBM maneja categóricas nativamente y XGBoost tiene soporte categórico; CatBoost las trata de fábrica.)
- **Interpretabilidad / cumplimiento regulatorio:** si necesitás un scorecard explicable o trazable, el binning ayuda aunque cueste algo de performance.
- **Robustez:** estabiliza outliers y reduce sobreajuste en datasets chicos o ruidosos.

**Regla práctica:** para **pura performance** con boosting, alimentá **valores crudos** (y manejo nativo o target encoding para categóricas de alta cardinalidad). Reservá WOE/binning para el modelo **logístico/scorecard** o cuando la **interpretabilidad** sea un requisito. El propio cap. 1 (págs. 14–15) lo respalda: reconoce que la independencia naive-Bayes se rompe con features correlacionados/interactuantes y que por eso el scoring moderno incorporó gradient boosting y redes.

### 3.2 ¿Es el WOE una forma de *target encoding*? ¿Es la única?

**Sí, el WOE es una forma de *target encoding* (codificación supervisada):** reemplaza cada categoría/bin por una **estadística derivada del target** —el log-ratio de las distribuciones de buenos y malos—. Lo distintivo del WOE es que está en **unidades de log-odds**, por eso encaja "de fábrica" con la regresión logística (el feature ya viene en la escala del modelo).

**No, no es la única.** Todas comparten la idea "reemplazar la categoría por una estadística del target", pero difieren en *qué* estadística y *cómo* se protegen del leakage:

| Encoding | Qué pone en la categoría | Unidades | Idea clave |
|---|---|---|---|
| **WOE** | `ln(%buenos / %malos)` (o su inverso) | log-odds | Aporte aditivo de evidencia; encaja con el logit |
| **Mean / target encoding** (clásico) | tasa del evento, `P(y=1 \| cat)` | probabilidad | La media del target por categoría, a secas |
| **Suavizado / regularizado** | media de la categoría mezclada con la media global | probabilidad | Estabiliza categorías con pocos casos |
| **Leave-one-out (LOO)** | media del target excluyendo la fila actual | probabilidad | Reduce leakage fila a fila |
| **K-fold / out-of-fold** | media estimada en folds distintos al de la fila | probabilidad | El estándar anti-leakage hoy |
| **Ordered target encoding** (CatBoost) | media acumulada según un orden aleatorio | probabilidad | Anti-leakage incorporado al modelo |

Detalle de cada una:

- **Mean / target encoding (clásico).** Reemplaza la categoría por la **media del target** en esa categoría: `enc(cat) = mean(y | cat)`. Simple y potente, pero **se sobreajusta** brutal en categorías raras: una categoría con un solo registro de `y=1` recibe `1.0`, lo cual es ruido, no señal. De ahí nacen las variantes siguientes.

- **Suavizado / regularizado (smoothing).** Mezcla la media de la categoría con la media global, ponderando por el tamaño del grupo:
  ```
  enc(cat) = (n_cat · mean_cat + m · mean_global) / (n_cat + m)
  ```
  Con `m` (peso del prior) las categorías chicas se "tiran" hacia la media global y las grandes mandan su propia media. Es **exactamente la lógica bayesiana / empirical Bayes** (un prior que se diluye con más datos). Variantes con nombre: **additive/Laplace smoothing**, **M-estimate**, **James-Stein**. *(El WOE también admite suavizado, sumando `0.5` a buenos y malos por bin para evitar `log(0)` — algo que tu `calcular_metricas_bin` hoy resuelve poniendo WOE=0 cuando un bin no tiene buenos o malos.)*

- **Leave-one-out (LOO).** Para cada fila, la media de su categoría calculada **sin esa fila**. Evita que el registro "se vea a sí mismo", pero deja una correlación residual con el target que puede filtrarse; suele necesitar ruido añadido.

- **K-fold / out-of-fold (OOF).** Partís el train en *K* folds; el encoding de cada fold se calcula con los **otros** folds. Es **el estándar moderno** para no filtrar el target dentro del train. Para test/OOT se usa el encoding calculado con todo el train.

- **Ordered target encoding (CatBoost).** CatBoost lo hace **internamente**: recorre los datos en un orden aleatorio y codifica cada fila solo con las filas **anteriores** (estadística "online"). Anti-leakage por diseño, sin tener que orquestar folds a mano. Es una de las razones de su buena performance con muchas categóricas.

> Distinto: **frequency / count encoding** (reemplaza por la frecuencia/conteo de la categoría) **no** usa el target → no es target encoding. Y **one-hot / ordinal / hashing** tampoco. Son "unsupervised encodings".

#### ¿Qué se usa en las soluciones ganadoras de Kaggle?

Patrón consistente en competiciones tabulares (riesgo, propensión, etc.):

- **El caballito de batalla es el target/mean encoding con K-fold (OOF) y smoothing**, alimentando modelos de boosting (LightGBM/XGBoost/CatBoost). Es la receta canónica para categóricas de **alta cardinalidad** (IDs, códigos postales, producto×región, etc.), donde one-hot explota en dimensionalidad y el modelo de árbol no puede partir bien.
- **CatBoost** gana terreno justamente porque trae el *ordered target encoding* incorporado: en datasets con muchas categóricas suele ser fuerte "out of the box".
- **WOE casi no aparece como tal en Kaggle.** ¿Por qué? Porque (a) el WOE está optimizado para **modelos lineales / interpretabilidad** (su escala log-odds), y en Kaggle gana la **performance** con boosting, donde el mean-encoding OOF rinde igual o mejor sin atarse a log-odds; y (b) Kaggle no tiene la **restricción regulatoria** de la banca, que es la que mantiene vivo al WOE/scorecard. En crédito real, el WOE sigue siendo el rey por explicabilidad y cumplimiento; en Kaggle, el ganador típico es **OOF mean target encoding + boosting** (a menudo CatBoost), más feature engineering y *ensembling/stacking*.
- Regla mental: **misma familia (target encoding), distinta prioridad.** Banca regulada → WOE (interpretable, log-odds, monótono). Kaggle/performance pura → mean encoding OOF/CatBoost (flexible, sin compromiso de signo ni monotonía).

**Matiz práctico (clave), que conecta con §2.5:** todos estos encodings comparten el mismo talón de Aquiles, el **leakage del target**. La protección es calcular la estadística **fuera de la fila/fold que se codifica** (OOF, LOO, ordered) y **congelar el encoding del train** para aplicarlo a test/OOT. Tu WOE de train aplicado a test ya respeta lo segundo; lo que falta afinar es congelar también los **cortes de binning** (ver §2.5).

---

## Referencias del capítulo (bibliografía del libro)

- A. M. Turing, *The Applications of Probability to Cryptography*, National Archives HW 25/37 (1941, desclasificado 2012).
- I. J. Good, *Probability and the Weighing of Evidence* (1950); y *"An error by Peirce concerning weight of evidence"* (1981).
- C. S. Peirce, *"The Probability of Induction"*, Popular Science Monthly (1878).
- C. E. Shannon, *"A Mathematical Theory of Communication"* (1948).
- D. J. C. MacKay, *Information Theory, Inference and Learning Algorithms* (2003).
- N. Siddiqi, *Credit Risk Scorecards* (2006) — origen de la convención "good-to-bad" que usás en `utils.py`.
