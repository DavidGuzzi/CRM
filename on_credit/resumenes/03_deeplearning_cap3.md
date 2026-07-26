# Capítulo 3 — *Deep Learning: Credit Risk and Waves of AI*

**Resumen conceptual del capítulo + conexión con la metodología propia + 4 preguntas finales**

Libro: *On Credit: Scoring Foundations in the Age of AI* — Denis Burakov (Berlín, 2026).
Fuente del capítulo: `on_credit/book/3.pdf` (22 páginas). Continúa a [`01_woe_cap1.md`](01_woe_cap1.md) y [`02_logistic_woe_cap2.md`](02_logistic_woe_cap2.md).

> A diferencia de los caps. 1-2 (centrados en el WOE con ejemplos numéricos), el cap. 3 es **histórico-conceptual**: cuenta cómo las redes neuronales y el credit scoring evolucionaron en paralelo, y llega hasta el presente (transformers, agentes LLM). Por eso este documento pesa más en el **resumen** (Parte 1) y más liviano en la comparación con tu código (Parte 2).

**Hilo del capítulo:** ¿pueden las máquinas *aprender*? Turing lo imaginó en 1948; tardó 80 años en volverse práctico. Mientras tanto, el crédito resolvió su propia versión del problema —el scorecard— por un camino más simple y regulado. Hoy ambos caminos convergen.

---

## Parte 1 — Resumen conceptual del Capítulo 3

### 3.1 Los orígenes olvidados de las redes neuronales

- **Neurona de McCulloch-Pitts (1943).** Primer modelo matemático de neurona artificial: suma ponderada de entradas + umbral → dispara (1) o no (0). Con eso se implementan operaciones lógicas (AND, OR, NOT). Construido directamente sobre *On Computable Numbers* de Turing.
- **"Máquinas no organizadas" de Turing (1948).** En un reporte poco conocido (*Intelligent Machinery*), Turing propuso algo distinto a cablear neuronas a mano: redes de neuronas conectadas **al azar** que **aprenden por entrenamiento**. Fue, en la práctica, el **primer concepto de red neuronal entrenable** — 10 años antes del Perceptron. Idea clave que anticipa el deep learning: no diseñar el circuito, sino dejar que la red se adapte con la experiencia.

### 3.2 El Perceptron

Rosenblatt (1958) construyó el **Perceptron**: la primera red neuronal en hardware que aprendía, acuñando el término "connectionist". Su regla de aprendizaje se inspiró en Hebb ("*cells that fire together, wire together*"): ajustar pesos según los errores.

**Conexión clave con lo que ya sabés (cap. 2):** el Perceptron es **una sola neurona** con activación **escalón** (sale 0 o 1). La regresión logística es la misma neurona pero con activación **sigmoide** (sale una probabilidad). Es decir:

> **La regresión logística es una red neuronal sin capas ocultas.**

Ambas calculan `Σ wᵢxᵢ + b` y deciden con un umbral; solo cambia si la salida es dura (escalón) o suave (sigmoide).

**La limitación XOR (Minsky-Papert, 1969).** Un Perceptron solo traza fronteras **lineales**, así que no puede resolver el XOR ("exactamente uno de dos, pero no ambos ni ninguno"). En clave crédito:

| Deuda alta | Ingreso bajo | ¿Riesgoso? |
|---|---|---|
| No | No | No |
| No | Sí | **Sí** |
| Sí | No | **Sí** |
| Sí | Sí | No |

Ninguna recta separa los "Sí" de los "No" en esta tabla. Esta limitación (sumada a la falta de método para entrenar redes más grandes) provocó el **primer invierno de la IA**, hasta mediados de los 1980s.

### 3.3 El primer invierno de la IA y la revolución silenciosa del crédito

Mientras la IA académica se frenaba, ocurría algo notable en las finanzas: el credit scoring se convirtió en **uno de los primeros éxitos comerciales del machine learning**. En 1956, el mismo año en que McCarthy acuñaba "inteligencia artificial", Bill Fair y Earl Isaac (con $400 cada uno) fundaron Fair Isaac Corporation. Para 1958 tenían el primer sistema comercial de scoring.

**La máquina del scorecard.** Compara al solicitante contra patrones poblacionales acumulados de comportamiento histórico, estandariza la información en preguntas con respuestas tabuladas, y **suma puntos** → una probabilidad de default. La Tabla 3.1 del libro reproduce un scorecard FICO real de 1977: cada atributo (años en el trabajo, cuentas bancarias, rating del buró, etc.) tiene puntos asignados según cómo se comportaron históricamente los clientes con ese atributo.

> El libro lo dice explícitamente: **es la misma lógica de acumulación de evidencia que los decibans del cap. 1.** Cada atributo aporta puntos (evidencia incremental), se suman, y al cruzar un umbral cae la decisión.

**Ingeniería para la simplicidad.** El scorecard triunfó *porque no requería inteligencia general*: el problema estaba bien definido (predecir default con patrones históricos) y los puntos podían sumarse por analistas sin formación estadística (igual que las "girls" de Turing sumaban decibans, cap. 1).

**La divergencia.** Mientras la IA perseguía arquitecturas cada vez más complejas, el crédito se quedó deliberadamente en métodos simples e interpretables. La razón no fue técnica sino **regulatoria**: los prestamistas deben explicar sus decisiones al regulador, justificar rechazos (*adverse action*) y mantener modelos estables que no cambien bruscamente en cada recalibración. Las redes neuronales ofrecían potencia pero no esa transparencia.

### 3.4 Redes multicapa y backpropagation

La solución al XOR era conocida en teoría desde los 1960s: **agregar capas**. Una red con capas ocultas —el **multilayer perceptron (MLP)** o red feedforward profunda— aprende fronteras **no-lineales** y resuelve XOR y patrones mucho más complejos. Faltaba un problema práctico: ¿cómo entrenarla?

**Backpropagation (1986).** El algoritmo que despertó a las redes de su primer invierno. Idea: dado el error en la salida, **propagarlo hacia atrás** capa por capa para saber cuánto contribuyó cada peso, usando la **regla de la cadena**. Descubierto varias veces (Werbos 1974), se popularizó con Rumelhart, Hinton y Williams (1986). Hinton recibió el **Nobel de Física 2024** por estos fundamentos.

Mecánica en dos pasadas (para una red con una capa oculta):

```
Forward pass (predicción):
   h  = σ(W⁽¹⁾·x + b⁽¹⁾)      (activaciones ocultas)
   ŷ  = σ(w⁽²⁾·h + b⁽²⁾)      (salida)

Backward pass (gradientes, regla de la cadena):
   ∂L/∂w⁽²⁾ = ∂L/∂ŷ · ∂ŷ/∂w⁽²⁾                     (pesos de salida)
   ∂L/∂w⁽¹⁾ = ∂L/∂ŷ · ∂ŷ/∂h · ∂h/∂w⁽¹⁾            (pesos ocultos)
```

**Teorema de aproximación universal (Cybenko, Hornik).** Una red con **una sola capa oculta** puede aproximar *cualquier* función continua con la precisión que se quiera, dadas suficientes neuronas. Pero solo garantiza que esos pesos **existen**; no dice si se pueden **entrenar**, cuántas neuronas hacen falta (a veces impracticablemente muchas), ni si la red **generalizará**. Por eso en la práctica se usan redes **profundas** (varias capas): logran lo mismo con menos neuronas y descubren patrones jerárquicos.

**Red neuronal de crédito (ejemplo del libro).** Una MLP de 7 features → 10 neuronas ocultas (tanh) → 1 salida (sigmoide = PD). Inspeccionando las matrices de pesos `W₁` (input→oculta) y `W₂` (oculta→salida), se ven patrones interpretables: p. ej. la neurona H1 (peso de salida 0.80) se activa fuerte con *delinquency* y *amount past due* → actúa como "detector primario de riesgo". Y algo clave:

> **La neurona de salida hace una regresión logística sobre las activaciones ocultas:** cada unidad oculta aporta un valor, y su peso en `W₂` es el coeficiente en el cálculo final de log-odds. (Callback directo al cap. 2.)

La contracara: a más capas y neuronas, **más difícil de interpretar** — el vínculo entre feature de entrada y predicción se difumina en capas de transformaciones no-lineales. Ese es el gran costo para un entorno regulado.

### 3.5 Crédito en la era de la IA

El cambio no es solo de modelos, sino de **datos**. Del snapshot estático (una foto: DTI actual, moras recientes, utilización total) se pasó a **flujos transaccionales en tiempo real** (open banking): meses de historial granular con estructura temporal.

**(a) Cash-flow underwriting.** Evaluar solvencia analizando la **trayectoria** del comportamiento financiero, no una foto. Las arquitecturas de secuencias encajan naturalmente:
- **RNN / LSTM**: procesan datos ordenados reteniendo información de pasos previos.
- **Transformers** (hoy dominantes): usan *self-attention* para relacionar todos los elementos de la secuencia a la vez → capturan dependencias de largo alcance mejor. La Figura 3.9 del libro muestra los pesos de atención: en un caso no-default la atención se reparte sobre movimientos estables; en un default se concentra en transacciones críticas (picos/caídas de saldo).

El punto: donde un scorecard usa "saldo promedio mensual", un transformer ve **toda la trayectoria** —cuándo sube y baja el saldo, cómo se relacionan con el timing y magnitud de cada transacción—. Albanesi & Vamossy argumentan que existen relaciones no-lineales multidimensionales entre default y covariables que el scorecard lineal no puede capturar.

**(b) Underwriting agents.** Buena parte del underwriting no es juicio subjetivo sino una **secuencia de chequeos estructurados** (traer datos del buró y del banco, validar documentación, confirmar salidas del modelo, verificar cumplimiento de política). Los LLMs permiten flujos **multi-agente**: cada agente especializado en un paso, comunicándose entre sí, con un **humano que mantiene la decisión** (human-in-the-loop). Un "AI underwriter" no es un modelo monolítico sino un ensamble coordinado de agentes; no reemplaza la metodología de riesgo, la **orquesta**.

Despliegue seguro en finanzas (del libro):
- **LLMs open-weights locales** (no APIs externas): evitan riesgos de privacidad, dependencia de terceros y falta de reproducibilidad/determinismo ante el regulador.
- **RAG** (retrieval-augmented generation): fundamenta el razonamiento en documentos internos aprobados (políticas, governance) → mismas entradas, mismas salidas.
- **Tool-calling**: el agente interactúa con sistemas internos (buró, verificación de ingresos, scoring) mediante funciones auditables.

**Cierre — "Towards Credit Risk AI".** El arco completo: Turing 1948 ("máquinas no organizadas") → inviernos de la IA → backprop (1986) → transformers y agentes (hoy). La convergencia: combinar la **interpretabilidad y aceptación regulatoria** del scorecard clásico con el **reconocimiento de patrones** del deep learning. El reto ya no es *si* las redes pueden aprender patrones de crédito (pueden), sino si lo hacen cumpliendo **transparencia, fairness y estabilidad** bajo los cambios de distribución propios de los mercados de crédito.

---

## Parte 2 — Conexión con tu metodología y los capítulos anteriores

### La escalera WOE → logística → MLP

Los tres capítulos describen la misma familia, subiendo un escalón de complejidad cada vez:

| Modelo | Qué es | Captura |
|---|---|---|
| **WOE (suma directa)** | naive-Bayes, cap. 1 | evidencia univariada, features independientes |
| **WOE + logística** | tu pipeline, cap. 2 | + ponderación/redundancia entre features (lineal en log-odds) |
| **MLP (red)** | cap. 3 | + no-linealidades e **interacciones** aprendidas automáticamente |

Tu WOE-logística **es el caso lineal** (una red sin capas ocultas). La MLP agrega capas ocultas que aprenden interacciones y no-linealidades **solas**, en vez de que vos las metas a mano vía binning/agrupamiento/features cruzadas. Y todo se entrena minimizando **la misma log loss** del cap. 2 §2.2 (ahora con backprop).

### Tu scorecard ya es "puntos = evidencia aditiva"

El WOE que calculás es, salvo escala y redondeo, lo mismo que los puntos de la Tabla 3.1 del cap. 3 (ver Parte 3, Q2). Estás parado exactamente en la tradición Fair-Isaac.

### Lo que tu pipeline no capta (y una red/árbol sí)

- **Interacciones tipo XOR** (Deuda × Ingreso): tu WOE trata cada variable por separado.
- **No-monotonías dentro de un feature** (relaciones en U): tu binning monótono las aplana — es su función, pero también su límite. Ver la discusión de monotonicidad en [01_woe_cap1.md](01_woe_cap1.md) §2.3.

### Tus features son "medio-transaccionales"

Las variables de tu notebook (`sum_recarga_u6m`, `sum_fact_imp_u3m`, saldos y billeteras `_u3m`/`_u6m`) ya son intentos de comprimir **tendencia temporal** en agregados (sumas, máximos, ratios u3m-vs-u6m). El cash-flow underwriting del cap. 3 va un paso más: en vez de comprimir a mano, alimentar la **secuencia cruda** de movimientos y dejar que el modelo aprenda la forma. Es una dirección natural de evolución si algún día tenés la serie transaccional disponible.

---

## Parte 3 — Las 4 preguntas

### 3.1 ¿Deep learning vs scorecard/WOE — cuándo cada uno?

No es "mejor/peor" sino "según el problema":

| Método | Usalo cuando… |
|---|---|
| **Scorecard / WOE-logística** | prima interpretabilidad, cumplimiento regulatorio y estabilidad; hay pocos datos; features tabulares agregadas; necesitás explicar cada rechazo. |
| **Árboles / boosting (XGBoost, LightGBM)** | datos **tabulares** con no-linealidades e interacciones; querés performance sin binning manual (tu pregunta del cap. 1). |
| **Deep learning (RNN/LSTM/transformers)** | hay **estructura secuencial/temporal** rica (transacciones), texto, o muy alta dimensión. |

Frase del libro que lo resume: *"las features tabulares favorecen los métodos basados en árboles; las secuencias de transacciones se alinean naturalmente con redes neuronales."* En la práctica: **híbrido** — scorecard/boosting para el core regulado, deep learning donde la señal es secuencial y el negocio tolera menos interpretabilidad.

### 3.2 ¿Los puntos del scorecard son WOE?

**Sí, esencialmente.** Un scorecard logístico se arma escalando linealmente el log-odds a "puntos". Partiendo de `logit = β₀ + Σ βⱼ·WOEⱼ` (cap. 2), los puntos de cada atributo son:

```
puntos_j = (βⱼ · WOEⱼ) · factor + offset_j
```

donde `factor` y `offset` fijan la escala legible (el famoso **PDO**, *points to double the odds*: cuántos puntos hacen que los odds se dupliquen). La transformación completa es `Score = a + b · log-odds`, un reescalado lineal. La Tabla 3.1 (FICO 1977) son, en el fondo, **WOE multiplicados por su coeficiente y escalados a enteros amigables** — igual que Turing escalaba log-factores a decibans (`WOE·10`, redondeado, cap. 1).

Conclusión práctica: **tu WOE es el paso previo; "hacer el scorecard" es solo escalar esos WOE (ya ponderados por la logística) a puntos.** No es un método distinto.

### 3.3 ¿Por qué la MLP capta lo que mi WOE+binning no?

Tres razones:

1. **Interacciones automáticas.** Las capas ocultas **combinan** features (Deuda × Ingreso = XOR). Tu WOE trata cada variable por separado (herencia naive-Bayes, cap. 2), salvo que crees features cruzadas a mano.
2. **No-monotonía dentro del feature.** Tu binning monótono aplana relaciones en U o en campana; una red (o un árbol) las representa sin problema.
3. **Representaciones aprendidas.** La MLP hace la logística sobre **activaciones ocultas que ella misma aprende**, no sobre WOE fijos calculados de antemano. Descubre la transformación óptima en vez de recibirla dada.

El costo es el de siempre: **interpretabilidad y encaje regulatorio** (§3.4). Por eso en crédito regulado el WOE sigue vivo aunque una red pueda "ver más".

### 3.4 ¿Cash-flow underwriting me aplica?

**Concepto:** modelar la **trayectoria temporal** de las transacciones (no un snapshot) con RNN/LSTM/transformers, para que el modelo aprenda la *forma* del comportamiento (tendencia de ingresos, volatilidad, estrés financiero, señales tempranas de deterioro).

**Relevancia para vos:** tus features `_u3m`/`_u6m` (recargas, facturación, saldos, billeteras) ya son un intento de capturar tendencia, pero **comprimida en agregados** (sumas, máximos, ratios). El cash-flow underwriting es el paso siguiente: alimentar la **secuencia cruda** de movimientos por cliente y dejar que el modelo extraiga la señal temporal que tus agregados promedian y pierden.

**Requisitos que menciona el libro** (si algún día vas por ahí): feature engineering temporal (montos, saldos, cambios), muestreo balanceado 50/50 default/no-default en train, estandarización (media 0, varianza 1), batch normalization, y estructurar los datos como **secuencias ordenadas por cuenta**.

**Trade-off honesto:** más señal potencial, pero también más complejidad, más datos, y menos interpretabilidad. En un entorno regulado (y con tus datos actuales, que son agregados y no la serie cruda) suele **convivir** con el scorecard, no reemplazarlo — al menos hasta tener la infraestructura de datos transaccionales y la tolerancia regulatoria.

---

## Puentes con el resto del material

- **Con el cap. 1 ([01_woe_cap1.md](01_woe_cap1.md)):** el scorecard de puntos = decibans/WOE (evidencia aditiva); el XOR es el límite del enfoque aditivo/monótono.
- **Con el cap. 2 ([02_logistic_woe_cap2.md](02_logistic_woe_cap2.md)):** logística = red sin capas ocultas; la salida de la MLP = logística sobre activaciones. La escalera WOE → logística → MLP es la misma log loss con más capacidad.
- **Hacia adelante (cap. 4):** "The Credit Game" — machine learning y explicabilidad (SHAP), que es la respuesta del libro al problema de interpretabilidad que el cap. 3 deja abierto para redes y boosting.

## Referencias del capítulo (bibliografía del libro)

- W. McCulloch & W. Pitts, *A Logical Calculus of the Ideas Immanent in Nervous Activity* (1943).
- A. M. Turing, *Intelligent Machinery* (1948, "unorganised machines").
- F. Rosenblatt, *The Perceptron* (1958); M. Minsky & S. Papert, *Perceptrons* (1969).
- D. Rumelhart, G. Hinton & R. Williams, *Learning Representations by Back-Propagating Errors* (1986); P. Werbos (1974).
- G. Cybenko (1989), K. Hornik et al. (1989) — teorema de aproximación universal.
- M. Poon — historia del scorecard Fair Isaac (2007, 2011, 2012).
- S. Albanesi & D. Vamossy, *Predicting Consumer Default: A Deep Learning Approach* (NBER, 2019).
- I. Goodfellow, Y. Bengio & A. Courville, *Deep Learning* (2016).
