# FastWoe — *Fast Weight of Evidence Encoding and Inference*

**Estructura de la librería, catálogo completo de funcionalidades, y cómo se relaciona con `woe_example/`**

Librería: [xRiskLab/FastWoe](https://github.com/xRiskLab/FastWoe) — v0.1.8, licencia MIT, *Development Status: 4 - Beta*.
Notebook de ejemplo: [`fastwoe_capacidades.ipynb`](fastwoe_capacidades.ipynb) (61 celdas de código, ejecutado de punta a punta sobre `credit-g`).

> Nota: este documento **no traduce** la documentación oficial. Cataloga lo que la librería hace y lo contrasta, función por función, con [`woe_example/utils.py`](../woe_example/utils.py). Todas las firmas y comportamientos de acá fueron **verificados por introspección contra la versión 0.1.8 instalada**, no copiados del README oficial — que en dos puntos está desactualizado (ver §9).

Si `utils.py` responde *"¿cómo calculo el WOE y el IV de mis variables?"*, FastWoe responde **"¿cuánto puedo confiar en ese WOE, y cómo lo llevo hasta un scorecard?"**.

---

## Parte 1 — Qué es y cómo se instala

### 1.1 Propósito

FastWoe hace *encoding* WOE de variables categóricas y numéricas apoyándose en el `TargetEncoder` de scikit-learn, y le agrega la capa que casi ninguna implementación casera tiene: **inferencia estadística**. Cada WOE viene con su error estándar y su intervalo de confianza; cada IV con su SE y un veredicto de significancia.

Soporta clasificación **binaria y multiclase** (*one-vs-rest*), y expone todo con la interfaz de *transformer* de sklearn.

### 1.2 Requisitos e instalación

| | |
|---|---|
| Python | `>=3.9` (probado hasta 3.14) |
| Núcleo | `numpy>=1.21`, `pandas>=1.3`, `scipy>=1.7`, **`scikit-learn>=1.3.0,<1.8.0`**, `rich>=13.7`, `loguru>=0.7`, `numba>=0.60`, `packaging>=21` |

```bash
pip install fastwoe                 # base
pip install "fastwoe[plotting]"     # + matplotlib (necesario para plots.py)
pip install "fastwoe[faiss]"        # + faiss-cpu   (binning por KMeans)
pip install "fastwoe[faiss-gpu]"    # + faiss-gpu-cu12
pip install "fastwoe[examples]"     # + seaborn, jupyter, statsmodels, pygam
```

**El pin `scikit-learn < 1.8` es la restricción a mirar.** Tu entorno local tiene Python 3.9.13 con sklearn 1.6.1, así que entra sin conflicto. En el notebook la instalación va en una celda con la *magic* `%pip` (no `!pip`), que instala en el intérprete del kernel seleccionado.

> **Si estás detrás del proxy de Telecom.** La instalación falla con `SSLCertVerificationError` porque la inspección SSL rompe la verificación de certificados de PyPI. Se resuelve con
> `%pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org "fastwoe[plotting]"`.
> Verificado en este entorno. OpenML, en cambio, sí responde.

---

## Parte 2 — Estructura de la librería

Diez módulos. Los tres primeros son el 90 % del uso real.

| Módulo | Qué contiene |
|---|---|
| `fastwoe.py` | **`FastWoe`** y **`WoePreprocessor`** — el núcleo |
| `fastwoe_multiclass.py` | Mixin *one-vs-rest*: WOE por clase cuando el target tiene 3+ niveles |
| `fastwoe_piecewise.py` | `PiecewiseWoeMixin` — `assign_pieces()` y la salida `output="piecewise"` |
| `interpret_fastwoe.py` | **`WeightOfEvidence`** — explicabilidad caso a caso |
| `metrics.py` | `gini_contributions` — contribución firmada de cada observación al Gini |
| `fast_somersd.py` | Somers' D con Numba: `somersd_yx`, `somersd_xy`, `somersd_pairwise`, `somersd_se`, `somersd_clustered_matrix` |
| `screening.py` | `marginal_somersd_selection`, `somersd_shapley` — selección de variables |
| `plots.py` | `visualize_woe` (curva WOE / *waterfall*), `plot_performance` (curva CAP + Gini) |
| `display.py` | `StyledDataFrame`, `styled`, `iv_styled`, `style_woe_mapping`, `style_iv_analysis` |
| `logging_config.py` | Logging vía `loguru` |

Los **11 nombres públicos** (`fastwoe.__all__`):

```python
['FastWoe', 'WoePreprocessor', 'WeightOfEvidence',
 'plot_performance', 'visualize_woe',
 'StyledDataFrame', 'style_iv_analysis', 'style_woe_mapping', 'styled', 'iv_styled',
 'gini_contributions']
```

`screening` y `fast_somersd` no están en `__all__`: se importan como submódulo (`from fastwoe import screening`).

---

## Parte 3 — API completa

### 3.1 `WoePreprocessor` — reducción de cardinalidad

```python
WoePreprocessor(max_categories=None, top_p=0.95, min_count=10, other_token="__other__")
```

| Método | Devuelve |
|---|---|
| `fit(X, y=None, cat_features=None)` | `self`. `cat_features` limita la reducción a ciertas columnas |
| `transform(X)` | `DataFrame` con la cola agrupada en `other_token` |
| `fit_transform(X, y=None, **fit_params)` | ídem |
| `get_category_mapping()` | `dict[str, set]` — categorías conservadas por columna. **Los valores son `set`, no `list`** |
| `get_reduction_summary(X)` | `DataFrame`: `feature`, `original_categories`, `kept_categories`, `reduction_pct` |

### 3.2 `FastWoe` — el núcleo

```python
FastWoe(encoder_kwargs=None, random_state=42, binner_kwargs=None,
        warn_on_numerical=False, numerical_threshold=20, binning_method="tree",
        tree_estimator=None, tree_kwargs=None, faiss_kwargs=None, monotonic_cst=None)
```

| Parámetro | Efecto |
|---|---|
| `binning_method` | `"tree"` (default, supervisado), `"kbins"` (`KBinsDiscretizer`), `"faiss_kmeans"` |
| `numerical_threshold` | valores únicos a partir de los cuales una numérica se bincea (default 20) |
| `tree_kwargs` / `binner_kwargs` / `faiss_kwargs` | parámetros del binner elegido |
| `tree_estimator` | clase de árbol alternativa (p. ej. `ExtraTreeClassifier`) |
| `monotonic_cst` | `dict` `{variable: 1 | -1 | 0}`. **Solo funciona con `binning_method="tree"`** |

**Ajuste y transformación**

| Método | Devuelve |
|---|---|
| `fit(X, y)` | `self` |
| `transform(X, output="woe")` | `DataFrame`. `output` ∈ `"woe"`, `"woe_norm"`, `"wald"`, `"woe_upper_ci"`, `"woe_lower_ci"`, `"piecewise"` |
| `fit_transform(X, y=None, **_fit_params)` | `DataFrame` |
| `transform_standardized(X, output="woe", col_name=None)` | versión estandarizada |
| `finetune(X_new, y_new, update_prior=False)` | `self` — actualización incremental |

**Inspección**

| Método | Devuelve |
|---|---|
| `get_mapping(feature, class_label=None)` | `category`, `count`, `count_pct`, `good_count`, `bad_count`, `event_rate`, `woe`, `woe_se`, `woe_ci_lower`, `woe_ci_upper` |
| `get_all_mappings()` | `dict[str, DataFrame]` |
| `get_probability_mapping(feature)` | probabilidad de evento por categoría, en vez del WOE |
| `get_feature_stats(col=None, class_label=None)` | `n_categories`, `total_observations`, `missing_count`, `missing_rate`, `gini`, `somersd_se`, `somersd_ci_lower/upper`, `iv`, `iv_se`, `iv_ci_lower/upper`, `min_woe`, `max_woe` |
| `get_feature_summary()` | versión compacta: `feature`, `gini`, `iv`, `n_categories`, ordenada por Gini |
| `get_iv_analysis(col=None, class_label=None, alpha=0.05)` | `iv`, `iv_se`, `iv_ci_lower`, `iv_ci_upper`, **`iv_significance`**, `n_categories`, `gini` |
| `get_binning_summary()` | `feature`, `values`, `n_bins`, `missing`, `method`, `monotonic_constraint` |
| `get_tree_estimator(feature)` | el `DecisionTreeClassifier` que produjo los cortes |
| `get_split_value_histogram(feature, as_array=True)` | los puntos de corte, con `-inf` / `+inf` en los extremos |

**Predicción**

| Método | Devuelve |
|---|---|
| `predict_proba(X)` | `ndarray (n, n_clases)` |
| `predict(X)` | `ndarray (n,)` |
| `predict_ci(X, alpha=0.05)` | **`ndarray (n, 2)`** → `[inferior, superior]` |
| `predict_proba_class(X, class_label)` | `ndarray (n,)` — multiclase |
| `predict_ci_class(X, class_label, alpha=0.05)` | `ndarray (n, 2)` — multiclase |

**Piecewise**

| Método | Devuelve |
|---|---|
| `assign_pieces(strategy="sign", piece_map=None)` | `self`. Agrega la columna `piece` a `mappings_` |

### 3.3 `WeightOfEvidence` — explicabilidad

```python
WeightOfEvidence(classifier=None, X_train=None, y_train=None,
                 feature_names=None, class_names=None, auto_infer=True)
```

**`classifier` tiene que ser un `FastWoe`.** Si le pasás un `LogisticRegression` tira `ValueError: Only FastWoe classifiers are supported`. Si lo dejás en `None`, crea y ajusta uno solo a partir de `X_train`/`y_train`. Recibe los datos **originales**, no los ya transformados a WOE.

| Método | Devuelve |
|---|---|
| `summary()` | `str` — variables, clases, N de entrenamiento, distribución de clases |
| `explain(x, sample_idx=None, class_to_explain=None, true_labels=None, return_dict=True)` | `dict`: `predicted_label`, `predicted_proba`, `explained_label`, `total_woe`, `interpretation`, `feature_contributions` |
| `explain_ci(..., alpha=0.05)` | lo anterior + `confidence_level`, `ci_conservative`, `ci_optimistic`, `uncertainty_range` |
| `predict_ci(X, alpha=0.05, return_probabilities=False)` | `dict`: `base_estimate`, `lower_bound`, `upper_bound`, `uncertainty_summary` |

### 3.4 Gráficos, display y métricas

```python
visualize_woe(woe_encoder, feature_name=None, explanation=None,
              mode="proba"|"logit", figsize=(10, None), show_plot=True) -> pd.DataFrame
plot_performance(y_true, y_pred, weights=None, ax=None, figsize=(6,4), dpi=110,
                 show_plot=True, labels=None, colors=None, top_p=None) -> (fig, ax, gini)

style_woe_mapping(df, feature_name, theme="light") -> StyledDataFrame
style_iv_analysis(df, theme="light") -> StyledDataFrame
styled(title=None, subtitle=None, highlight_cols=None, precision=4, theme="light")  # decorador

gini_contributions(scores, labels) -> (contribuciones_por_obs, gini)   # las contribuciones suman el Gini
```

`plot_performance` acepta una **lista** de scores en `y_pred` para superponer varios modelos en la misma curva CAP y devolver una lista de Ginis.

### 3.5 Selección de variables (`screening`)

```python
marginal_somersd_selection(X, y, X_test=None, y_test=None, min_msd=0.02,
                           max_features=None, correlation_threshold=0.5, ties="y",
                           random_state=None, woe_model=None, verbose=False) -> dict
somersd_shapley(score_dict, y, availability_mask=None, base_score_name=None, ties="y") -> pd.DataFrame
```

`marginal_somersd_selection` devuelve `selected_features`, `msd_history`, `univariate_somersd`, `model`, `test_performance`, `correlation_matrix`.

Es conceptualmente distinto de ordenar por IV: en cada paso ajusta un modelo con lo ya seleccionado, calcula los **residuos**, y elige la variable que mejor correlaciona con lo que **todavía no está explicado**. Penaliza redundancia de forma natural, cosa que un ranking univariado por IV no puede hacer.

---

## Parte 4 — Fundamento matemático y la diferencia con tu `utils.py`

FastWoe define:

```
WOE(bin) = ln( P(evento | bin) / P(no evento | bin) )  −  ln( P(evento) / P(no evento) )
           └────────── log-odds del bin ──────────┘      └──── log-odds prior ────┘
```

con **evento = malo**. Tu [`calcular_metricas_bin`](../woe_example/utils.py#L53) define:

```python
# utils.py:82-86
np.log(df_grouped['porcbuenostotal'] / df_grouped['porcmalostotal'])   # ln(%buenos / %malos)
```

Contra lo que podría parecer, **las dos fórmulas son algebraicamente la misma cosa cambiada de signo**. Desarrollando la tuya:

```
ln( (bᵢ/B) / (mᵢ/M) )  =  ln(bᵢ/mᵢ) + ln(M/B)  =  −[ ln(mᵢ/bᵢ) − ln(M/B) ]
                                                   └─ log-odds bin ─┘  └ prior ┘
```

Es decir: usar `%buenos_total` y `%malos_total` (distribuciones sobre el total de buenos y el total de malos) **ya centra el WOE en la tasa poblacional**, exactamente como lo hace la resta explícita del *prior* en FastWoe. Es el mismo punto que ya marca [`02_logistic_woe_cap2.md` §2.3](../resumenes/02_logistic_woe_cap2.md).

**Verificación numérica** sobre `checking_status` de `credit-g`:

| categoría | n | `woe_utils` | `woe_fastwoe` | suma |
|---|---:|---:|---:|---:|
| `<0` | 274 | −0,8180 | +0,8181 | 0,00014 |
| `0<=X<200` | 269 | −0,4013 | +0,4014 | 0,00006 |
| `>=200` | 63 | +0,4048 | −0,4055 | −0,00071 |
| `no checking` | 394 | +1,1764 | −1,1763 | 0,00013 |

La suma da cero salvo por el `np.round(..., 2)` que aplicás a los porcentajes en [utils.py:74-79](../woe_example/utils.py#L74-L79). Y el log-odds *prior* de esta cartera (−0,8473) reproduce `woe_fastwoe` exactamente al restarlo del log-odds crudo de cada bin.

De acá salen **tres diferencias reales**, ninguna de las cuales es un error de tu lado — son convenciones distintas que hay que tener presentes al comparar números.

### 4.1 Signo: `WOE_utils = −WOE_fastwoe`

| | Fórmula | WOE positivo significa |
|---|---|---|
| **`utils.py`** (good-to-bad, Siddiqi) | `ln(%buenos / %malos)` | **menos** riesgo |
| **FastWoe** (bad-to-good) | `ln(P(malo) / P(bueno))` | **más** riesgo |

En tus tablas el bin más riesgoso tiene WOE **negativo**; en FastWoe, **positivo**. Es la misma diferencia que el libro marca contra tu código en [`01_woe_cap1.md` §2.1](../resumenes/01_woe_cap1.md) — FastWoe usa la convención del libro, no la tuya.

El notebook cierra verificando esto sobre `duration`: bin `(47.5, 57.0]`, tasa de malos 55,88 %, `woe = +1.0837` en FastWoe → `−1.0837` en tu convención.

### 4.2 Escala del IV: `IV_utils ≈ 100 × IV_fastwoe`

En [utils.py:74-79](../woe_example/utils.py#L74-L79) los porcentajes están en **puntos porcentuales**. Como `iv = (%buenos − %malos) · woe`, tu IV queda ~100× la escala convencional (de ahí los `iv_total_variable` de ~105). FastWoe trabaja en proporciones, así que sus IV se leen directamente contra los umbrales clásicos: `<0.02` inútil · `0.02–0.1` débil · `0.1–0.3` medio · `0.3–0.5` fuerte · `>0.5` sospechoso.

Ya está documentado en [`01_woe_cap1.md` §2.4](../resumenes/01_woe_cap1.md). Verificado: sobre `checking_status`, `IV_utils = 66,59` y `IV_fastwoe = 0,666013` → ratio **99,99**.

### 4.3 Redondeo

Vos redondeás los porcentajes a 2 decimales antes de calcular el WOE ([utils.py:74-79](../woe_example/utils.py#L74-L79)); FastWoe trabaja en doble precisión. Es la única fuente de discrepancia en la tabla de arriba, y es del orden de 10⁻⁴ — irrelevante para el ranking, pero explica por qué la comparación no da cero exacto.

### 4.4 Celdas vacías

`utils.py` fuerza `woe = 0` cuando un bin no tiene buenos o no tiene malos ([utils.py:82-86](../woe_example/utils.py#L82-L86)). FastWoe hereda el *smoothing* del `TargetEncoder` de sklearn, así que no necesita el caso especial: el bin degenerado se contrae hacia el prior en vez de anularse.

---

## Parte 5 — Comparación función por función

| Tu `utils.py` | Equivalente en FastWoe | Nota |
|---|---|---|
| [`clasificar_variables`](../woe_example/utils.py#L9) | automático, vía `numerical_threshold` | FastWoe no expone la lista; se ve en `get_binning_summary()` |
| [`calcular_metricas_bin`](../woe_example/utils.py#L53) | interno | se materializa en `get_mapping()` |
| [`calcular_bivariados`](../woe_example/utils.py#L96) | `fit()` + `get_mapping()` / `get_feature_stats()` / `get_iv_analysis()` | equivalencia más cercana |
| [`discretizar_variables`](../woe_example/utils.py#L238) | `binning_method` + `tree_kwargs` / `binner_kwargs` | **no** cubre `nivel_cero` ni `config_bins` |
| `percentiles_precalculados` | `fit(train)` → `transform(test)` | el congelado de cortes es implícito |
| [`calcular_woe_mapping`](../woe_example/utils.py#L509) | `get_all_mappings()` | ya calculado dentro del `fit` |
| [`aplicar_woe`](../woe_example/utils.py#L576) | `transform()` | maneja categorías no vistas; `aplicar_woe` deja `NaN` |
| [`exportar_multiples_bivariados_excel`](../woe_example/utils.py#L411) | `style_woe_mapping()` / `style_iv_analysis()` | en notebook, no en Excel |
| `metodo='quantile'`, `n_bins=5` | `binning_method="kbins"`, `binner_kwargs={"n_bins":5,"strategy":"quantile"}` | equivalente casi exacto |
| `metodo='equal_width'` | `binner_kwargs={"strategy":"uniform"}` | |

### 5.1 Lo que FastWoe te da y hoy no tenés

- **`woe_se` e intervalos de confianza por bin.** Con esto podés distinguir un bin genuinamente informativo de uno cuyo WOE es grande solo porque tiene 12 casos. Hoy en el Excel esa distinción no se puede hacer.
- **IV con SE y significancia.** `get_iv_analysis()` marca `Not Significant` cuando el IV no es distinguible de cero. Una variable con IV = 0,04 puede ser ruido puro y hoy se te ve igual que una débil pero real.
- **`monotonic_cst`.** Hoy verificás monotonicidad **a ojo**, con la escala verde-amarillo-rojo del Excel. FastWoe la **impone en el binning**, que es lo que pide cualquier validación de scorecard.
- **Binning supervisado.** El árbol elige cortes que separan el target; los quintiles ciegos no. `get_split_value_histogram()` te devuelve esos cortes como array, listos para escribirse como constantes en un `CASE WHEN`.
- **Consistencia train/test por construcción**, y con ella la eliminación del *leakage*: hoy `calcular_bivariados` corre sobre el dataframe entero traído de BigQuery, así que si después partís train/test el WOE ya vio los datos de test.
- **Interfaz sklearn.** Entra en `Pipeline` y `cross_val_score` sin adaptadores. En el notebook, la validación cruzada de 5 *folds* recalcula el WOE dentro de cada uno.
- **WOE piecewise → el paso a la logística.** Es exactamente el eslabón que tus resúmenes marcan como faltante ([`02_logistic_woe_cap2.md`](../resumenes/02_logistic_woe_cap2.md)).
- **Explicabilidad caso a caso** (`WeightOfEvidence`) y **curva CAP** (`plot_performance`).

### 5.2 Lo que tenés vos y FastWoe no cubre

- **El control declarativo de cortes de [`discretizar_variables`](../woe_example/utils.py#L238).** Tu `config_bins` con `tipo='percentiles'` + `nivel_cero`, `'dicotomica'`, `'custom'`, y sobre todo `base.quantile(cortes, interpolation='higher')` sobre los positivos estrictos, está diseñado para **replicar exactamente un `CASE WHEN` / `PERCENTILE_CONT` de BigQuery** sin partir valores iguales entre bins. FastWoe **no tiene equivalente**. Si el modelo tiene que correr en SQL, esto no lo podés tirar.
- **El bucket `nivel_cero`.** Separar `x <= 0` en su propia categoría antes de percentilar es lógica de negocio que ninguna heurística automática va a reproducir.
- **El Excel multi-hoja con escala de color** ([utils.py:411](../woe_example/utils.py#L411)). `style_woe_mapping` sirve para mirar en el notebook, pero no reemplaza un entregable que se manda por mail y se revisa fuera de Python.

---

## Parte 6 — Rutas de adopción

### (a) Reemplazo total

Tirar `utils.py` y usar FastWoe end-to-end. **No lo recomiendo**: perdés la replicabilidad en BigQuery, que es justamente lo que hace que tu binning sea deployable.

### (b) Híbrido — **la que recomiendo**

Seguís binneando con `discretizar_variables` para mantener la paridad con SQL, y le pasás a `FastWoe` las columnas **ya binneadas, como categóricas**. FastWoe las trata como cualquier categórica (no las re-bincea, porque tienen pocos niveles únicos) y te suma encima toda la capa estadística.

```python
from fastwoe import FastWoe

# 1) Tu binning de siempre, el que se traduce a CASE WHEN
df_bin, percentiles = discretizar_variables(df_train, config_bins)
df_bin_test, _ = discretizar_variables(
    df_test, config_bins,
    calcular_percentiles=False, percentiles_precalculados=percentiles,
)

cols_bin = [c for c in df_bin.columns if c.endswith("_perc")]

# 2) FastWoe encima de los bins ya congelados
woe = FastWoe(random_state=42)
woe.fit(df_bin[cols_bin].astype(str), df_bin["target"])

woe.get_iv_analysis()          # IV con SE y significancia sobre TUS bins
woe.get_mapping("edad_perc")   # WOE + woe_se + IC por bin
X_test_woe = woe.transform(df_bin_test[cols_bin].astype(str))
```

Te quedás con lo mejor de los dos lados: cortes replicables en SQL **más** errores estándar, significancia del IV, y el camino directo a `assign_pieces()` + logística.

> **Cuidado con el dtype.** Pasá las columnas binneadas como `str`, no como `category`. En 0.1.8, `predict_ci()` escribe *floats* sobre las columnas de entrada y pandas lo rechaza en una columna `category` con `TypeError: Cannot setitem on a Categorical with a new category`. `fit`, `transform` y `predict_proba` no se ven afectados: **solo rompe `predict_ci`**. Está documentado en la celda de carga del notebook.

### (c) FastWoe como validador de `utils.py`

Correr ambos caminos sobre el mismo dataset y verificar que `WOE_utils ≈ −WOE_fastwoe` y `IV_utils ≈ 100 × IV_fastwoe`. Es un test de regresión barato para `utils.py`: si algún día alguien toca `calcular_metricas_bin`, la comparación lo detecta.

---

## Parte 7 — Puentes con los resúmenes del libro

- **[`01_woe_cap1.md`](../resumenes/01_woe_cap1.md)** — el cap. 1 define el WOE con la convención *bad-to-good*, que es **la misma que usa FastWoe**. El §2.1 de ese resumen ya anticipa la inversión de signo respecto de tu código; acá queda confirmada contra una librería de referencia. El §2.4 anticipa la escala ×100 del IV; ídem.
- **[`02_logistic_woe_cap2.md`](../resumenes/02_logistic_woe_cap2.md)** — el cap. 2 trata el paso de WOE a regresión logística, y el resumen marca (líneas 149-151) que ese paso **no está implementado** en `woe_example`. `assign_pieces()` + `transform(output="piecewise")` + `LogisticRegression(penalty=None)` es exactamente esa implementación, en la variante de Raymond Anderson: un coeficiente por *pieza* en vez de uno por variable (WOE clásico) o uno por bin (dummies).
- **Monotonicidad** — el §2.3 del cap. 1 justifica teóricamente por qué exigís monotonicidad: la logística es lineal en log-odds y el WOE lineariza la relación. `monotonic_cst` convierte ese requisito en una restricción del binning en vez de una inspección visual.

---

## Parte 8 — Riesgos y limitaciones

- **Versión 0.1.8, *Development Status: 4 - Beta*.** El API puede moverse entre versiones menores. Si lo llevás a producción, pineá la versión exacta.
- **Pin `scikit-learn < 1.8`.** Puede chocar con otras dependencias del entorno.
- **`numba>=0.60` es dependencia del núcleo**, no opcional. En entornos corporativos restringidos puede ser un problema de instalación.
- **`monotonic_cst` solo funciona con `binning_method="tree"`.** Con `"kbins"` se ignora en silencio.
- **`predict_ci()` rompe con columnas de dtype `category`** (ver §6b).
- **No reemplaza `discretizar_variables`** si necesitás paridad exacta con SQL.
- **WOE extremos con bins degenerados.** Si un bin queda sin eventos, el WOE puede irse a valores enormes (en pruebas sintéticas vi `min_woe = −12,9`). Mirá siempre `min_woe`/`max_woe` en `get_feature_stats()` antes de dar el encoding por bueno.

---

## Parte 9 — Correcciones a la documentación oficial

Dos cosas que el README del repo dice y **no se cumplen en 0.1.8** (verificado por introspección):

| El README oficial dice | Lo que realmente pasa |
|---|---|
| `ci_results[['prediction', 'lower_ci', 'upper_ci']]` — como si `predict_ci` devolviera un `DataFrame` | Devuelve un **`np.ndarray` de shape `(n, 2)`**, sin nombres de columna |
| Los ejemplos de `WeightOfEvidence` sugieren que acepta cualquier clasificador | Exige un **`FastWoe`**; con un `LogisticRegression` tira `ValueError: Only FastWoe classifiers are supported` |

---

## Referencias

- Repositorio: <https://github.com/xRiskLab/FastWoe> — MIT, xRiskLab (`contact@xrisklab.ai`)
- Guías del repo: `docs/woe_standard_errors.md`, `docs/piecewise_woe_guide.md`, `docs/multiclass_woe_guide.md`, `docs/somersd_ase.md`, `docs/marginal_somersd_guide.md`
- Notebooks oficiales: `examples/notebooks/` — `fastwoe_monotonic`, `fastwoe_piecewise`, `fastwoe_multiclass`, `woe_standard_errors`, `msd_feature_selection`, `fastwoe_cap_curve`, entre otros
- Anderson, R., *Credit Intelligence & Modelling* (2022) — origen del enfoque *piecewise* WOE
- Siddiqi, N., *Credit Risk Scorecards* (2006) — la convención good-to-bad que usa tu `utils.py`
- Dataset del notebook: [OpenML `credit-g`](https://www.openml.org/d/31) (German Credit, 1.000 casos, 20 variables)

---

> **Nota sobre los links.** `woe_example/` está en `.gitignore` (`/on_credit/woe_example`), así que los enlaces a `../woe_example/utils.py#L…` funcionan en local pero quedan muertos en GitHub. Es el mismo comportamiento que ya tienen los links de `resumenes/*.md`.
