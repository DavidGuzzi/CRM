"""
pipeline_woe — Discretización monótona automática, WOE y generación de SQL para BigQuery.

Dado un DataFrame con `cuil_cuit`, `target` y N variables candidatas, este módulo:

  1. Busca automáticamente una discretización con relación **monótona** contra el target
     (árbol de decisión con restricción de monotonicidad, vía FastWoe).
  2. Descarta las variables que no logran monotonicidad, o cuyo IV es débil o no
     estadísticamente significativo.
  3. Calcula el WOE de las que sobreviven.
  4. Emite dos queries de BigQuery: una de discretización y otra de asignación de WOE.

CONVENCIÓN DE SIGNO
-------------------
El WOE que produce este módulo es:

    WOE(bin) = ln( P(malo | bin) / P(bueno | bin) ) - ln( P(malo) / P(bueno) )
             = ln( (malos_bin / malos_total) / (buenos_bin / buenos_total) )

es decir la convención de FastWoe y del libro *On Credit*: **WOE positivo = MÁS riesgo**.

Es el signo **opuesto** al de `woe_example/utils.py:82-86`, que usa `ln(%buenos/%malos)`
(Siddiqi, WOE positivo = menos riesgo). Los valores de este módulo NO son comparables
contra las tablas bivariadas históricas sin invertir el signo.

El IV está en escala de proporciones (umbrales clásicos 0.02 / 0.1 / 0.3 / 0.5), no en la
escala x100 de `utils.py`.

Requiere: fastwoe>=0.1.8, pandas, numpy, scipy, scikit-learn.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from fastwoe import FastWoe, WoePreprocessor

__all__ = [
    "detectar_direccion",
    "buscar_binning_monotono",
    "ajustar_pipeline",
    "resumen_seleccion",
    "aplicar_pipeline",
    "generar_sql_discretizacion",
    "generar_sql_woe",
    "guardar_spec",
    "cargar_spec",
]

# Etiquetas reservadas de bin
BIN_NULO = "nulo"
BIN_OTROS = "otros"

# Suavizado de Laplace para bins degenerados (sin buenos o sin malos).
# utils.py resuelve ese caso forzando woe=0; acá se contrae hacia el prior, que es
# el tratamiento estándar y evita infinitos sin perder la dirección del bin.
_ALPHA_SUAVIZADO = 0.5


# ---------------------------------------------------------------------------
# 1. Dirección y búsqueda de binning monótono
# ---------------------------------------------------------------------------

def detectar_direccion(x: pd.Series, y: pd.Series) -> int:
    """
    Detecta el sentido de la relación entre una variable y el target.

    Devuelve +1 si a mayor `x` mayor riesgo, -1 si a mayor `x` menor riesgo.
    Se calcula con la correlación de Spearman sobre los casos no nulos, así que
    solo mira el orden: no asume linealidad.

    Ante empate exacto (o correlación nula) devuelve +1 por convención.
    """
    mask = x.notna()
    if mask.sum() < 2 or x[mask].nunique() < 2:
        return 1

    rho, _ = spearmanr(x[mask], y[mask])
    if np.isnan(rho) or rho == 0:
        return 1
    return 1 if rho > 0 else -1


def _tabla_woe(bins: pd.Series, y: pd.Series) -> pd.DataFrame:
    """
    Calcula la tabla WOE/IV por bin, en la convención de FastWoe (positivo = más riesgo).

    Se calcula acá y no se toma de `FastWoe.get_mapping()` a propósito: así el WOE que
    va al `spec`, el que aplica `aplicar_pipeline` y el que se emite en el SQL salen
    todos de la misma fórmula, sin depender del suavizado interno del TargetEncoder.
    """
    df = pd.DataFrame({"bin": bins.astype(str), "y": y.to_numpy()})
    agg = df.groupby("bin", dropna=False)["y"].agg(n="size", malos="sum").reset_index()
    agg["buenos"] = agg["n"] - agg["malos"]

    malos_total = int(agg["malos"].sum())
    buenos_total = int(agg["buenos"].sum())
    if malos_total == 0 or buenos_total == 0:
        raise ValueError("El target no tiene ambas clases; no se puede calcular WOE.")

    a = _ALPHA_SUAVIZADO
    k = len(agg)
    p_malos = (agg["malos"] + a) / (malos_total + a * k)
    p_buenos = (agg["buenos"] + a) / (buenos_total + a * k)

    agg["event_rate"] = agg["malos"] / agg["n"]
    agg["pct"] = agg["n"] / agg["n"].sum()
    agg["woe"] = np.log(p_malos / p_buenos)
    agg["iv"] = (p_malos - p_buenos) * agg["woe"]
    return agg


def _es_monotona(valores: Sequence[float], tolerancia: float = 1e-9) -> bool:
    """True si la secuencia es monótona (creciente o decreciente), con tolerancia."""
    d = np.diff(np.asarray(valores, dtype=float))
    if len(d) == 0:
        return True
    return bool((d >= -tolerancia).all() or (d <= tolerancia).all())


def _etiquetas_numericas(n_bins: int) -> List[str]:
    """bin_01, bin_02, ... cero-padded para que ordenen bien como string."""
    return [f"bin_{i:02d}" for i in range(1, n_bins + 1)]


def _asignar_bins_numericos(x: pd.Series, cortes: Sequence[float]) -> pd.Series:
    """
    Asigna etiquetas de bin a partir de los cortes interiores.

    Regla: `bin_k` es el primer k tal que `x <= cortes[k-1]`; si no hay ninguno,
    cae en el último bin. Los nulos van a BIN_NULO. Es exactamente la semántica de
    la cadena de `WHEN x <= corte THEN ...` del SQL generado.
    """
    etiquetas = _etiquetas_numericas(len(cortes) + 1)
    # np.searchsorted con side="left" devuelve el índice del primer corte >= x,
    # que es justo el "primer corte tal que x <= corte".
    idx = np.searchsorted(np.asarray(cortes, dtype=float), x.to_numpy(dtype=float), side="left")
    out = pd.Series([etiquetas[i] for i in idx], index=x.index, dtype=object)
    out[x.isna()] = BIN_NULO
    return out


def buscar_binning_monotono(
    x: pd.Series,
    y: pd.Series,
    *,
    nombre: str = "var",
    max_depth_grid: Sequence[int] = (4, 3, 2),
    min_pct_bin: float = 0.05,
    min_bins: int = 2,
    random_state: int = 42,
) -> dict:
    """
    Busca la discretización más fina que mantiene una relación monótona con el target.

    Estrategia: se detecta la dirección con Spearman, se ajusta un árbol con esa
    restricción de monotonicidad (`monotonic_cst` de FastWoe) probando profundidades
    de mayor a menor, y se acepta la primera que produce bins válidos. Se prefiere la
    más profunda porque da más IV.

    Un bin es válido si tiene al menos `min_pct_bin` de la población. Los nulos se
    tratan aparte, en su propio bin, y no participan de la búsqueda de cortes.

    Devuelve un dict con:
        ok          : bool — si se encontró un binning aceptable
        motivo      : str  — por qué se aceptó o se rechazó
        direccion   : int  — +1 / -1
        cortes      : list[float] — cortes interiores, ascendentes
        tabla       : DataFrame con bin, n, pct, event_rate, woe, iv
        iv          : float
        n_bins      : int
        max_depth   : int | None
        pct_nulos   : float
    """
    resultado = {
        "ok": False,
        "motivo": "",
        "direccion": None,
        "cortes": [],
        "tabla": None,
        "iv": np.nan,
        "n_bins": 0,
        "max_depth": None,
        "pct_nulos": float(x.isna().mean()),
    }

    mask = x.notna()
    n_validos = int(mask.sum())

    if n_validos == 0:
        resultado["motivo"] = "todos los valores son nulos"
        return resultado
    if x[mask].nunique() < 2:
        resultado["motivo"] = "sin variabilidad (un solo valor observado)"
        return resultado
    if y.nunique() < 2:
        resultado["motivo"] = "el target no tiene ambas clases"
        return resultado

    direccion = detectar_direccion(x, y)
    resultado["direccion"] = direccion

    x_val = x[mask]
    y_val = y[mask]
    min_leaf = max(20, int(min_pct_bin * n_validos))

    for depth in max_depth_grid:
        try:
            enc = FastWoe(
                binning_method="tree",
                monotonic_cst={nombre: direccion},
                tree_kwargs={"max_depth": depth, "min_samples_leaf": min_leaf},
                numerical_threshold=2,   # forzar binning aunque haya pocos valores únicos
                random_state=random_state,
            )
            enc.fit(x_val.to_frame(nombre), y_val)
            crudos = np.asarray(enc.get_split_value_histogram(nombre), dtype=float)
        except Exception as e:  # noqa: BLE001 — cualquier fallo del árbol descarta esa profundidad
            resultado["motivo"] = f"el binning falló: {type(e).__name__}"
            continue

        cortes = [float(c) for c in crudos if np.isfinite(c)]
        if len(cortes) + 1 < min_bins:
            resultado["motivo"] = f"el árbol no encontró cortes (max_depth={depth})"
            continue

        bins = _asignar_bins_numericos(x, cortes)
        tabla = _tabla_woe(bins, y)

        # Los nulos no entran en el chequeo de monotonicidad: no tienen posición en el orden.
        tabla_ord = tabla[tabla["bin"] != BIN_NULO].sort_values("bin")

        if (tabla_ord["pct"] < min_pct_bin).any():
            resultado["motivo"] = (
                f"algún bin queda por debajo de {min_pct_bin:.0%} de la población "
                f"(max_depth={depth})"
            )
            continue

        if not _es_monotona(tabla_ord["woe"].tolist()):
            resultado["motivo"] = f"el WOE no resultó monótono (max_depth={depth})"
            continue

        resultado.update(
            ok=True,
            motivo=f"monótona con {len(cortes) + 1} bins (max_depth={depth})",
            cortes=cortes,
            tabla=tabla,
            iv=float(tabla["iv"].sum()),
            n_bins=int(tabla_ord.shape[0]),
            max_depth=depth,
        )
        return resultado

    if not resultado["motivo"]:
        resultado["motivo"] = "no se encontró binning monótono en la grilla de profundidades"
    return resultado


# ---------------------------------------------------------------------------
# 2. Significancia del IV
# ---------------------------------------------------------------------------

def _stats_iv(bins: pd.Series, y: pd.Series, nombre: str, alpha: float = 0.05) -> dict:
    """
    Error estándar y significancia del IV, calculados por FastWoe sobre los bins finales.

    Se ajusta un FastWoe sobre la columna ya discretizada tratada como categórica, de
    modo que `iv_se` e `iv_significance` corresponden exactamente a los bins que van
    a terminar en el SQL.
    """
    vacio = {"iv_se": np.nan, "iv_ci_lower": np.nan, "iv_ci_upper": np.nan,
             "significancia": "desconocida"}
    try:
        enc = FastWoe(random_state=42)
        enc.fit(bins.astype(str).to_frame(nombre), y)
        fila = enc.get_iv_analysis(nombre, alpha=alpha).iloc[0]
        return {
            "iv_se": float(fila["iv_se"]),
            "iv_ci_lower": float(fila["iv_ci_lower"]),
            "iv_ci_upper": float(fila["iv_ci_upper"]),
            "significancia": str(fila["iv_significance"]),
        }
    except Exception:  # noqa: BLE001
        return vacio


# ---------------------------------------------------------------------------
# 3. Pipeline completo
# ---------------------------------------------------------------------------

def _clasificar(
    df: pd.DataFrame,
    excluir: Sequence[str],
    umbral_categorica: int,
    forzar_numericas: Sequence[str] = (),
    forzar_categoricas: Sequence[str] = (),
) -> Dict[str, List[str]]:
    """
    Separa columnas en numéricas y categóricas.

    Mismo criterio que `woe_example/utils.py:clasificar_variables`: una numérica con
    menos de `umbral_categorica` valores únicos se trata como categórica.

    Ese umbral manda a la rama categórica a las variables **ordinales** de pocos niveles
    (una cuota de 1 a 4, una antigüedad en tramos), que no reciben entonces la exigencia
    de monotonicidad. Si el orden de esos niveles es informativo, conviene listarlas en
    `forzar_numericas` para que pasen por el árbol monótono.
    """
    forzar_numericas = set(forzar_numericas)
    forzar_categoricas = set(forzar_categoricas)

    numericas, categoricas = [], []
    for col in df.columns:
        if col in excluir:
            continue
        if col in forzar_categoricas:
            categoricas.append(col)
        elif col in forzar_numericas:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(
                    f"'{col}' está en forzar_numericas pero su dtype no es numérico."
                )
            numericas.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique(dropna=True) >= umbral_categorica:
                numericas.append(col)
            else:
                categoricas.append(col)
        else:
            categoricas.append(col)
    return {"numericas": numericas, "categoricas": categoricas}


def _binning_categorico(
    x: pd.Series, y: pd.Series, *, nombre: str, max_categorias: int,
    min_count_categoria: int, min_pct_bin: float,
) -> dict:
    """
    Discretiza una categórica plegando la cola rara en BIN_OTROS.

    No se le exige monotonicidad: una categórica no tiene orden natural, así que el
    criterio de conservación es solo IV + significancia.
    """
    resultado = {
        "ok": False, "motivo": "", "direccion": None, "grupos": {},
        "tabla": None, "iv": np.nan, "n_bins": 0,
        "pct_nulos": float(x.isna().mean()),
    }

    if x.notna().sum() == 0:
        resultado["motivo"] = "todos los valores son nulos"
        return resultado
    if y.nunique() < 2:
        resultado["motivo"] = "el target no tiene ambas clases"
        return resultado

    s = x.astype(object).where(x.notna(), None)
    conteo = s.dropna().astype(str).value_counts()

    if len(conteo) < 2:
        resultado["motivo"] = "sin variabilidad (una sola categoría observada)"
        return resultado

    piso = max(min_count_categoria, int(min_pct_bin * len(s)))
    conservadas = [c for c in conteo.index[:max_categorias] if conteo[c] >= piso]
    if not conservadas:
        resultado["motivo"] = (
            f"ninguna categoría alcanza el mínimo de {piso} casos"
        )
        return resultado

    # grupos: etiqueta de bin -> lista de valores crudos
    grupos = {f"cat_{i:02d}": [c] for i, c in enumerate(sorted(conservadas), start=1)}

    def _asignar(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return BIN_NULO
        v = str(v)
        for etiqueta, valores in grupos.items():
            if v in valores:
                return etiqueta
        return BIN_OTROS

    bins = s.map(_asignar)
    tabla = _tabla_woe(bins, y)

    resultado.update(
        ok=True,
        motivo=f"{len(grupos)} categorías conservadas + '{BIN_OTROS}'",
        grupos=grupos,
        tabla=tabla,
        iv=float(tabla["iv"].sum()),
        n_bins=int(tabla.shape[0]),
    )
    return resultado


def ajustar_pipeline(
    df: pd.DataFrame,
    *,
    target_col: str = "target",
    id_col: str = "cuil_cuit",
    excluir_cols: Optional[Sequence[str]] = None,
    umbral_categorica: int = 10,
    forzar_numericas: Optional[Sequence[str]] = None,
    forzar_categoricas: Optional[Sequence[str]] = None,
    iv_min: float = 0.02,
    iv_max: float = 0.5,
    exigir_significancia: bool = True,
    max_categorias: int = 20,
    min_count_categoria: int = 50,
    min_pct_bin: float = 0.05,
    max_depth_grid: Sequence[int] = (4, 3, 2),
    random_state: int = 42,
) -> dict:
    """
    Ajusta el pipeline completo sobre un DataFrame de desarrollo.

    Para cada variable candidata busca una discretización (monótona si es numérica),
    calcula WOE e IV, y decide si se conserva o se descarta.

    Criterios de conservación:
      - numéricas: relación monótona + IV >= iv_min + (opcional) IV significativo
      - categóricas: IV >= iv_min + (opcional) IV significativo

    Una numérica con menos de `umbral_categorica` valores únicos se trata como
    categórica y por lo tanto NO se le exige monotonicidad. Si es una ordinal cuyo
    orden importa (una cuota de 1 a 4, un tramo de antigüedad), pasala por
    `forzar_numericas` para que vaya al árbol monótono.

    `iv_max` NO descarta: marca la variable como sospechosa de *leakage* en el motivo
    y deja la decisión en manos del analista.

    Devuelve el `spec`: un dict serializable a JSON que es el artefacto central del
    modelo. De él se derivan `aplicar_pipeline` y las dos queries de SQL.
    """
    if target_col not in df.columns:
        raise ValueError(f"No existe la columna target '{target_col}'.")

    y = df[target_col]
    if y.isna().any():
        raise ValueError(f"El target '{target_col}' tiene nulos.")
    if y.nunique() != 2:
        raise ValueError(f"El target '{target_col}' debe ser binario (0/1).")

    excluir = set(excluir_cols or []) | {target_col, id_col}
    tipos = _clasificar(
        df, excluir, umbral_categorica,
        forzar_numericas=forzar_numericas or (),
        forzar_categoricas=forzar_categoricas or (),
    )

    variables: Dict[str, dict] = {}
    auditoria: List[dict] = []

    for tipo, columnas in (("numerica", tipos["numericas"]),
                           ("categorica", tipos["categoricas"])):
        for col in columnas:
            if tipo == "numerica":
                r = buscar_binning_monotono(
                    df[col], y, nombre=col, max_depth_grid=max_depth_grid,
                    min_pct_bin=min_pct_bin, random_state=random_state,
                )
            else:
                r = _binning_categorico(
                    df[col], y, nombre=col, max_categorias=max_categorias,
                    min_count_categoria=min_count_categoria, min_pct_bin=min_pct_bin,
                )

            fila = {
                "variable": col,
                "tipo": tipo,
                "n_bins": r["n_bins"],
                "direccion": r["direccion"],
                "pct_nulos": r["pct_nulos"],
                "iv": r["iv"],
                "iv_se": np.nan,
                "iv_ci_lower": np.nan,
                "iv_ci_upper": np.nan,
                "significancia": "desconocida",
                "monotona": bool(r["ok"]) if tipo == "numerica" else None,
                "decision": "descartada",
                "motivo": r["motivo"],
            }

            if not r["ok"]:
                auditoria.append(fila)
                continue

            bins = (
                _asignar_bins_numericos(df[col], r["cortes"]) if tipo == "numerica"
                else _asignar_categoricos(df[col], r["grupos"])
            )
            fila.update(_stats_iv(bins, y, col))

            motivos_descarte = []
            if r["iv"] < iv_min:
                motivos_descarte.append(f"IV {r['iv']:.4f} < {iv_min}")
            if exigir_significancia and fila["significancia"] == "Not Significant":
                motivos_descarte.append("IV no significativo")

            if motivos_descarte:
                fila["motivo"] = "; ".join(motivos_descarte)
                auditoria.append(fila)
                continue

            fila["decision"] = "conservada"
            if r["iv"] > iv_max:
                fila["motivo"] = f"{r['motivo']} — OJO: IV {r['iv']:.4f} > {iv_max}, revisar leakage"

            entrada = {
                "tipo": tipo,
                "iv": float(r["iv"]),
                "n_bins": int(r["n_bins"]),
                "pct_nulos": float(r["pct_nulos"]),
                "bins": [
                    {
                        "label": str(t.bin),
                        "woe": float(t.woe),
                        "iv": float(t.iv),
                        "n": int(t.n),
                        "event_rate": float(t.event_rate),
                    }
                    for t in r["tabla"].itertuples()
                ],
            }
            if tipo == "numerica":
                entrada["direccion"] = int(r["direccion"])
                entrada["cortes"] = [float(c) for c in r["cortes"]]
                entrada["max_depth"] = int(r["max_depth"])
            else:
                entrada["grupos"] = {k: list(v) for k, v in r["grupos"].items()}

            variables[col] = entrada
            auditoria.append(fila)

    return {
        "variables": variables,
        "auditoria": auditoria,
        "meta": {
            "generado": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_desarrollo": int(len(df)),
            "tasa_base": float(y.mean()),
            "target_col": target_col,
            "id_col": id_col,
            "convencion_woe": "ln(P(malo)/P(bueno)) centrado — WOE positivo = MÁS riesgo",
            "escala_iv": "proporciones (umbrales 0.02 / 0.1 / 0.3 / 0.5)",
            "parametros": {
                "iv_min": iv_min, "iv_max": iv_max,
                "exigir_significancia": exigir_significancia,
                "min_pct_bin": min_pct_bin,
                "umbral_categorica": umbral_categorica,
                "max_categorias": max_categorias,
                "min_count_categoria": min_count_categoria,
                "max_depth_grid": list(max_depth_grid),
            },
            "n_candidatas": len(auditoria),
            "n_conservadas": len(variables),
        },
    }


def _asignar_categoricos(x: pd.Series, grupos: Dict[str, List[str]]) -> pd.Series:
    """Asigna etiquetas de bin a una categórica según el mapa `grupos`."""
    lookup = {v: etiqueta for etiqueta, valores in grupos.items() for v in valores}
    out = x.astype(object).map(lambda v: lookup.get(str(v), BIN_OTROS) if pd.notna(v) else BIN_NULO)
    return pd.Series(out, index=x.index, dtype=object)


def resumen_seleccion(spec: dict) -> pd.DataFrame:
    """
    Tabla de auditoría: una fila por variable candidata, con la decisión y su motivo.

    Es el reemplazo del Excel de bivariados: lo que se revisa antes de aceptar el modelo.
    """
    cols = ["variable", "tipo", "decision", "n_bins", "direccion", "pct_nulos",
            "iv", "iv_se", "iv_ci_lower", "iv_ci_upper", "significancia",
            "monotona", "motivo"]
    out = pd.DataFrame(spec["auditoria"], columns=cols)
    return out.sort_values(
        ["decision", "iv"], ascending=[True, False], na_position="last"
    ).reset_index(drop=True)


def aplicar_pipeline(
    df: pd.DataFrame, spec: dict, *, sufijo_bin: str = "_bin", sufijo_woe: str = "_woe",
) -> pd.DataFrame:
    """
    Aplica el `spec` a un DataFrame nuevo, creando las columnas `{var}_bin` y `{var}_woe`.

    Equivalente de `woe_example/utils.py:aplicar_woe`, pero sin dejar NaN: las categorías
    no vistas caen en 'otros' y los nulos en 'nulo', ambos con su propio WOE.

    Es también la referencia contra la cual se valida el SQL generado.
    """
    out = df.copy()
    for var, cfg in spec["variables"].items():
        if var not in out.columns:
            print(f"Warning: la variable '{var}' no está en el DataFrame; se saltea.")
            continue

        bins = (
            _asignar_bins_numericos(out[var], cfg["cortes"]) if cfg["tipo"] == "numerica"
            else _asignar_categoricos(out[var], cfg["grupos"])
        )
        mapa = {b["label"]: b["woe"] for b in cfg["bins"]}
        # Un bin que no se vio en desarrollo (p.ej. 'nulo' inexistente en train) recibe 0,
        # que es el WOE neutro: no mueve el score.
        out[f"{var}{sufijo_bin}"] = bins
        out[f"{var}{sufijo_woe}"] = bins.map(mapa).astype(float).fillna(0.0)
    return out


# ---------------------------------------------------------------------------
# 4. Generación de SQL (BigQuery / GoogleSQL)
# ---------------------------------------------------------------------------

def _ident(nombre: str) -> str:
    """Identificador entre backticks, escapando los backticks internos."""
    return "`" + nombre.replace("`", "\\`") + "`"


def _literal_str(valor: str) -> str:
    """Literal de texto en SQL, con las comillas simples escapadas."""
    return "'" + str(valor).replace("\\", "\\\\").replace("'", "\\'") + "'"


def _num(valor: float) -> str:
    """
    Número con precisión completa (repr), no redondeado.

    Redondear un corte puede mover un caso de bin, y redondear un WOE desalinea el
    score contra el calculado en Python.
    """
    return repr(float(valor))


def _cabecera(spec: dict, titulo: str) -> str:
    m = spec["meta"]
    descartadas = [f["variable"] for f in spec["auditoria"] if f["decision"] == "descartada"]
    lineas = [
        f"-- {titulo}",
        f"-- Generado por pipeline_woe el {m['generado']}",
        f"-- Desarrollo: {m['n_desarrollo']:,} filas | tasa base: {m['tasa_base']:.4%}",
        f"-- Convención WOE: {m['convencion_woe']}",
        f"-- Variables conservadas: {m['n_conservadas']} de {m['n_candidatas']} candidatas",
    ]
    if descartadas:
        lineas.append(f"-- Descartadas: {', '.join(descartadas)}")
    return "\n".join(lineas)


def _case_discretizacion(var: str, cfg: dict) -> str:
    """Construye el bloque CASE que discretiza una variable."""
    col = _ident(var)
    ramas = [f"    WHEN {col} IS NULL THEN {_literal_str(BIN_NULO)}"]

    if cfg["tipo"] == "numerica":
        etiquetas = _etiquetas_numericas(len(cfg["cortes"]) + 1)
        # El orden ascendente de los WHEN es lo que hace correcta la cadena:
        # el primer match gana, igual que en _asignar_bins_numericos.
        for etiqueta, corte in zip(etiquetas, cfg["cortes"]):
            ramas.append(f"    WHEN {col} <= {_num(corte)} THEN {_literal_str(etiqueta)}")
        ramas.append(f"    ELSE {_literal_str(etiquetas[-1])}")
    else:
        for etiqueta, valores in cfg["grupos"].items():
            lista = ", ".join(_literal_str(v) for v in valores)
            ramas.append(f"    WHEN CAST({col} AS STRING) IN ({lista}) THEN {_literal_str(etiqueta)}")
        ramas.append(f"    ELSE {_literal_str(BIN_OTROS)}")

    cuerpo = "\n".join(ramas)
    return f"  CASE\n{cuerpo}\n  END AS {_ident(var + '_bin')}"


def generar_sql_discretizacion(
    spec: dict,
    tabla_origen: str,
    *,
    id_col: Optional[str] = None,
    target_col: Optional[str] = None,
    cols_extra: Optional[Sequence[str]] = None,
    incluir_cabecera: bool = True,
) -> str:
    """
    Genera la query de discretización: una columna `{var}_bin` por variable conservada.

    `tabla_origen` se escribe tal cual entre backticks, así que se espera el formato
    completo `proyecto.dataset.tabla`.

    La rama `IS NULL` va SIEMPRE primero: en SQL `NULL <= 7.5` evalúa a NULL, no a
    FALSE, así que sin esa rama los nulos caerían silenciosamente en el ELSE.
    """
    if not spec["variables"]:
        raise ValueError("El spec no tiene variables conservadas; no hay SQL para generar.")

    id_col = id_col or spec["meta"]["id_col"]
    target_col = target_col or spec["meta"]["target_col"]

    select = [f"  {_ident(c)}" for c in [id_col, target_col, *(cols_extra or [])] if c]
    select += [_case_discretizacion(var, cfg) for var, cfg in spec["variables"].items()]

    sql = "SELECT\n" + ",\n".join(select) + f"\nFROM {_ident(tabla_origen)}"
    if incluir_cabecera:
        sql = _cabecera(spec, "Paso 1 — discretización de variables") + "\n" + sql
    return sql


def _case_woe(var: str, cfg: dict) -> str:
    """Construye el bloque CASE que mapea etiqueta de bin -> valor de WOE."""
    col = _ident(var + "_bin")
    ramas = [
        f"    WHEN {_literal_str(b['label'])} THEN {_num(b['woe'])}"
        for b in sorted(cfg["bins"], key=lambda b: b["label"])
    ]
    cuerpo = "\n".join(ramas)
    return f"  CASE {col}\n{cuerpo}\n    ELSE 0.0\n  END AS {_ident(var + '_woe')}"


def generar_sql_woe(
    spec: dict,
    tabla_origen: Optional[str] = None,
    *,
    sql_discretizacion: Optional[str] = None,
    id_col: Optional[str] = None,
    target_col: Optional[str] = None,
    incluir_score: bool = True,
    incluir_bins: bool = False,
    incluir_cabecera: bool = True,
) -> str:
    """
    Genera la query de asignación de WOE, montada sobre la de discretización como CTE.

    Se le pasa `tabla_origen` (y arma la discretización internamente) o directamente un
    `sql_discretizacion` ya generado.

    Con `incluir_score=True` agrega la columna `score_woe`, que suma todos los `_woe`:
    el score aditivo en log-odds. Para pasar a probabilidad hay que sumarle el log-odds
    base y aplicar la sigmoide, o —mejor— ajustar una logística sobre estas columnas.
    """
    if sql_discretizacion is None:
        if tabla_origen is None:
            raise ValueError("Pasá `tabla_origen` o `sql_discretizacion`.")
        sql_discretizacion = generar_sql_discretizacion(
            spec, tabla_origen, id_col=id_col, target_col=target_col,
            incluir_cabecera=False,
        )

    id_col = id_col or spec["meta"]["id_col"]
    target_col = target_col or spec["meta"]["target_col"]
    variables = list(spec["variables"].keys())

    cte = "\n".join("  " + l for l in sql_discretizacion.strip().splitlines())

    select = [f"  {_ident(c)}" for c in [id_col, target_col] if c]
    if incluir_bins:
        select += [f"  {_ident(v + '_bin')}" for v in variables]
    select += [_case_woe(var, cfg) for var, cfg in spec["variables"].items()]

    if incluir_score:
        suma = "\n    + ".join(_ident(v + "_woe") for v in variables)
        select.append(f"  (\n    {suma}\n  ) AS {_ident('score_woe')}")

    sql = (
        "WITH discretizado AS (\n" + cte + "\n)\n"
        "SELECT\n" + ",\n".join(select) + "\nFROM discretizado"
    )
    if incluir_cabecera:
        sql = _cabecera(spec, "Paso 2 — asignación de WOE") + "\n" + sql
    return sql


# ---------------------------------------------------------------------------
# 5. Persistencia del spec
# ---------------------------------------------------------------------------

def guardar_spec(spec: dict, ruta: str) -> None:
    """Guarda el spec a JSON. Es el artefacto versionable del modelo."""
    serializable = {k: v for k, v in spec.items() if k != "auditoria"}
    serializable["auditoria"] = [
        {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in fila.items()}
        for fila in spec["auditoria"]
    ]
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def cargar_spec(ruta: str) -> dict:
    """Recarga un spec guardado, para re-emitir SQL sin re-entrenar."""
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)
