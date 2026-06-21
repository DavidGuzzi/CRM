import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings


def clasificar_variables(df: pd.DataFrame,
                         umbral_categorica: int = 10,
                         excluir_cols: List[str] = None,
                         forzar_categoricas: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """
    Clasifica variables en numéricas y categóricas.

    Parámetros:
        df: DataFrame con datos
        umbral_categorica: Umbral de valores únicos para considerar categórica
        excluir_cols: Lista de columnas a excluir
        forzar_categoricas: Lista de columnas a tratar como categóricas sin importar tipo o umbral

    Retorna:
        Dict con keys 'numericas' y 'categoricas'
    """
    if excluir_cols is None:
        excluir_cols = []
    if forzar_categoricas is None:
        forzar_categoricas = []

    numericas = []
    categoricas = []

    for col in df.columns:
        if col in excluir_cols:
            continue

        if col in forzar_categoricas:
            categoricas.append(col)
            continue

        # Variables numéricas
        if pd.api.types.is_numeric_dtype(df[col]):
            n_unique = df[col].nunique()
            if n_unique >= umbral_categorica:
                numericas.append(col)
            else:
                categoricas.append(col)
        else:
            categoricas.append(col)

    return {'numericas': numericas, 'categoricas': categoricas}

def calcular_metricas_bin(df_grouped: pd.DataFrame,
                          cant_buenos_total: int,
                          cant_malos_total: int) -> pd.DataFrame:
    """
    Calcula métricas WOE, IV y porcentajes por bin.

    Parámetros:
        df_grouped: DataFrame agrupado con cantidad, cantmalos, cantbuenos
        cant_buenos_total: Total de buenos en dataset
        cant_malos_total: Total de malos en dataset

    Retorna:
        DataFrame con métricas calculadas
    """
    # Porcentajes
    df_grouped['porcmalogrupo'] = np.round(
        (df_grouped['cantmalos'] / df_grouped['cantidad']) * 100, 2
    )
    df_grouped['porctotal'] = np.round(
        (df_grouped['cantidad'] / (cant_buenos_total + cant_malos_total)) * 100, 2
    )
    df_grouped['porcbuenostotal'] = np.round(
        (df_grouped['cantbuenos'] / cant_buenos_total) * 100, 2
    )
    df_grouped['porcmalostotal'] = np.round(
        (df_grouped['cantmalos'] / cant_malos_total) * 100, 2
    )

    # WOE
    df_grouped['woe'] = np.where(
        (df_grouped['porcmalostotal'] == 0) | (df_grouped['porcbuenostotal'] == 0),
        0,
        np.log(df_grouped['porcbuenostotal'] / df_grouped['porcmalostotal'])
    )
    df_grouped['woe'] = np.round(df_grouped['woe'], 6)

    # IV
    df_grouped['iv'] = (
        df_grouped['porcbuenostotal'] - df_grouped['porcmalostotal']
    ) * df_grouped['woe']

    return df_grouped

def calcular_bivariados(
    df: pd.DataFrame,
    target_col: str = 'target',
    n_bins: int = 5,
    metodo: str = 'quantile',
    umbral_categorica: int = 5,
    excluir_cols: Optional[List[str]] = None,
    forzar_categoricas: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calcula análisis bivariado con WOE e IV para todas las variables.

    Parámetros:
        df: DataFrame con datos
        target_col: Nombre de columna target (binaria 0/1)
        n_bins: Número de bins (5=quintiles, 10=deciles)
        metodo: Método de binning para variables numéricas:
            - 'quantile': percentiles reales con pd.qcut sobre valores directos.
              Puede generar bins de tamaño desigual si hay muchos valores repetidos.
            - 'quantile_equal_size': pd.qcut sobre el rank de los valores.
              Garantiza bins de igual cantidad de registros.
            - 'equal_width': rangos de ancho igual con pd.cut.
        umbral_categorica: Umbral para clasificar variable como categórica
        excluir_cols: Lista de columnas a excluir del análisis
        forzar_categoricas: Lista de columnas a tratar como categóricas sin importar tipo o umbral

    Retorna:
        DataFrame con columnas: variable, bin, minimo, maximo, cantidad, cantmalos,
        cantbuenos, porcmalogrupo, porctotal, porcbuenostotal, porcmalostotal,
        woe, iv, iv_total_variable
    """
    # Validaciones
    if target_col not in df.columns:
        raise ValueError(f"Columna target '{target_col}' no encontrada en DataFrame")

    if df[target_col].nunique() != 2:
        raise ValueError(f"Target debe ser binario (0/1)")

    if excluir_cols is None:
        excluir_cols = []
    excluir_cols.append(target_col)

    # Clasificar variables
    vars_clasificadas = clasificar_variables(df, umbral_categorica, excluir_cols, forzar_categoricas)

    # Totales
    cant_buenos_total = (df[target_col] == 0).sum()
    cant_malos_total = (df[target_col] == 1).sum()

    resultados = []

    # Procesar variables numéricas
    for var in vars_clasificadas['numericas']:
        try:
            df_temp = df[[var, target_col]].copy()

            # Crear bins según método elegido
            if metodo == 'quantile':
                df_temp['bin'] = pd.qcut(df_temp[var], q=n_bins, labels=False, duplicates='drop') + 1
            elif metodo == 'quantile_equal_size':
                df_temp['bin'] = pd.qcut(df_temp[var].rank(method='first'), q=n_bins, labels=False, duplicates='drop') + 1
            elif metodo == 'equal_width':
                df_temp['bin'] = pd.cut(df_temp[var], bins=n_bins, labels=False) + 1
            else:
                raise ValueError(f"Método '{metodo}' no soportado. Use 'quantile', 'quantile_equal_size' o 'equal_width'")

            # Agregar por bin
            agg = df_temp.groupby('bin').agg({
                var: ['min', 'max', 'count'],
                target_col: ['sum']
            }).reset_index()

            agg.columns = ['bin', 'minimo', 'maximo', 'cantidad', 'cantmalos']
            agg['cantbuenos'] = agg['cantidad'] - agg['cantmalos']
            agg['variable'] = var

            # Calcular métricas
            agg = calcular_metricas_bin(agg, cant_buenos_total, cant_malos_total)

            resultados.append(agg)

        except Exception as e:
            print(f"Warning: No se pudo procesar variable '{var}': {e}")
            continue

    # Procesar variables categóricas
    for var in vars_clasificadas['categoricas']:
        try:
            df_temp = df[[var, target_col]].copy()

            df_temp['bin'] = df_temp[var].astype(str)

            agg = df_temp.groupby(['bin', var]).agg({
                target_col: ['count', 'sum']
            }).reset_index()

            agg.columns = ['bin', 'valor_variable', 'cantidad', 'cantmalos']
            agg['cantbuenos'] = agg['cantidad'] - agg['cantmalos']
            agg['variable'] = var

            agg['minimo'] = agg['valor_variable']
            agg['maximo'] = agg['valor_variable']

            agg = agg.drop(columns=['valor_variable'])

            # Calcular métricas
            agg = calcular_metricas_bin(
                agg, cant_buenos_total, cant_malos_total
            )

            resultados.append(agg)

        except Exception as e:
            print(f"Warning: No se pudo procesar variable '{var}': {e}")
            continue

    # Consolidar resultados
    if not resultados:
        raise ValueError("No se pudieron procesar variables")

    df_resultado = pd.concat(resultados, ignore_index=True)

    # Calcular IV total por variable
    iv_total = df_resultado.groupby('variable')['iv'].sum().reset_index()
    iv_total.columns = ['variable', 'iv_total_variable']

    df_resultado = df_resultado.merge(iv_total, on='variable')

    # Ordenar
    df_resultado = df_resultado.sort_values(
        ['iv_total_variable', 'variable', 'bin'],
        ascending=[False, True, True]
    ).reset_index(drop=True)

    # Reordenar columnas
    cols_order = ['variable', 'bin', 'minimo', 'maximo', 'cantidad', 'cantmalos',
                  'cantbuenos', 'porcmalogrupo', 'porctotal', 'porcbuenostotal',
                  'porcmalostotal', 'woe', 'iv', 'iv_total_variable']
    df_resultado = df_resultado[cols_order]

    return df_resultado

def discretizar_variables(
    df: pd.DataFrame,
    config_bins: Dict,
    calcular_percentiles: bool = True,
    percentiles_precalculados: Optional[Dict] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Discretiza variables numéricas y categóricas según una configuración declarativa,
    utilizando percentiles reales sobre valores observados.

    Diseñada para modelado de riesgo / scorecards. Principios:
    - No divide observaciones con el mismo valor en bins distintos.
    - No crea bins artificiales cuando los percentiles colapsan en el mismo valor.
    - Maneja explícitamente un nivel especial para valores cero (nivel_0),
      replicando la lógica CASE WHEN de SQL sin interpolaciones continuas.
    - Los percentiles se calculan sobre valores observados, garantizando
      estabilidad y consistencia out-of-sample.

    Parámetros:
        df: DataFrame de entrada con las variables a discretizar.
        config_bins: Diccionario de configuración por variable. Tipos soportados:
            - 'passthrough': no discretiza
            - 'percentiles': cortes por percentiles, con opción nivel_cero
            - 'categorica': convierte a string
            - 'dicotomica': corte binario por umbral
            - 'custom': cortes fijos definidos manualmente
        calcular_percentiles: Si True, calcula percentiles sobre df (train).
            Si False, usa percentiles_precalculados (test / OOT).
        percentiles_precalculados: Percentiles ya calculados en train.
            Requerido si calcular_percentiles=False.

    Retorna:
        Tuple con:
        - df_binneado: DataFrame con columnas adicionales {variable}_bin
        - percentiles_calculados: dict con los percentiles usados por variable

    Notas:
        Si una variable no permite generar todos los bins (por colapso de percentiles),
        los bins inexistentes simplemente no se crean. Esta función NO replica la
        interpolación continua de PERCENTILE_CONT de SQL.
    """

    df_result = df.copy()
    percentiles_calculados = {}

    for var, config in config_bins.items():

        if var not in df_result.columns:
            print(f"Warning: Variable '{var}' no encontrada")
            continue

        tipo = config.get('tipo', 'percentiles')

        # =========================
        # VARIABLES PASSTHROUGH (NO DISCRETIZAR)
        # =========================
        if tipo == 'passthrough':
            continue

        # =========================
        # VARIABLES CATEGÓRICAS
        # =========================
        if tipo == 'categorica':
            df_result[f'{var}_bin'] = df_result[var].astype(str)

        # =========================
        # VARIABLES POR PERCENTILES
        # =========================
        elif tipo == 'percentiles':

            cortes = config['cortes']
            nivel_cero = config.get('nivel_cero', False)

            # --------
            # Cálculo de percentiles (solo valores observados)
            # --------
            if calcular_percentiles:
                base = (
                    df_result.loc[df_result[var] > 0, var]
                    if nivel_cero
                    else df_result[var]
                )

                if base.empty or base.nunique() <= 1:
                    print(f"Warning: '{var}' sin variabilidad suficiente para binning")
                    df_result[f'{var}_bin'] = 'sin_variacion'
                    continue

                percentiles = base.quantile(
                    cortes,
                    interpolation='higher'
                ).tolist()
            else:
                if percentiles_precalculados is None or var not in percentiles_precalculados:
                    raise ValueError(f"Percentiles para '{var}' no proporcionados")
                percentiles = percentiles_precalculados[var]

            percentiles_calculados[var] = percentiles

            # =========================
            # NIVEL CERO
            # =========================
            if nivel_cero:

                def asignar_bin(x):
                    if x <= 0:
                        return 'nivel_0'
                    for c, p in zip(cortes, percentiles):
                        if x <= p:
                            return f'percentil_{int(c * 100)}'
                    return 'percentil_100'

                df_result[f'{var}_bin'] = df_result[var].apply(asignar_bin)

            # =========================
            # SIN NIVEL CERO
            # =========================
            else:
                percentiles_unicos = np.unique(percentiles)

                bins = [-np.inf] + percentiles_unicos.tolist() + [np.inf]
                labels = (
                    [f'percentil_{int(c * 100)}'
                     for c in cortes[:len(percentiles_unicos)]]
                    + ['percentil_100']
                )

                df_result[f'{var}_bin'] = pd.cut(
                    df_result[var],
                    bins=bins,
                    labels=labels,
                    include_lowest=True
                )

        # =========================
        # VARIABLES DICOTÓMICAS (UMBRAL)
        # =========================
        elif tipo == 'dicotomica':

            if 'umbral' not in config:
                raise ValueError(
                    f"Variable '{var}' tipo dicotomica requiere 'umbral'"
                )

            umbral = config['umbral']

            df_result[f'{var}_bin'] = np.where(
                df_result[var] <= umbral,
                f'menor_igual_{umbral}',
                f'mayor_{umbral}'
            )

        # =========================
        # VARIABLES CON CORTES CUSTOM
        # =========================
        elif tipo == 'custom':

            cortes = config['cortes']
            bins = [-np.inf] + cortes + [np.inf]
            labels = [f'bin_{i}' for i in range(len(bins) - 1)]

            df_result[f'{var}_bin'] = pd.cut(
                df_result[var],
                bins=bins,
                labels=labels,
                include_lowest=True
            )

        else:
            print(f"Warning: Tipo '{tipo}' no soportado para '{var}'")

    return df_result, percentiles_calculados

def exportar_multiples_bivariados_excel(
    hojas: Dict[str, pd.DataFrame],
    archivo_salida: str = 'bivariados.xlsx',
    columnas_color: Optional[List[str]] = None
) -> None:
    """
    Exporta múltiples DataFrames de bivariados a un mismo archivo Excel,
    cada uno en una hoja separada, con formato condicional.

    Parámetros:
        hojas: Diccionario {nombre_hoja: DataFrame_bivariados}
               Ejemplo: {'Quintiles': df_quintiles, 'Deciles': df_deciles}
        archivo_salida: Ruta del archivo Excel de salida
        columnas_color: Lista de columnas con formato condicional degradado.
                        Default: ['woe']

    Retorna:
        None (genera archivo Excel)
    """
    if columnas_color is None:
        columnas_color = ['woe']

    with pd.ExcelWriter(archivo_salida, engine='xlsxwriter') as writer:
        workbook = writer.book

        # Formato de encabezado (compartido entre hojas)
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#4472C4',
            'font_color': 'white',
            'border': 1,
            'text_wrap': True,
            'valign': 'vcenter'
        })

        for nombre_hoja, bivariados_df in hojas.items():
            # Validar columnas
            cols_validas = [c for c in columnas_color if c in bivariados_df.columns]

            # Insertar renglones vacíos entre cambios de variable
            filas_con_separador = []
            variable_anterior = None

            for _, fila in bivariados_df.iterrows():
                if variable_anterior is not None and fila['variable'] != variable_anterior:
                    fila_vacia = pd.Series({col: np.nan for col in bivariados_df.columns})
                    filas_con_separador.append(fila_vacia)
                filas_con_separador.append(fila)
                variable_anterior = fila['variable']

            df_export = pd.DataFrame(filas_con_separador).reset_index(drop=True)

            # Escribir hoja
            df_export.to_excel(writer, sheet_name=nombre_hoja, index=False)
            worksheet = writer.sheets[nombre_hoja]

            # Formato encabezados
            for col_idx, col_name in enumerate(df_export.columns):
                worksheet.write(0, col_idx, col_name, header_format)

            # Formato condicional degradado
            n_filas = len(df_export)

            for col_name in cols_validas:
                col_idx = df_export.columns.get_loc(col_name)
                col_letter = chr(ord('A') + col_idx) if col_idx < 26 else \
                             chr(ord('A') + col_idx // 26 - 1) + chr(ord('A') + col_idx % 26)

                rango = f'{col_letter}2:{col_letter}{n_filas + 1}'

                worksheet.conditional_format(rango, {
                    'type': '3_color_scale',
                    'min_type': 'num',
                    'min_value': 3,
                    'mid_type': 'num',
                    'mid_value': 10,
                    'max_type': 'num',
                    'max_value': 50,
                    'min_color': '#1A9850',
                    'mid_color': '#FEE08B',
                    'max_color': '#D73027',
                })

            # Ajustar ancho de columnas
            for col_idx, col_name in enumerate(df_export.columns):
                max_len = max(
                    len(str(col_name)),
                    df_export[col_name].astype(str).str.len().max() if len(df_export) > 0 else 0
                )
                worksheet.set_column(col_idx, col_idx, min(max_len + 2, 25))

            print(f"  ✓ Hoja '{nombre_hoja}': {len(bivariados_df)} filas de datos")

    print(f"\n✓ Excel exportado a: {archivo_salida}")
    print(f"  - Hojas: {list(hojas.keys())}")
    print(f"  - Columnas con formato condicional: {columnas_color}")


def calcular_woe_mapping(
    df_binneado: pd.DataFrame,
    target_col: str = 'target',
    variables_binneadas: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calcula mapeo de WOE para variables discretizadas y no discretizadas.

    Parámetros:
        df_binneado: DataFrame con variables discretizadas (sufijo _bin)
        target_col: Nombre de columna target
        variables_binneadas: Lista explícita de variables _bin (default: detectar automáticamente)

    Retorna:
        DataFrame con columnas: variable, bin, woe, iv, iv_total
    """

    # Variables binneadas
    if variables_binneadas is None:
        variables_binneadas = [c for c in df_binneado.columns if c.endswith('_bin')]

    if not variables_binneadas:
        raise ValueError("No se especificaron variables para calcular WOE")

    # Totales
    cant_buenos_total = (df_binneado[target_col] == 0).sum()
    cant_malos_total = (df_binneado[target_col] == 1).sum()

    resultados = []

    for var in variables_binneadas:
        if var not in df_binneado.columns:
            raise ValueError(f"La variable '{var}' no existe en el DataFrame")

        # Agrupar (cada valor es un bin si no está discretizada)
        agg = df_binneado.groupby(var, dropna=False).agg({
            target_col: ['count', 'sum']
        }).reset_index()

        agg.columns = ['bin', 'cantidad', 'cantmalos']
        agg['cantbuenos'] = agg['cantidad'] - agg['cantmalos']
        agg['variable'] = var

        # Calcular métricas WOE / IV
        agg = calcular_metricas_bin(
            agg,
            cant_buenos_total,
            cant_malos_total
        )

        resultados.append(agg[['variable', 'bin', 'woe', 'iv']])

    # Consolidar
    df_mapeo = pd.concat(resultados, ignore_index=True)

    # IV total por variable
    iv_total = (
        df_mapeo
        .groupby('variable', as_index=False)['iv']
        .sum()
        .rename(columns={'iv': 'iv_total'})
    )

    df_mapeo = df_mapeo.merge(iv_total, on='variable')

    return df_mapeo

def aplicar_woe(
    df: pd.DataFrame,
    mapeo_woe: pd.DataFrame,
    variables: Optional[List[str]] = None,
    sufijo_woe: str = '_woe'
) -> pd.DataFrame:
    """
    Aplica valores WOE a DataFrame usando mapeo pre-calculado.
    
    Parámetros:
        df: DataFrame con variables binneadas
        mapeo_woe: DataFrame de mapeo (output de calcular_woe_mapping)
        variables: Lista de variables a transformar (default: todas en mapeo)
        sufijo_woe: Sufijo para nuevas columnas WOE
    
    Retorna:
        DataFrame con columnas WOE adicionales
    """
    df_result = df.copy()
    
    if variables is None:
        variables = mapeo_woe['variable'].unique()
    
    for var in variables:
        if var not in df.columns:
            print(f"Warning: Variable '{var}' no encontrada en DataFrame")
            continue
        
        # Obtener mapeo para esta variable
        mapeo_var = mapeo_woe[mapeo_woe['variable'] == var][['bin', 'woe']]
        mapeo_dict = dict(zip(mapeo_var['bin'], mapeo_var['woe']))
        
        # Aplicar WOE
        df_result[f'{var}{sufijo_woe}'] = df_result[var].map(mapeo_dict)
        
        # Manejar bins no encontrados en mapeo
        if df_result[f'{var}{sufijo_woe}'].isnull().any():
            n_null = df_result[f'{var}{sufijo_woe}'].isnull().sum()
            print(f"Warning: {n_null} valores sin WOE en '{var}' (bins no encontrados en mapeo)")
    
    return df_result