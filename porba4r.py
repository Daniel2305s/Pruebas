import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import warnings

# Configuración inicial
warnings.filterwarnings('ignore')

# =============================================
# CONFIGURACIÓN DE PARÁMETROS (AJUSTAR SEGÚN DATOS)
# =============================================
excel_filename = 'C:\Users\User\Documents\GitHub\Pruebas\base.xlsx'  # Nombre del archivo Excel
columna_geografica = 'CIUDAD'   # Columna con ubicaciones (ciudades/departamentos)
columna_producto = 'PRODUCTO'   # Columna con nombres de productos
columna_fecha = 'FECHA'         # Columna con fechas de transacciones
columna_cantidad = 'CANTIDAD'   # Columna con cantidades vendidas (opcional)

min_meses_historia = 2        # Mínimo de meses con datos para hacer predicción
default_cantidad = 10         # Valor por defecto cuando no hay suficiente historia o para productos nuevos

# =============================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# =============================================
print("Cargando y preparando datos...")

try:
    # Cargar datos originales
    data_original = pd.read_excel(excel_filename, engine='openpyxl')
    print(f"Datos cargados. Dimensiones: {data_original.shape}")
    
    # Verificar columnas requeridas
    columnas_requeridas = [columna_geografica, columna_producto, columna_fecha]
    for col in columnas_requeridas:
        if col not in data_original.columns:
            raise ValueError(f"Columna requerida '{col}' no encontrada en el archivo")
    
    # Limpieza básica
    data = data_original[columnas_requeridas + ([columna_cantidad] if columna_cantidad in data_original.columns else [])].copy()
    data.dropna(subset=columnas_requeridas, inplace=True)
    
    # Convertir fecha y extraer mes-año
    data[columna_fecha] = pd.to_datetime(data[columna_fecha], errors='coerce')
    data.dropna(subset=[columna_fecha], inplace=True)
    data['MES_ANO'] = data[columna_fecha].dt.to_period('M')
    
    # Si no hay columna cantidad, asumimos 1 unidad por registro
    # Si existe, llenar NaNs con 1 (o considerar 0 si tiene más sentido para el negocio)
    if columna_cantidad in data.columns:
        data[columna_cantidad] = pd.to_numeric(data[columna_cantidad], errors='coerce').fillna(1)
    else:
        data[columna_cantidad] = 1
        print(f"'{columna_cantidad}' no encontrada, se usará 1 por defecto para cada transacción.")
    
    print("Datos preparados correctamente.")
    print(f"Ubicaciones únicas: {data[columna_geografica].nunique()}")
    print(f"Productos únicos: {data[columna_producto].nunique()}")
    print(f"Rango de fechas: {data[columna_fecha].min()} a {data[columna_fecha].max()}")
    
except Exception as e:
    print(f"Error al cargar/preparar datos: {e}")
    raise

# =============================================
# 2. SISTEMA DE RECOMENDACIÓN (CLUSTERING)
# =============================================
print("\nGenerando recomendaciones por similitud...")

# Crear matriz de interacciones (ubicación x producto)
# Usamos sum() para agregar cantidades si hay múltiples registros del mismo producto en la misma ubicación (implícito por el groupby posterior)
# Aquí, para la matriz de interacción, es mejor usar la presencia/ausencia o una cuenta agregada simple.
# La suma ya se hace para ventas_mensuales. Para la matriz de interacción, la suma de cantidades está bien.
interaction_matrix = data.groupby([columna_geografica, columna_producto])[columna_cantidad].sum().unstack(fill_value=0)

# Calcular similitud coseno entre ubicaciones
if interaction_matrix.shape[0] >= 2:
    lugar_similarity = cosine_similarity(interaction_matrix)
    lugar_sim_df = pd.DataFrame(lugar_similarity, 
                                index=interaction_matrix.index, 
                                columns=interaction_matrix.index)
elif interaction_matrix.shape[0] > 0:
    print("Solo hay una ubicación. No se pueden calcular similitudes entre ubicaciones.")
    lugar_sim_df = pd.DataFrame(index=interaction_matrix.index, columns=interaction_matrix.index) # DataFrame vacío con índice/columnas
else:
    raise ValueError("No hay datos de interacción suficientes para construir la matriz de interacción.")


def obtener_recomendaciones(lugar_actual, n_recomendaciones=5):
    """
    Obtiene recomendaciones de productos para una ubicación basado en ubicaciones similares
    """
    if lugar_actual not in lugar_sim_df.index or lugar_sim_df.empty:
        return []
    
    # Obtener ubicaciones similares (excluyendo la actual)
    # Asegurarse de que hay otras ubicaciones para comparar
    if len(lugar_sim_df.columns) < 2:
        return []
        
    similar_lugares = lugar_sim_df[lugar_actual].sort_values(ascending=False).index[1:]
    
    # Productos que ya tiene la ubicación actual
    productos_actuales = interaction_matrix.loc[lugar_actual][interaction_matrix.loc[lugar_actual] > 0].index.tolist()
    
    # Recolectar recomendaciones con sus scores
    recomendaciones = {}
    for lugar_similar in similar_lugares:
        if lugar_similar == lugar_actual: # Doble chequeo
            continue

        # Productos del lugar similar que no están en el actual
        productos_lugar_similar = interaction_matrix.loc[lugar_similar]
        productos_nuevos = productos_lugar_similar[
            (productos_lugar_similar > 0) & 
            (~productos_lugar_similar.index.isin(productos_actuales))
        ].index.tolist()
        
        # Ponderar por similitud
        sim_score = lugar_sim_df.loc[lugar_similar, lugar_actual] # Similitud entre lugar_similar y lugar_actual
        for producto in productos_nuevos:
            recomendaciones[producto] = recomendaciones.get(producto, 0) + sim_score
    
    # Ordenar y devolver top N
    return sorted(recomendaciones.items(), key=lambda x: x[1], reverse=True)[:n_recomendaciones]

# Generar recomendaciones para todas las ubicaciones
recomendaciones_por_ubicacion = {}
if not interaction_matrix.empty:
    for ubicacion_idx in interaction_matrix.index:
        recs = obtener_recomendaciones(ubicacion_idx)
        if recs:
            recomendaciones_por_ubicacion[ubicacion_idx] = recs

if not recomendaciones_por_ubicacion:
    print("\nNo se pudieron generar recomendaciones (pocas ubicaciones o datos).")
else:
    print("\nRecomendaciones generadas:")
    for ubicacion_rec, recs_list in recomendaciones_por_ubicacion.items():
        print(f"\n{ubicacion_rec}:")
        for producto_rec, score_rec in recs_list:
            print(f"  - {producto_rec} (score: {score_rec:.4f})")

# =============================================
# 3. PREDICCIÓN DE CANTIDADES (REGRESIÓN LINEAL Y HEURÍSTICAS)
# =============================================
print("\n\nPrediciendo cantidades para productos recomendados...")

# Preparar datos históricos mensuales
ventas_mensuales = data.groupby(
    [columna_geografica, columna_producto, 'MES_ANO']
)[columna_cantidad].sum().reset_index()
ventas_mensuales['MES_ANO'] = ventas_mensuales['MES_ANO'].dt.to_timestamp()

# Calcular la venta mensual promedio de cada producto en las ubicaciones donde se vende
# Esto servirá como base para predecir en nuevas ubicaciones.
promedio_ventas_producto_global = ventas_mensuales.groupby(columna_producto)[columna_cantidad].mean().reset_index()
promedio_ventas_producto_global.rename(columns={columna_cantidad: 'CANTIDAD_PROMEDIO_GLOBAL'}, inplace=True)

predicciones_finales = {}

for ubicacion, productos_recomendados in recomendaciones_por_ubicacion.items():
    predicciones_finales[ubicacion] = []
    
    for producto, score in productos_recomendados:
        # Filtrar datos históricos para este producto y ubicación actual
        historial_local = ventas_mensuales[
            (ventas_mensuales[columna_geografica] == ubicacion) &
            (ventas_mensuales[columna_producto] == producto)
        ].sort_values('MES_ANO')
        
        cantidad_predicha = default_cantidad  # Valor por defecto inicial
        meses_historicos_locales = len(historial_local)
        
        if meses_historicos_locales >= min_meses_historia:
            # Hay suficiente historia local, usar regresión lineal
            historial_local = historial_local.assign(
                TIME_INDEX=range(len(historial_local))
            )
            try:
                model = LinearRegression()
                model.fit(historial_local[['TIME_INDEX']], historial_local[columna_cantidad])
                
                next_index = [[len(historial_local)]] # Predecir para el siguiente período
                pred = model.predict(next_index)[0]
                
                cantidad_predicha = max(1, round(pred)) # Asegurar valor positivo, mínimo 1
                
                # Limitar predicciones muy altas usando el percentil 75 del histórico local
                # Solo aplicar si el percentil es positivo para evitar problemas con historiales de ceros.
                p75_local = historial_local[columna_cantidad].quantile(0.75)
                if p75_local > 0 and cantidad_predicha > 3 * p75_local:
                    cantidad_predicha = max(default_cantidad, round(p75_local))
                elif p75_local == 0 and cantidad_predicha > 2 * default_cantidad: # Historial bajo pero predicción alta
                    cantidad_predicha = 2 * default_cantidad

            except Exception as e_reg:
                # print(f"Advertencia: Falla en regresión para {ubicacion}-{producto}: {e_reg}. Usando mediana local.")
                # Si la regresión falla, usar la mediana histórica local, asegurando que sea al menos default_cantidad
                cantidad_predicha = max(default_cantidad, round(historial_local[columna_cantidad].median()))
                cantidad_predicha = max(1, cantidad_predicha) # Asegurar mínimo 1
        
        elif meses_historicos_locales > 0: # Poca historia local (menos que min_meses_historia pero > 0)
            # print(f"Advertencia: Poca historia local para {ubicacion}-{producto}. Usando media local.")
            # Usar promedio histórico local, asegurando que sea al menos default_cantidad
            cantidad_predicha = max(default_cantidad, round(historial_local[columna_cantidad].mean()))
            cantidad_predicha = max(1, cantidad_predicha) # Asegurar mínimo 1
        
        else: # No hay historia local (producto verdaderamente nuevo para la ubicación)
            # print(f"Info: Sin historia local para {ubicacion}-{producto}. Usando promedio de otras ubicaciones o default.")
            # Buscar la venta promedio de este producto en otras ubicaciones
            info_producto_otras_ubic = promedio_ventas_producto_global[
                promedio_ventas_producto_global[columna_producto] == producto
            ]
            
            if not info_producto_otras_ubic.empty:
                cantidad_promedio_global_producto = info_producto_otras_ubic['CANTIDAD_PROMEDIO_GLOBAL'].iloc[0]
                # Usar el promedio global del producto, pero no menos que default_cantidad.
                # Esto significa que si el producto generalmente se vende poco (ej., 3 unidades),
                # y default_cantidad es 10, se predecirán 10.
                # Si se vende bien globalmente (ej., 20 unidades), se predecirán 20.
                cantidad_predicha = max(default_cantidad, round(cantidad_promedio_global_producto))
                cantidad_predicha = max(1, cantidad_predicha) # Asegurar mínimo 1
            else:
                # El producto no tiene historial en ninguna otra ubicación (producto nuevo en general)
                # print(f"Info: Producto {producto} sin historial global. Usando default_cantidad.")
                cantidad_predicha = default_cantidad # Ya es max(1, default_cantidad) implicitamente si default_cantidad >=1
        
        predicciones_finales[ubicacion].append({
            'Producto': producto,
            'Score_Recomendacion': score,
            'Cantidad_Predicha': int(cantidad_predicha), # Asegurar que la cantidad sea entera
            
        })

# =============================================
# 4. RESULTADOS FINALES
# =============================================
print("\n\n=== RESULTADOS FINALES ===")
if not predicciones_finales:
    print(f"No se generaron predicciones finales.")
else:
    print(f"Recomendaciones con predicciones de cantidad (mínimo {max(1,default_cantidad)} unidades si no hay otra info)")

    for ubicacion_res, predicciones_list in predicciones_finales.items():
        print(f"\n--- {ubicacion_res} ---")
        if predicciones_list:
            df_res = pd.DataFrame(predicciones_list)
            df_res['Score_Recomendacion'] = df_res['Score_Recomendacion'].round(4)
            # Cambiado 'Meses_Historicos' a 'Meses_Historicos_Locales'
            print(df_res[['Producto', 'Score_Recomendacion', 'Cantidad_Predicha']].to_string(index=False))
        else:
            print("No hay predicciones para esta ubicación.")