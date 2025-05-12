import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import warnings

warnings.filterwarnings('ignore')

# =============================================
# CONFIGURACIÓN DE PARÁMETROS
# =============================================
excel_filename = 'base.xlsx'
columna_geografica = 'CIUDAD'
columna_producto = 'PRODUCTO'
columna_fecha = 'FECHA'
columna_cantidad = 'CANTIDAD'
min_meses_historia = 2
default_cantidad = 10

st.title("📦 Recomendación y Predicción de Productos por Ubicación")

# =============================================
# 1. CARGA DE DATOS
# =============================================
try:
    data_original = pd.read_excel(excel_filename, engine='openpyxl')
    columnas_requeridas = [columna_geografica, columna_producto, columna_fecha]
    for col in columnas_requeridas:
        if col not in data_original.columns:
            st.error(f"Columna requerida '{col}' no encontrada en el archivo.")
            st.stop()

    data = data_original[columnas_requeridas + ([columna_cantidad] if columna_cantidad in data_original.columns else [])].copy()
    data.dropna(subset=columnas_requeridas, inplace=True)

    data[columna_fecha] = pd.to_datetime(data[columna_fecha], errors='coerce')
    data.dropna(subset=[columna_fecha], inplace=True)
    data['MES_ANO'] = data[columna_fecha].dt.to_period('M')

    if columna_cantidad in data.columns:
        data[columna_cantidad] = pd.to_numeric(data[columna_cantidad], errors='coerce').fillna(1)
    else:
        data[columna_cantidad] = 1

    st.success("✅ Datos cargados correctamente")
    st.write(f"Ubicaciones únicas: {data[columna_geografica].nunique()}")
    st.write(f"Productos únicos: {data[columna_producto].nunique()}")
    st.write(f"Rango de fechas: {data[columna_fecha].min().date()} a {data[columna_fecha].max().date()}")

except Exception as e:
    st.error(f"❌ Error al cargar/preparar datos: {e}")
    st.stop()

# =============================================
# 2. SIMILITUD Y RECOMENDACIONES
# =============================================
interaction_matrix = data.groupby([columna_geografica, columna_producto])[columna_cantidad].sum().unstack(fill_value=0)

if interaction_matrix.shape[0] >= 2:
    lugar_similarity = cosine_similarity(interaction_matrix)
    lugar_sim_df = pd.DataFrame(lugar_similarity, 
                                index=interaction_matrix.index, 
                                columns=interaction_matrix.index)
else:
    lugar_sim_df = pd.DataFrame(index=interaction_matrix.index, columns=interaction_matrix.index)

def obtener_recomendaciones(lugar_actual, n_recomendaciones=5):
    if lugar_actual not in lugar_sim_df.index or lugar_sim_df.empty:
        return []

    similar_lugares = lugar_sim_df[lugar_actual].sort_values(ascending=False).index[1:]
    productos_actuales = interaction_matrix.loc[lugar_actual][interaction_matrix.loc[lugar_actual] > 0].index.tolist()
    recomendaciones = {}

    for lugar_similar in similar_lugares:
        productos_lugar_similar = interaction_matrix.loc[lugar_similar]
        productos_nuevos = productos_lugar_similar[(productos_lugar_similar > 0) & (~productos_lugar_similar.index.isin(productos_actuales))].index.tolist()
        sim_score = lugar_sim_df.loc[lugar_similar, lugar_actual]
        for producto in productos_nuevos:
            recomendaciones[producto] = recomendaciones.get(producto, 0) + sim_score

    return sorted(recomendaciones.items(), key=lambda x: x[1], reverse=True)[:n_recomendaciones]

recomendaciones_por_ubicacion = {}
if not interaction_matrix.empty:
    for ubicacion_idx in interaction_matrix.index:
        recs = obtener_recomendaciones(ubicacion_idx)
        if recs:
            recomendaciones_por_ubicacion[ubicacion_idx] = recs

# =============================================
# 3. PREDICCIÓN DE CANTIDADES
# =============================================
ventas_mensuales = data.groupby(
    [columna_geografica, columna_producto, 'MES_ANO']
)[columna_cantidad].sum().reset_index()
ventas_mensuales['MES_ANO'] = ventas_mensuales['MES_ANO'].dt.to_timestamp()

promedio_ventas_producto_global = ventas_mensuales.groupby(columna_producto)[columna_cantidad].mean().reset_index()
promedio_ventas_producto_global.rename(columns={columna_cantidad: 'CANTIDAD_PROMEDIO_GLOBAL'}, inplace=True)

predicciones_finales = {}

for ubicacion, productos_recomendados in recomendaciones_por_ubicacion.items():
    predicciones_finales[ubicacion] = []
    for producto, score in productos_recomendados:
        historial_local = ventas_mensuales[
            (ventas_mensuales[columna_geografica] == ubicacion) &
            (ventas_mensuales[columna_producto] == producto)
        ].sort_values('MES_ANO')
        cantidad_predicha = default_cantidad
        meses_historicos_locales = len(historial_local)

        if meses_historicos_locales >= min_meses_historia:
            historial_local = historial_local.assign(TIME_INDEX=range(len(historial_local)))
            try:
                model = LinearRegression()
                model.fit(historial_local[['TIME_INDEX']], historial_local[columna_cantidad])
                next_index = [[len(historial_local)]]
                pred = model.predict(next_index)[0]
                cantidad_predicha = max(1, round(pred))
                p75_local = historial_local[columna_cantidad].quantile(0.75)
                if p75_local > 0 and cantidad_predicha > 3 * p75_local:
                    cantidad_predicha = max(default_cantidad, round(p75_local))
                elif p75_local == 0 and cantidad_predicha > 2 * default_cantidad:
                    cantidad_predicha = 2 * default_cantidad
            except:
                cantidad_predicha = max(default_cantidad, round(historial_local[columna_cantidad].median()))
                cantidad_predicha = max(1, cantidad_predicha)
        elif meses_historicos_locales > 0:
            cantidad_predicha = max(default_cantidad, round(historial_local[columna_cantidad].mean()))
            cantidad_predicha = max(1, cantidad_predicha)
        else:
            info_producto_otras_ubic = promedio_ventas_producto_global[
                promedio_ventas_producto_global[columna_producto] == producto
            ]
            if not info_producto_otras_ubic.empty:
                cantidad_promedio_global_producto = info_producto_otras_ubic['CANTIDAD_PROMEDIO_GLOBAL'].iloc[0]
                cantidad_predicha = max(default_cantidad, round(cantidad_promedio_global_producto))
                cantidad_predicha = max(1, cantidad_predicha)
            else:
                cantidad_predicha = default_cantidad

        predicciones_finales[ubicacion].append({
            'Producto': producto,
            'Score_Recomendacion': round(score, 4),
            'Cantidad_Predicha': int(cantidad_predicha),
        })

# =============================================
# 4. VISUALIZACIÓN FINAL
# =============================================
st.subheader("📊 Resultados por Ubicación")

if not predicciones_finales:
    st.warning("No se generaron predicciones finales.")
else:
    for ubicacion_res, predicciones_list in predicciones_finales.items():
        st.markdown(f"### 📍 {ubicacion_res}")
        if predicciones_list:
            df_res = pd.DataFrame(predicciones_list)
            st.dataframe(df_res)
        else:
            st.info("No hay predicciones para esta ubicación.")
