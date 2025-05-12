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
EXCEL_FILENAME = 'base.xlsx'
COLUMNA_GEOGRAFICA = 'CIUDAD'
COLUMNA_PRODUCTO = 'PRODUCTO'
COLUMNA_FECHA = 'FECHA'
COLUMNA_CANTIDAD = 'CANTIDAD'
MIN_MESES_HISTORIA = 2
DEFAULT_CANTIDAD = 10

# =============================================
# CARGA DE DATOS
# =============================================
@st.cache_data

def cargar_datos():
    try:
        data_original = pd.read_excel(EXCEL_FILENAME, engine='openpyxl')
        columnas_requeridas = [COLUMNA_GEOGRAFICA, COLUMNA_PRODUCTO, COLUMNA_FECHA]
        for col in columnas_requeridas:
            if col not in data_original.columns:
                st.error(f"Columna requerida '{col}' no encontrada en el archivo")
                return None

        data = data_original[columnas_requeridas + ([COLUMNA_CANTIDAD] if COLUMNA_CANTIDAD in data_original.columns else [])].copy()
        data.dropna(subset=columnas_requeridas, inplace=True)

        data[COLUMNA_FECHA] = pd.to_datetime(data[COLUMNA_FECHA], errors='coerce')
        data.dropna(subset=[COLUMNA_FECHA], inplace=True)
        data['MES_ANO'] = data[COLUMNA_FECHA].dt.to_period('M')

        if COLUMNA_CANTIDAD in data.columns:
            data[COLUMNA_CANTIDAD] = pd.to_numeric(data[COLUMNA_CANTIDAD], errors='coerce').fillna(1)
        else:
            data[COLUMNA_CANTIDAD] = 1

        return data
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return None

# =============================================
# SISTEMA DE RECOMENDACIÓN
# =============================================
def generar_recomendaciones(data):
    interaction_matrix = data.groupby([COLUMNA_GEOGRAFICA, COLUMNA_PRODUCTO])[COLUMNA_CANTIDAD].sum().unstack(fill_value=0)

    if interaction_matrix.shape[0] < 2:
        return {}, pd.DataFrame()

    lugar_similarity = cosine_similarity(interaction_matrix)
    lugar_sim_df = pd.DataFrame(lugar_similarity, index=interaction_matrix.index, columns=interaction_matrix.index)

    def obtener_recomendaciones(lugar_actual, n=5):
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
        return sorted(recomendaciones.items(), key=lambda x: x[1], reverse=True)[:n]

    recomendaciones_por_ubicacion = {}
    for ubicacion in interaction_matrix.index:
        recomendaciones = obtener_recomendaciones(ubicacion)
        if recomendaciones:
            recomendaciones_por_ubicacion[ubicacion] = recomendaciones

    return recomendaciones_por_ubicacion, interaction_matrix

# =============================================
# PREDICCIÓN DE CANTIDADES
# =============================================
def predecir_cantidades(data, recomendaciones_por_ubicacion):
    ventas_mensuales = data.groupby([COLUMNA_GEOGRAFICA, COLUMNA_PRODUCTO, 'MES_ANO'])[COLUMNA_CANTIDAD].sum().reset_index()
    ventas_mensuales['MES_ANO'] = ventas_mensuales['MES_ANO'].dt.to_timestamp()
    promedio_ventas_producto_global = ventas_mensuales.groupby(COLUMNA_PRODUCTO)[COLUMNA_CANTIDAD].mean().reset_index()
    promedio_ventas_producto_global.rename(columns={COLUMNA_CANTIDAD: 'CANTIDAD_PROMEDIO_GLOBAL'}, inplace=True)

    predicciones_finales = {}
    for ubicacion, productos in recomendaciones_por_ubicacion.items():
        predicciones_finales[ubicacion] = []
        for producto, score in productos:
            historial_local = ventas_mensuales[(ventas_mensuales[COLUMNA_GEOGRAFICA] == ubicacion) & (ventas_mensuales[COLUMNA_PRODUCTO] == producto)].sort_values('MES_ANO')
            cantidad_predicha = DEFAULT_CANTIDAD
            if len(historial_local) >= MIN_MESES_HISTORIA:
                historial_local = historial_local.assign(TIME_INDEX=range(len(historial_local)))
                try:
                    model = LinearRegression()
                    model.fit(historial_local[['TIME_INDEX']], historial_local[COLUMNA_CANTIDAD])
                    pred = model.predict([[len(historial_local)]])[0]
                    cantidad_predicha = max(1, round(pred))
                    p75_local = historial_local[COLUMNA_CANTIDAD].quantile(0.75)
                    if p75_local > 0 and cantidad_predicha > 3 * p75_local:
                        cantidad_predicha = max(DEFAULT_CANTIDAD, round(p75_local))
                except:
                    cantidad_predicha = max(DEFAULT_CANTIDAD, round(historial_local[COLUMNA_CANTIDAD].median()))
            elif len(historial_local) > 0:
                cantidad_predicha = max(DEFAULT_CANTIDAD, round(historial_local[COLUMNA_CANTIDAD].mean()))
            else:
                info_producto = promedio_ventas_producto_global[promedio_ventas_producto_global[COLUMNA_PRODUCTO] == producto]
                if not info_producto.empty:
                    cantidad_predicha = max(DEFAULT_CANTIDAD, round(info_producto['CANTIDAD_PROMEDIO_GLOBAL'].iloc[0]))

            predicciones_finales[ubicacion].append({
                'Producto': producto,
                'Score_Recomendacion': score,
                'Cantidad_Predicha': int(cantidad_predicha)
            })
    return predicciones_finales

# =============================================
# INTERFAZ STREAMLIT
# =============================================
st.title("Recomendación y Predicción de Productos por Ubicación")

data = cargar_datos()
if data is not None:
    st.success("Datos cargados correctamente")
    recomendaciones, matriz = generar_recomendaciones(data)
    if not recomendaciones:
        st.warning("No se pudieron generar recomendaciones. Verifica los datos.")
    else:
        predicciones = predecir_cantidades(data, recomendaciones)
        ubicaciones = list(recomendaciones.keys())
        ubicacion_sel = st.selectbox("Selecciona una ubicación", ubicaciones)
        if ubicacion_sel:
            st.subheader(f"Recomendaciones para {ubicacion_sel}")
            df_pred = pd.DataFrame(predicciones[ubicacion_sel])
            df_pred['Score_Recomendacion'] = df_pred['Score_Recomendacion'].round(4)
            st.dataframe(df_pred)
else:
    st.stop()
