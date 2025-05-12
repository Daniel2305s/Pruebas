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
# Cambia el nombre del archivo si ahora es .csv
# Si tu archivo sigue llamándose 'base.xlsx' pero es un CSV, puedes dejarlo,
# pero es mejor renombrarlo a 'base.csv' por claridad.
# data_filename = 'base.xlsx' # Si insistes en mantener la extensión .xlsx para un CSV
data_filename = 'base2.csv' # Recomendado si tu archivo es realmente un CSV

columna_geografica = 'CIUDAD'
columna_producto = 'PRODUCTO'
columna_fecha = 'FECHA'
columna_cantidad = 'CANTIDAD'
min_meses_historia = 2
default_cantidad = 10

# =============================================
# CARGA DE DATOS
# =============================================
st.set_page_config(page_title="Recomendación de Productos", layout="wide")
st.title("📦 Recomendación y Predicción de Productos por Ubicación")

try:
    # Usa pd.read_csv() para archivos separados por comas
    # Si tu archivo CSV usa un separador diferente a la coma (ej. punto y coma),
    # puedes especificarlo con el parámetro sep, ej: pd.read_csv(data_filename, sep=';')
    data_original = pd.read_csv(data_filename)

    columnas_requeridas = [columna_geografica, columna_producto, columna_fecha]
    for col in columnas_requeridas:
        if col not in data_original.columns:
            raise ValueError(f"Columna requerida '{col}' no encontrada en el archivo CSV")

    # El resto de tu lógica de procesamiento de datos debería funcionar igual
    # siempre que las columnas existan y los datos estén en el formato esperado.
    data = data_original[columnas_requeridas + ([columna_cantidad] if columna_cantidad in data_original.columns else [])].copy()
    data.dropna(subset=columnas_requeridas, inplace=True)
    data[columna_fecha] = pd.to_datetime(data[columna_fecha], errors='coerce')
    data.dropna(subset=[columna_fecha], inplace=True)
    data['MES_ANO'] = data[columna_fecha].dt.to_period('M')

    if columna_cantidad in data.columns:
        data[columna_cantidad] = pd.to_numeric(data[columna_cantidad], errors='coerce').fillna(1)
    else:
        data[columna_cantidad] = 1

    st.success("Datos cargados correctamente desde el archivo CSV")

except FileNotFoundError:
    st.error(f"Error al cargar datos: No se encontró el archivo '{data_filename}'. Asegúrate de que está en el mismo directorio que tu script.")
    st.stop()
except ValueError as ve:
    st.error(f"Error en los datos: {ve}")
    st.stop()
except Exception as e:
    st.error(f"Error al cargar datos: {e}")
    st.stop()

# ... (el resto de tu código sigue igual)

# =============================================
# DESCRIPCIÓN Y FILTRO POR CIUDAD
# =============================================
st.markdown("""
### 📘 Descripción Técnica
Este sistema utiliza un enfoque híbrido de recomendación basado en la similitud de consumo entre ciudades 
(utilizando similitud coseno) y un modelo de regresión lineal para predecir la cantidad de unidades recomendadas 
para cada ciudad. El sistema también permite explorar los productos más vendidos por ciudad de forma interactiva.
""")

st.subheader("🔍 Consulta por Ciudad: Productos Más Vendidos")
ciudad_seleccionada = st.selectbox("Selecciona una ciudad para ver sus productos más vendidos:", 
                                   sorted(data[columna_geografica].unique()))

if ciudad_seleccionada:
    top_productos_ciudad = (
        data[data[columna_geografica] == ciudad_seleccionada]
        .groupby(columna_producto)[columna_cantidad]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    top_productos_ciudad.columns = ['Producto', 'Cantidad Vendida']
    st.write(f"Top productos más vendidos en **{ciudad_seleccionada}**:")
    st.dataframe(top_productos_ciudad)

# =============================================
# SISTEMA DE RECOMENDACIÓN
# =============================================
interaction_matrix = data.groupby([columna_geografica, columna_producto])[columna_cantidad].sum().unstack(fill_value=0)

if interaction_matrix.shape[0] >= 2:
    lugar_similarity = cosine_similarity(interaction_matrix)
    lugar_sim_df = pd.DataFrame(lugar_similarity, index=interaction_matrix.index, columns=interaction_matrix.index)
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
        productos_nuevos = productos_lugar_similar[
            (productos_lugar_similar > 0) & (~productos_lugar_similar.index.isin(productos_actuales))
        ].index.tolist()

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
# PREDICCIÓN DE CANTIDADES
# =============================================
ventas_mensuales = data.groupby([columna_geografica, columna_producto, 'MES_ANO'])[columna_cantidad].sum().reset_index()
ventas_mensuales['MES_ANO'] = ventas_mensuales['MES_ANO'].dt.to_timestamp()

promedio_ventas_producto_global = ventas_mensuales.groupby(columna_producto)[columna_cantidad].mean().reset_index()
promedio_ventas_producto_global.rename(columns={columna_cantidad: 'CANTIDAD_PROMEDIO_GLOBAL'}, inplace=True)

predicciones_finales = {}

for ubicacion, productos_recomendados in recomendaciones_por_ubicacion.items():
    predicciones_finales[ubicacion] = []
    for producto, score in productos_recomendados:
        historial_local = ventas_mensuales[(ventas_mensuales[columna_geografica] == ubicacion) &
                                           (ventas_mensuales[columna_producto] == producto)].sort_values('MES_ANO')

        cantidad_predicha = default_cantidad
        meses_historicos_locales = len(historial_local)

        if meses_historicos_locales >= min_meses_historia:
            historial_local = historial_local.assign(TIME_INDEX=range(len(historial_local)))
            try:
                model = LinearRegression()
                model.fit(historial_local[['TIME_INDEX']], historial_local[columna_cantidad])
                pred = model.predict([[len(historial_local)]])[0]
                cantidad_predicha = max(1, round(pred))
                p75_local = historial_local[columna_cantidad].quantile(0.75)
                if p75_local > 0 and cantidad_predicha > 3 * p75_local:
                    cantidad_predicha = max(default_cantidad, round(p75_local))
            except:
                cantidad_predicha = max(default_cantidad, round(historial_local[columna_cantidad].median()))
        elif meses_historicos_locales > 0:
            cantidad_predicha = max(default_cantidad, round(historial_local[columna_cantidad].mean()))
        else:
            info_producto_otras_ubic = promedio_ventas_producto_global[
                promedio_ventas_producto_global[columna_producto] == producto]
            if not info_producto_otras_ubic.empty:
                cantidad_promedio_global_producto = info_producto_otras_ubic['CANTIDAD_PROMEDIO_GLOBAL'].iloc[0]
                cantidad_predicha = max(default_cantidad, round(cantidad_promedio_global_producto))

        predicciones_finales[ubicacion].append({
            'Producto': producto,
            'Score_Recomendacion': score,
            'Cantidad_Predicha': int(cantidad_predicha)
        })

# =============================================
# RESULTADOS FINALES
# =============================================
st.header("📊 Resultados Finales de Recomendación")
for ubicacion_res, predicciones_list in predicciones_finales.items():
    st.subheader(f"📍 {ubicacion_res}")
    if predicciones_list:
        df_res = pd.DataFrame(predicciones_list)
        df_res['Score_Recomendacion'] = df_res['Score_Recomendacion'].round(4)
        st.dataframe(df_res[['Producto', 'Score_Recomendacion', 'Cantidad_Predicha']])
    else:
        st.write("No hay predicciones para esta ubicación.")
