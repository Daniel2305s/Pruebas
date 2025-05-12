import streamlit as st
import pandas as pd

# =============================================
# CONFIGURACIÓN DE PARÁMETROS
# =============================================
# Asegúrate que este es el nombre correcto de tu archivo Excel
# y que contiene las columnas finales que quieres mostrar.
excel_filename = 'basesi2.xlsx'

# Asegúrate que estos nombres coinciden EXACTAMENTE con los encabezados
# de las columnas en tu archivo Excel.
columna_ciudad = 'CIUDAD'
columna_producto = 'PRODUCTO'
columna_fecha = 'FECHA'
columna_score = 'SCORE_RECOMENDACION'
columna_predicha = 'CANTIDAD_PREDICHA'

# Lista de columnas que quieres que se muestren en la tabla
columnas_a_mostrar = [
    columna_ciudad,
    columna_producto,
    columna_fecha,
    columna_score,
    columna_predicha
]

# =============================================
# CARGA Y MUESTRA DE DATOS
# =============================================
st.set_page_config(page_title="Visualizador de Datos", layout="wide")
st.title("📊 Visualizador de Datos desde Excel")
st.write(f"Mostrando datos desde el archivo: **{excel_filename}**")

try:
    # Lee el archivo Excel directamente
    # Puedes añadir dtype={'columna_texto': str} si alguna columna numérica se lee mal
    data_existente = pd.read_excel(excel_filename, engine='openpyxl')

    # Verifica que las columnas que queremos mostrar existan en el archivo
    columnas_faltantes = [col for col in columnas_a_mostrar if col not in data_existente.columns]
    if columnas_faltantes:
        st.error(f"Error: Las siguientes columnas no se encontraron en el archivo '{excel_filename}': {', '.join(columnas_faltantes)}")
        st.stop() # Detiene la ejecución si faltan columnas clave

    st.success(f"Datos cargados correctamente.")

    # --- Filtro por Ciudad ---
    st.subheader("🔍 Filtrar por Ciudad")
    # Obtiene lista única de ciudades, maneja posibles NaN convirtiéndolos a string y luego filtrándolos si es necesario
    lista_ciudades = sorted([str(c) for c in data_existente[columna_ciudad].unique() if pd.notna(c)])

    # Crea el menú desplegable con la opción "Todas" al principio
    ciudad_seleccionada = st.selectbox(
        "Selecciona una ciudad para filtrar (o 'Todas' para ver todo):",
        options=["Todas"] + lista_ciudades
    )

    # --- Mostrar Datos ---
    st.subheader("📄 Datos Cargados")

    if ciudad_seleccionada == "Todas":
        # Muestra todas las filas, pero solo las columnas seleccionadas
        st.dataframe(data_existente[columnas_a_mostrar])
    else:
        # Filtra el DataFrame por la ciudad seleccionada y muestra las columnas seleccionadas
        data_filtrada = data_existente[data_existente[columna_ciudad] == ciudad_seleccionada]
        st.dataframe(data_filtrada[columnas_a_mostrar])

    st.info(f"Mostrando {data_filtrada.shape[0] if ciudad_seleccionada != 'Todas' else data_existente.shape[0]} filas.")

except FileNotFoundError:
     st.error(f"❌ Error Fatal: No se encontró el archivo '{excel_filename}'.")
     st.error("Por favor, asegúrate de que el archivo Excel esté en la misma carpeta que este script de Python.")
except Exception as e:
    st.error(f"❌ Ocurrió un error inesperado al procesar el archivo:")
    st.exception(e) # Muestra detalles técnicos del error