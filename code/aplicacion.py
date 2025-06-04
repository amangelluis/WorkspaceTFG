import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Visualizador CSV", layout="wide")

st.title("Visualizador de CSV con Streamlit y Plotly")

# Subida de archivo CSV
archivo_csv = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if archivo_csv is not None:
    # Leer el CSV en un DataFrame
    df = pd.read_csv(archivo_csv)
    st.write("Vista previa de los datos:")
    st.dataframe(df)

    # Selección de columnas para el eje X e Y
    columnas = df.columns.tolist()
    col_x = st.selectbox("Selecciona la columna para el eje X", columnas)
    col_y = st.selectbox("Selecciona la columna para el eje Y", columnas)

    # Selección del tipo de gráfica
    tipo_grafica = st.selectbox("Tipo de gráfica", ["Línea", "Barra", "Dispersión"])

    # Crear gráfica con Plotly
    if tipo_grafica == "Línea":
        fig = px.line(df, x=col_x, y=col_y, title=f"{tipo_grafica}: {col_y} vs {col_x}")
    elif tipo_grafica == "Barra":
        fig = px.bar(df, x=col_x, y=col_y, title=f"{tipo_grafica}: {col_y} vs {col_x}")
    elif tipo_grafica == "Dispersión":
        fig = px.scatter(df, x=col_x, y=col_y, title=f"{tipo_grafica}: {col_y} vs {col_x}")

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Por favor, sube un archivo CSV para comenzar.")