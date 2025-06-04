import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

def aplicar_imputacion(serie):

    serie_imputada = serie.copy()
    # Tratar problema de nulos al inicio, interpolate no los puede tratar bien
    parada=False
    j=0
    while(j<len(serie_imputada) and parada!=True):
        valor = serie_imputada.iloc[j]
        if np.isnan(valor):
            serie_imputada.iloc[j] = serie_imputada.mean()
        else:
            parada=True
        j+=1

    # Tratar problema de nulos al final, interpolate no los puede tratar bien
    parada=False
    j=len(serie_imputada)-1
    while(j>=0 and parada!=True):
        valor = serie_imputada.iloc[j]
        if np.isnan(valor):
            serie_imputada.iloc[j] = serie_imputada.mean()
        else:
            parada=True
        j-=1
        
        # Imputamos con interpolacion
    serie_imputada = serie_imputada.interpolate(method='linear')

    return serie_imputada

st.set_page_config(page_title="Imputación", layout="wide")

st.title("Imputador de datos para series temporales")

# Subida de archivo CSV
archivo_csv = st.file_uploader("Suba su serie temporal en formato CSV", type=["csv"])

if archivo_csv is not None:
    # Leer el CSV en un DataFrame
    df = pd.read_csv(archivo_csv, header=0, index_col=0, parse_dates=True)
    st.write("Vista previa de los datos:")
    st.dataframe(df)

    # Periocidad de los datos (por si meto algo de machine learning)
    periocidad = st.selectbox("Periocidad de los datos", ["Segundo", "Minuto", "Hora", "Dia", "Mes", "Año"])

    # Pasar el dataframe a serie
    serie = df.squeeze('columns')
    serie = serie.astype(float)

    # Aplicar imputacion
    serie_imputada = aplicar_imputacion(serie)

    # Devolverlo a dataset para graficar
    df_resultado = pd.DataFrame()
    df_resultado['Date'] = serie_imputada.index
    df_resultado[serie_imputada.name] = serie_imputada.values

    # Selección de columnas para el eje X e Y
    columnas = df_resultado.columns.tolist()
    col_x = columnas[0]
    col_y = columnas[1]

    # Crear gráfica con Plotly
    fig = px.line(df_resultado, x=col_x, y=col_y, title=f"Resultado de la imputación")

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Por favor, sube un archivo CSV para comenzar.")