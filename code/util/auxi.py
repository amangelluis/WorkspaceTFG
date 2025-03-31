import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import os
import math
import sklearn.preprocessing

def listar_datasets(ruta):
    return os.listdir(ruta)

def leer_series(ruta):
    lista_nombres = listar_datasets(ruta)
    lista_series = []
    for i in lista_nombres:
        # Lo leemos
        df = pd.read_csv(ruta+i, header=0, index_col=0, parse_dates=True)
        serie = df.squeeze('columns')
        serie.name = i
        lista_series.append(serie)
    return lista_series

def RMSE(serie_original, serie_imputada):
    suma=0
    for i in serie_original.index:
        suma += (serie_original[i] - serie_imputada[i])**2
    suma = math.sqrt(suma/len(serie_original))
    return suma

def MAE(serie_original, serie_imputada):
    suma=0
    for i in serie_original.index:
        suma += abs(serie_original[i] - serie_imputada[i])
    suma = suma/len(serie_original)
    return suma

def MAPE(serie_original, serie_imputada):
    suma=0
    for i in serie_original.index:
        if(serie_original[i] != 0):
            suma += float(abs(serie_original[i] - serie_imputada[i]))/((serie_original[i]+serie_imputada[i])/2)
    suma = suma/len(serie_original)
    return 100*suma

def BIAS(serie_original, serie_imputada):
    suma=0
    for i in serie_original.index:
        suma += serie_original[i] - serie_imputada[i]
    suma = suma/len(serie_original)
    return suma

def aplicar_escalado(serie, escala):
    valores_escalados = escala.transform(serie.values.reshape(-1,1))
    serie_escalada = serie.copy()
    for i in range(len(serie)):
        serie_escalada.iloc[i] = valores_escalados[i]
    return serie_escalada

def aplicar_escalado_matrix(serie, escala):
    valores_escalados = escala.transform(serie.values)
    return valores_escalados

def desaplicar_escalado(serie_escalada, escala):
    valores_originales = escala.inverse_transform(serie_escalada.values.reshape(-1,1))
    serie_original = serie_escalada.copy()
    for i in range(len(serie_escalada)):
        serie_original.iloc[i] = valores_originales[i]
    return serie_original

def escalar(lista_entrenamiento, lista_test, funcion_escala):
    lista_entrenamiento_resultante = []
    lista_test_resultante = []
    lista_scalers = []
    for i in range(0,len(lista_entrenamiento)):
        #Entrenamos el scaler
        scaler = funcion_escala.fit(lista_entrenamiento[i].values.reshape(-1,1))
        # Escalamos
        serie_entrenamiento_escalada = aplicar_escalado(lista_entrenamiento[i], scaler)
        serie_test_escalada = aplicar_escalado(lista_test[i], scaler)
        lista_entrenamiento_resultante.append(serie_entrenamiento_escalada)
        lista_test_resultante.append(serie_test_escalada)
        lista_scalers.append(scaler)
    return lista_entrenamiento_resultante, lista_test_resultante, lista_scalers

def escalar_matrix(lista_entrenamiento, lista_test, funcion_escala):
    lista_entrenamiento_resultante = []
    lista_test_resultante = []
    lista_scalers = []
    for i in range(0,len(lista_entrenamiento)):
        #Entrenamos el scaler
        scaler = funcion_escala.fit(lista_entrenamiento[i].values)
        # Escalamos
        serie_entrenamiento_escalada = aplicar_escalado_matrix(lista_entrenamiento[i], scaler)
        serie_test_escalada = aplicar_escalado_matrix(lista_test[i], scaler)
        lista_entrenamiento_resultante.append(serie_entrenamiento_escalada)
        lista_test_resultante.append(serie_test_escalada)
        lista_scalers.append(scaler)
    return lista_entrenamiento_resultante, lista_test_resultante, lista_scalers

def desescalar(lista_scalers, lista):
    lista_resultante = []
    for i in range(0,len(lista)):
        # Desescalamos
        serie = desaplicar_escalado(lista[i], lista_scalers[i])
        lista_resultante.append(serie)
    return lista_resultante

def graficar_comparacion(serie_original, lista):

    for serie in lista:

        fig, ax = plt.subplots(figsize=(20, 12))

        ax.plot(serie_original.index, serie_original.values, label=serie_original.name)
        ax.plot(serie.index, serie.values, label=serie.name)

        # Add a legend and labels
        plt.legend()
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Valor')
        ax.set_title('Comparacion entre '+serie_original.name+' y '+serie.name)

        # Rotar las etiquetas del eje x para mejor legibilidad
        plt.xticks(rotation=45)

        # Display the plot
        plt.tight_layout()
        plt.show()

def guardar_series(ruta, lista):
    for serie in lista:
        serie.to_csv(ruta+serie.name)

def train_test(lista):
    lista_train = []
    lista_imputar = []
    for serie in lista:
        entrenamiento_serie = serie[serie.isna() == False] # Los datos con los que vamos a entrenar el modelo
        imputar_serie = serie[serie.isna()]  # Los datos con los que vamos a usar el modelo (todos los na) para imputar
        lista_train.append(entrenamiento_serie)
        lista_imputar.append(imputar_serie)
    return lista_train, lista_imputar

def train_test_matrix(lista):
    lista_train = []
    lista_imputar = []
    for serie in lista:
        entrenamiento_serie = serie[serie[serie.columns[0]].isna() == False] # Los datos con los que vamos a entrenar el modelo
        imputar_serie = serie[serie[serie.columns[0]].isna()]  # Los datos con los que vamos a usar el modelo (todos los na) para imputar
        lista_train.append(entrenamiento_serie)
        lista_imputar.append(imputar_serie)
    return lista_train, lista_imputar

def obtencion_datos_test_final(lista, serie_og):
    lista_test_final = []
    for serie in lista:
        test_final_serie = serie_og[serie.isna()]   # Los datos originales que vamos a tratar de replicar
        lista_test_final.append(test_final_serie)
    return lista_test_final

def obtencion_datos_test_final_matrix(lista, serie_og):
    lista_test_final = []
    for serie in lista:
        test_final_serie = serie_og[serie[serie.columns[0]].isna()]   # Los datos originales que vamos a tratar de replicar
        lista_test_final.append(test_final_serie)
    return lista_test_final
