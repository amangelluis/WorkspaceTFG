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

def RMSE(serie_original, serie_nula, serie_imputada):
    suma=0
    for i in serie_nula[serie_nula.isna()].index:
        suma += (serie_original[i] - serie_imputada[i])**2
    suma = math.sqrt(suma/len(serie_original))
    return suma

def MAE(serie_original, serie_nula, serie_imputada):
    suma=0
    for i in serie_nula[serie_nula.isna()].index:
        suma += abs(serie_original[i] - serie_imputada[i])
    suma = suma/len(serie_original)
    return suma

def MAPE(serie_original, serie_nula, serie_imputada):
    suma=0
    for i in serie_nula[serie_nula.isna()].index:
        if(serie_original[i] != 0):
            suma += float(abs(serie_original[i] - serie_imputada[i]))/((serie_original[i]+serie_imputada[i])/2)
    suma = suma/len(serie_original)
    return 100*suma

def BIAS(serie_original, serie_nula, serie_imputada):
    suma=0
    for i in serie_nula[serie_nula.isna()].index:
        suma += serie_original[i] - serie_imputada[i]
    suma = suma/len(serie_original)
    return suma

def aplicar_escalado(serie, escala):
    valores_escalados = escala.transform(serie.values.reshape(-1,1))
    serie_escalada = serie.copy()
    for i in range(len(serie)):
        serie_escalada.iloc[i] = valores_escalados[i]
    return serie_escalada

def desaplicar_escalado(serie_escalada, escala):
    valores_originales = escala.inverse_transform(serie_escalada.values.reshape(-1,1))
    serie_original = serie_escalada.copy()
    for i in range(len(serie_escalada)):
        serie_original.iloc[i] = valores_originales[i]
    return serie_original

def escalar(escala, lista):
    lista_resultante = []
    for serie in lista:
        # Escalamos
        serie = aplicar_escalado(serie, escala)
        lista_resultante.append(serie)
    return lista_resultante

def desescalar(escala, lista):
    lista_resultante = []
    for serie in lista:
        # Desescalamos
        serie = desaplicar_escalado(serie, escala)
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
        ax.set_title('Comparacion entre series')

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
    lista_test = []
    for serie in lista:
        entrenamiento_serie = serie[serie.isna() == False]
        test_serie = serie[serie.isna()]
        lista_train.append(entrenamiento_serie)
        lista_test.append(test_serie)
    return lista_train, lista_test

def train_test_individual(serie):
    entrenamiento_serie = serie[serie.isna() == False]
    test_serie = serie[serie.isna()]
    return entrenamiento_serie, test_serie
