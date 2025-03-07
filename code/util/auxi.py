#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
import os
import math

def listar_datasets(ruta):
    return os.listdir(ruta)

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